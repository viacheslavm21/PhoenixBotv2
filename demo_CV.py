import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]

#import sys
#sys.path.append('vision-main/references/detection')
#import transforms, utils, engine, train
#from utils import collate_fn

import pyrealsense2 as rs
import cv2, numpy as np, matplotlib.pyplot as plt, pandas as pd, urx, json

from PIL import Image
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.transforms import functional as F

from scipy.spatial.transform import Rotation as R
from threading import Thread

import torchvision.models.detection.mask_rcnn
import random
import time
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from insertion_utils import triangulate
import data.experiments.insertion.input_points as pts

from models.ResNet50_model import get_model as get_ResNet50
from models.detectron2_MaskRCNN import get_model as get_MaskRCNN

from insertion_utils import plane


class ResNetCV():
    def __init__(self):
        # configure stream pipeline:
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # Start streaming
        #self.pipeline.start(self.config)
        prof = self.pipeline.start(self.config)
        EXPOSURE_VALUE = 100
        s = prof.get_device().query_sensors()[1]
        s.set_option(rs.option.exposure, EXPOSURE_VALUE)

        # Get device product line for setting a supporting resolution
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in self.device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The program requires Depth camera with Color sensor")
            exit(0)

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if self.device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # initalize model
        self.NUM_KEYPOINTS = 19
        #self.device = torch.device('cpu')
        try:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        except Exception as e:
            print(e, 'device: cpu')


        self.rough_model = get_MaskRCNN(weights_path='data/outputs/MaskRCNN_model_0164999.pth', device=0)
        self.rough_model_metadata = MetadataCatalog.get('CBR1_train')

        self.precise_model = get_ResNet50(num_keypoints=self.NUM_KEYPOINTS, weights_path='data/outputs/keypointsrcnn_weights_6_epoch260.pth', load_device=self.device)
        self.precise_model.to(self.device)

        # configure robot

        ROBOT_IP = "192.168.88.100"
        self.robot = urx.Robot(ROBOT_IP, use_rt=True)
        self.base_coord_sys = self.robot.csys
        ROBOT_IP = "192.168.88.200"
        self.UR3 = urx.Robot(ROBOT_IP, use_rt=True)

        # calibrate camera
        self.set_manual_calibration()

    def set_manual_calibration(self):
        self.mtx = [[638.,   0.,         635.35739312],
                     [  0.,         634.88634125, 354.91528429],
                     [  0.,           0.,           1.        ]]
        self.t_cam2gripper = [[-23.],
                              [107.],
                              [26.]]

        self.R_cam2gripper = [[-0.99926641, -0.03782104, 0.00526198],
                              [0.03777663, -0.99922319, -0.00994069],
                              [0.0056323, -0.00973294, 0.99992018]]

    def calibrate(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((8 * 5, 3), np.float32)

        # step size of the checkboard is 100 mm
        objp[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2) * 100

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        ids = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,22]

        imagas = [f'data/calib_horizontal/frame{i}.jpg' for i in ids]
        datas = [f'data/calib_horizontal/meta{i}.json' for i in ids]

        images = imagas

        def check_calib_frames(frames):
            for fname in frames:
                img = cv2.imread(fname)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Find the chess board corners
                ret, corners = cv2.findChessboardCorners(gray, (8, 5), None)
                print(fname)
                print(len(corners))
            rvecs = []
            tvecs = []
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corn = []
            grip2base = []
            for fname in frames:
                img = cv2.imread(fname)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (8, 5), None)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                corn.append(np.array([corners2[0], corners2[1], corners2[8], corners2[9]], dtype=np.float32))
                #     ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
                #     imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
                #     visualise pixels
                """        pr0 = np.array(corners2[0], dtype=int)
                pr1 = np.array(corners2[1], dtype=int)
                pr2 = np.array(corners2[38], dtype=int)
                pr3 = np.array(corners2[39], dtype=int)
                img = cv.circle(img, (pr0[0, 0], pr0[0, 1]), radius=5, color=(255, 0, 0))
                img = cv.circle(img, (pr1[0, 0], pr1[0, 1]), radius=5, color=(255, 0, 0))
                img = cv.circle(img, (pr2[0, 0], pr2[0, 1]), radius=5, color=(255, 0, 0))
                img = cv.circle(img, (pr3[0, 0], pr3[0, 1]), radius=5, color=(255, 0, 0))"""
                for _ in range(40):
                    pr = np.array(corners2[_], dtype=int)
                    img = cv2.circle(img, (pr[0, 0], pr[0, 1]), radius=5, color=(255, 0, 0))

                cv2.imshow(fname, img)
                k = cv2.waitKey(0) & 0xFF
        #check_calib_frames(images)

        dataposes_rot = []  # set of poses from robot tcp
        dataposes_shift = []

        for dname in datas:
            with open(dname) as json_file:
                json_data = json.load(json_file)
                # by Slava:
                r = R.from_rotvec(np.array(json_data['l'][3:])).as_matrix()
                dataposes_rot.append(r)
                mm_data = np.array(json_data['l'][:3]) * 1000
                dataposes_shift.append(mm_data)

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (8, 5), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        rvecs = []
        tvecs = []

        for fname in imagas:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (8, 5), None)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            #print(corners2)
            ret, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)
            rvecs.append(rvec)
            tvecs.append(tvec)

        self.R_cam2gripper, self.t_cam2gripper = cv2.calibrateHandEye(R_gripper2base=dataposes_rot,
                                                                   t_gripper2base=dataposes_shift,
                                                                   R_target2cam=rvecs,
                                                                   t_target2cam=tvecs,
                                                                   method=cv2.CALIB_HAND_EYE_ANDREFF)
        self.dataposes_rot = dataposes_rot
        self.mtx = mtx

        print("intrinsic matrix from images")
        print(mtx)

        print("camera in gripper coordinates:")
        print(self.t_cam2gripper)

        print("rotation vector:")
        print(R.from_matrix(self.R_cam2gripper).as_rotvec())

    def predict_corners(self):
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            # predict and visualize predictions
            outputs = self.rough_model(
                color_image)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

            try:
                corner_keypoints = np.concatenate(np.asarray(outputs["instances"]._fields['pred_keypoints'].cpu()[0][:, :2]))
                return corner_keypoints
            except:
                print("No keypoints detected")
                continue

    def predict_socket_points(self):
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            color_image_tensor = [F.to_tensor(color_image)]
            color_image_tensors = list(image.to(self.device) for image in color_image_tensor)

            with torch.no_grad():
                self.precise_model.to(self.device)
                self.precise_model.eval()
                output = self.precise_model(color_image_tensors)

            scores = output[0]['scores'].detach().cpu().numpy()

            high_scores_idxs = np.where(scores > 0.7)[0].tolist()  # Indexes of boxes with scores > 0.7
            post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs],
                                                output[0]['scores'][high_scores_idxs],
                                                0.3).cpu().numpy()  # Indexes of boxes left after applying NMS (iou_threshold=0.3)

            keypoints = []
            for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                keypoints.append([list(map(int, kp[:2])) for kp in kps])
            if len(keypoints[0]) != 19:
                print('Not all keypoints are seen')
                continue
            flattened_keypoints = [coordiante for coordiantes in keypoints[0] for coordiante in coordiantes]
            return flattened_keypoints

    def estimate_pose_precise(self, rough_socket_pose, method='triangulation'):
        # firstly, I want decide, how to make movements...
        # I will move around estimated pose of the socket....
        # I will start my journey from rotational movement
        rough_socket_rotation_in_base = R.from_rotvec(rough_socket_pose[3:]).as_matrix()
        robot_pose = np.asarray(self.robot.getl())

        look_positions_in_socket = [[0.05, 0, 0],
                                    [0.0, 0.1, -0.04],
                                    [-0.065, 0, 0],
                                    [0, -0.05, 0.03]]

        look_angles_in_socket = [[0, 30, 0],
                                 [-25, 0, 0],
                                 [0, -30, 0],
                                 [20, 0, 0]]

        frame_tcp0 = list([])
        frame_points = list([])

        for look in zip(look_positions_in_socket, look_angles_in_socket):
            rotation_in_socket = R.from_euler('xyz', look[1], degrees=True).as_matrix()
            position_in_socket = np.asarray(look[0])
            # calculate rotation
            rotation_in_base = np.asarray(rough_socket_rotation_in_base) @ rotation_in_socket.T
            rotvec_in_base = R.from_matrix(rotation_in_base).as_rotvec()
            # calculate position
            position_in_base = np.asarray(rough_socket_rotation_in_base) @ position_in_socket.T + rough_socket_pose[:3]
            self.robot.movel(np.concatenate((position_in_base, rotvec_in_base)), acc=0.15, vel=0.25)
            time.sleep(5)
            frame_tcp0.append(self.robot.get_pose().pose_vector)
            frame_points.append(list([]))
            look_points = []
            for _ in range(5):
                time.sleep(0.1)
                while True:
                    try:
                        look_points.append(self.predict_socket_points())
                        #frame_points[-1] = np.append(frame_points[-1], self.predict_socket_points())
                        #print(self.predict_socket_points())
                        break
                    except Exception as e:
                        print(e)
                        continue
                frame_points[-1] = np.mean(look_points, axis=0)

        combinations = []
        #print(frame_points)

        for _ in range(20):
            while True:
                idx = random.sample(range(len(frame_points)), 2)
                if abs(idx[0] - idx[1]) > 2:  # we don't want to take close frames as combinations
                    break
            combinations.append(idx)

        # run combinations
        combinations = [[0, 1],
                        [1, 2],
                        [2, 3],
                        [3, 0],
                        [1, 3],
                        [0, 2]]

        points_in3d_robot_base = []
        for combination in combinations:
            if method == 'triangulation':
                points_in3d_robot_base.append(triangulate(frame_points[combination[0]], frame_points[combination[1]],
                                              frame_tcp0[combination[0]], frame_tcp0[combination[1]],
                                              self.R_cam2gripper, self.t_cam2gripper, self.mtx))
                points_in3d_robot_base[-1] = points_in3d_robot_base[-1]/1000

            else:
                raise ValueError(f'Sorry, method {method} is not supported yet.')

        points_in3d_robot_base = np.asarray(points_in3d_robot_base)
        points = np.mean(points_in3d_robot_base, axis=0)
        std = np.std(points_in3d_robot_base, axis=0)
        #print(f'points before idx {len(points)}')
        #print(points[4])
        #print('Keypoint positions estimated. ')
        #print('Mean: ')
        #print(points)
        #print('Std: ')
        #print(std)
        valid_idx = [2, 3, 4, 5,
                     7, 8,
                     10, 11, 12, 13,
                     14]
        points = points[valid_idx]
        plane_combinations = [[10, 3, 0],
                              [6, 8, 1],
                              [4, 9, 3],
                              [5, 0, 6],
                              [7, 9, 2]]
        direction_combinations = [[0, 3], [6, 9]]



        x_from_combinations = []
        y_from_combinations = []
        z_from_combinations = []

        for direction_idx in direction_combinations:
            x = points[direction_idx[0]] - points[direction_idx[1]]
            x = x / np.linalg.norm(x)
            x_from_combinations.append(x)

        x = np.mean(x_from_combinations, axis=0)
        #print(f'mean x: {x}')

        for plane_idx in plane_combinations:
            my_plane = plane(points[plane_idx[0]], points[1], points[0])
            z = np.asarray([my_plane.a, my_plane.b, my_plane.c])
            z = z / np.linalg.norm(z)
            #print(f'z for combination: {z}')
            y = np.cross(z, x)
            x = np.cross(y, z)
            y_from_combinations.append(y)
            z_from_combinations.append(z)

        y = np.mean(y_from_combinations, axis=0)
        #print(f'mean y: {y}')

        z = np.mean(z_from_combinations, axis=0)
        #print(f'mean z: {z}')

        centerpoint = np.mean(points, axis=0)

        refpoint = np.asarray([x, y, z]).T @ [[0.02], [-0.08], [-0.12]]
        refpoint = refpoint.T + centerpoint
        refpoint = refpoint[0]

        r = R.from_matrix(np.asarray([x, y, z]).T).as_rotvec()
        reference_pose = np.concatenate((refpoint, r))
        #print(f'estimated reference pose {reference_pose}')
        return reference_pose


        # back...
        self.robot.movel(rough_socket_pose)
        # and to the precise pose

    def estimate_position_rough(self, method='triangulation', from_where='fixed_points'):
        if from_where=='fixed_points':
            poses = [[0.3873028216882321, 0.28819162279248414, 0.6094467790964786, 1.3694276429552872, 0.9751484335261905, 0.992689301046317],
                     [0.38731271938153067, -0.3656970666638562, 0.6094022842797057, 1.0260766857347172, 1.5056316098530242, 1.505089150682654],
                     [0.3873197718355029, -0.045470497332361234, 1.0220643746903034, 1.7259847218692594, 1.6541910076297102, 0.7794939580177054],
                     [0.38728422499533627, -0.04546660606221704, 0.29545275427915358, 0.9947059189802091, 0.9236223266753295, 1.351278614783244]]

            poses[0][1] = poses[0][1] + random.uniform(-0.08, 0.1)
            poses[1][1] = poses[1][1] + random.uniform(-0.08, 0.1)
            poses[2][2] = poses[2][2] + random.uniform(-0.08, 0.1)
            poses[3][2] = poses[3][2] + random.uniform(-0.02, 0.15)

            frame_tcp0 = []
            frame_points = []

            # frames collection:
            for pose in poses:
                self.robot.movel(pose, acc=0.2, vel=0.25)
                time.sleep(5)
                frame_tcp0.append(self.robot.get_pose().pose_vector)
                frame_points.append([])
                for _ in range(5):
                    time.sleep(0.1)
                    while True:
                        try:
                            frame_points[-1].append(self.predict_corners())
                            #print(frame_points[-1][-1])
                            break
                        except:
                            continue

                frame_points[-1] = np.mean(frame_points[-1],axis=0)
            # run combinations
            combinations = [[0, 1],
                            [1, 2],
                            [2, 3],
                            [3, 0],
                            [1, 3],
                            [0, 2]]
            combinations = []

            for _ in range(20):
                while True:
                    idx = random.sample(range(len(frame_points)), 2)
                    if abs(idx[0] - idx[1]) > 2:  # we don't want to take close frames as combinations
                        break
                combinations.append(idx)
        else:
            raise ValueError("Wrong value for 'from_where'.")

        points_in3d_robot_base = []
        for combination in combinations:
            if method == 'triangulation':
                points_in3d_robot_base.append(triangulate(frame_points[combination[0]], frame_points[combination[1]],
                                              frame_tcp0[combination[0]], frame_tcp0[combination[1]],
                                              self.R_cam2gripper, self.t_cam2gripper, self.mtx))
                points_in3d_robot_base[-1] = points_in3d_robot_base[-1]/1000

            else:
                raise ValueError(f'Sorry, method {method} is not supported yet.')

        std = np.std(points_in3d_robot_base, axis=0)
        points_in3d_robot_base = np.mean(points_in3d_robot_base, axis=0)
        #print('Corners position estimated. ')
        #print('Mean: ')
        #print(points_in3d_robot_base)
        #print('Std: ')
        #print(std)

        points = np.asarray(points_in3d_robot_base)
        centerpoint = np.mean(points, axis=0)
        my_plane = plane(points[2], points[1], points[0])
        z = np.asarray([my_plane.a, my_plane.b, my_plane.c])
        z = z / np.linalg.norm(z)
        x = points[0]-points[1]
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        x = np.cross(y, z)

        refpoint = np.asarray([x, y, z]).T @ [[0.02], [-0.08], [-0.12]]
        refpoint = refpoint.T + centerpoint
        refpoint = refpoint[0]

        r = R.from_matrix(np.asarray([x, y, z]).T).as_rotvec()
        reference_pose = np.concatenate((refpoint, r))
        return reference_pose

    def move_to_ref(self, reference_pose):
        #print(f'Moving to {reference_pose}')
        self.robot.movel(reference_pose, acc=0.18, vel=0.28)

    def draw_precise_keypoints(self, image, bboxes, keypoints, image_original=None, bboxes_original=None, keypoints_original=None):
        fontsize = 2
        for bbox in bboxes:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            image = cv2.rectangle(image.copy(), start_point, end_point, (0, 255, 0), 2)

        for kps in keypoints:
            for idx, kp in enumerate(kps):
                image = cv2.circle(image.copy(), tuple(kp), 2, (255, 0, 0), 2)

        return image

    def get_pose_error(self,ur3_pose_id):
        pose_in_ref = self.robot.getl()
        r = R.from_rotvec(pose_in_ref[3:])
        pose_in_ref[3:] = r.as_euler('xyz', degrees=False)
        pose_in_ref = np.asarray(pose_in_ref)
        pose_error = pose_in_ref - pts.true_poses[ur3_pose_id]

        pose_error_xyz_in_tcp = np.asarray(R.from_rotvec(self.robot.getl()[3:]).as_matrix()) @ pose_error[:3]
        pose_error_rpy_in_tcp = np.asarray(R.from_rotvec(self.robot.getl()[3:]).as_matrix()) @ pose_error[3:]

        pose_error_in_tcp = np.concatenate((pose_error_xyz_in_tcp, pose_error_rpy_in_tcp))

        return pose_error_in_tcp

    @torch.no_grad()
    def visualize_precise(self):
        try:
            cv2.namedWindow('Inference', cv2.WINDOW_AUTOSIZE)
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

                color_image_tensor = [F.to_tensor(color_image)]
                color_image_tensors = list(image.to(self.device) for image in color_image_tensor)
                #color_image_tensor = color_image_tensor.float()
                #print(color_image_tensor.size())
                with torch.no_grad():
                    self.precise_model.to(self.device)
                    self.precise_model.eval()
                    output = self.precise_model(color_image_tensors)

                image = (color_image_tensors[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                scores = output[0]['scores'].detach().cpu().numpy()

                high_scores_idxs = np.where(scores > 0.7)[0].tolist()  # Indexes of boxes with scores > 0.7
                post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs],
                                                    output[0]['scores'][high_scores_idxs],
                                                    0.3).cpu().numpy()  # Indexes of boxes left after applying NMS (iou_threshold=0.3)

                keypoints = []
                for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                    keypoints.append([list(map(int, kp[:2])) for kp in kps])

                bboxes = []
                for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                    bboxes.append(list(map(int, bbox.tolist())))

                color_image = self.draw_precise_keypoints(image, bboxes, keypoints)

                cv2.imshow('Inference', color_image)

                cv2.waitKey(1)
                #time.sleep(3)
        finally:
            # Stop streaming
            self.pipeline.stop()

    def visualize_by_points(self):
        if self.robot or not self.robot:
            try:
                n = 0
                cv2.namedWindow('Inference', cv2.WINDOW_AUTOSIZE)
                # cv2.setMouseCallback('Inference', self.mouse_callback)
                while True:
                    #print(self.robot.getl()[2])

                    # Wait for a coherent pair of frames: depth and color
                    frames = self.pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue

                    # Convert images to numpy arrays
                    color_image = np.asanyarray(color_frame.get_data())
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

                    if self.robot.getl()[2] > 0.5:
                        # predict and visualize predictions ROUGH
                        outputs = self.rough_model(
                            color_image)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
                        v = Visualizer(color_image[:, :, ::-1],
                                       metadata=self.rough_model_metadata,
                                       scale=1.0,
                                       )
                        try:
                            self.nkpts = np.concatenate(
                                np.asarray(outputs["instances"]._fields['pred_keypoints'].cpu()[0][:, :2]))
                            frame1_points = self.nkpts
                            corn = [[frame1_points[0], frame1_points[1]],
                                    [frame1_points[2], frame1_points[3]],
                                    [frame1_points[4], frame1_points[5]],
                                    [frame1_points[6], frame1_points[7]]]
                            middle = np.mean(corn, axis=0).astype(int)

                            self.move_vec = middle - [640, 360]

                            self.z_rot = corn[0][1] - corn[1][1]

                        except:
                            print("No keypoints detected")

                        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                        # draw inference or raw image
                        cv2.imshow('Inference', out.get_image()[:, :, ::-1])
                    else:

                        color_image_tensor = [F.to_tensor(color_image)]
                        color_image_tensors = list(image.to(self.device) for image in color_image_tensor)
                        with torch.no_grad():
                            self.precise_model.to(self.device)
                            self.precise_model.eval()
                            output = self.precise_model(color_image_tensors)

                        image = (color_image_tensors[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                        scores = output[0]['scores'].detach().cpu().numpy()

                        high_scores_idxs = np.where(scores > 0.7)[0].tolist()  # Indexes of boxes with scores > 0.7
                        post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs],
                                                            output[0]['scores'][high_scores_idxs],
                                                            0.3).cpu().numpy()  # Indexes of boxes left after applying NMS (iou_threshold=0.3)

                        keypoints = []
                        for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                            keypoints.append([list(map(int, kp[:2])) for kp in kps])

                        bboxes = []
                        for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                            bboxes.append(list(map(int, bbox.tolist())))

                        color_image = self.draw_precise_keypoints(image, bboxes, keypoints)
                        cv2.imshow('Inference', color_image)

                    # cv.imshow('Demo', color_image)

                    cv2.waitKey(1)
                    # time.sleep(3)
            finally:
                # Stop streaming
                self.pipeline.stop()

    @torch.no_grad()
    def visualize_rough(self):
        try:
            n = 0
            cv2.namedWindow('Inference', cv2.WINDOW_AUTOSIZE)
            #cv2.setMouseCallback('Inference', self.mouse_callback)
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

                # predict and visualize predictions
                outputs = self.rough_model(
                    color_image)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
                v = Visualizer(color_image[:, :, ::-1],
                               metadata=self.rough_model_metadata,
                               scale=1.0,
                               )
                try:
                    self.nkpts = np.concatenate(np.asarray(outputs["instances"]._fields['pred_keypoints'].cpu()[0][:, :2]))
                    frame1_points = self.nkpts
                    corn =  [[frame1_points[0], frame1_points[1]],
                             [frame1_points[2], frame1_points[3]],
                             [frame1_points[4], frame1_points[5]],
                             [frame1_points[6], frame1_points[7]]]
                    middle = np.mean(corn, axis=0).astype(int)

                    self.move_vec = middle - [640, 360]

                    self.z_rot = corn[0][1] - corn[1][1]

                except:
                    print("No keypoints detected")


                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

                # draw inference or raw image
                cv2.imshow('Inference', out.get_image()[:, :, ::-1])
                #cv.imshow('Demo', color_image)

                cv2.waitKey(1)
                #time.sleep(3)
        finally:
            # Stop streaming
            self.pipeline.stop()

    def move_back_exp(self):
        base_coord_sys = self.robot.csys
        self.robot.set_csys(self.robot.get_pose())
        # for safety reasons, let's not leave tool inside the plug
        self.robot.translate((0, 0, -0.25), acc=0.2, vel=0.25)
        self.robot.set_csys(base_coord_sys)


    # pose estimators: return points_in_3d_robot_base

go = ResNetCV()
stream = Thread(target=go.visualize_by_points)
stream.start()

