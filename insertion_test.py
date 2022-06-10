import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]

#import sys
#sys.path.append('vision-main/references/detection')
#import transforms, utils, engine, train
#from utils import collate_fn

import pyrealsense2 as rs
import cv2, numpy as np, matplotlib.pyplot as plt, pandas as pd, urx, json, os, pickle

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
#import data.experiments.insertion.input_points as pts

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
        EXPOSURE_VALUE = 150
        prof = self.pipeline.start(self.config)

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

        self.precise_model = get_ResNet50(num_keypoints=self.NUM_KEYPOINTS, weights_path='data/outputs/keypointsrcnn_weights_10_epoch50.pth', load_device=self.device)
        self.precise_model.to(self.device)

        # configure robot
        while True:
            try:
                ROBOT_IP = "192.168.88.100"
                self.robot = urx.Robot(ROBOT_IP, use_rt=True)
                self.base_coord_sys = self.robot.csys
                ROBOT_IP = "192.168.88.200"
                self.UR3 = urx.Robot(ROBOT_IP, use_rt=True)
                break
            except urx.ursecmon.TimeoutException:
                continue

        # calibrate camera
        self.set_manual_calibration()

        # iniitalize step
        self.step = 'Rough'

        # random
        self.img_count = 0

        self.set_tool_manualy()

        self.pause_stream = False

    def set_tool_manualy(self):
        # 11 May
        plug_in_flange = [[0.99663655, -0.00311845, -0.08188932, -0.0115887],
                          [0.00407635,  0.99992518,  0.01153297,  0.0524041],
                          [0.08184722, -0.01182799,  0.9965747,   0.2242077],
                          [0.,          0.,          0.,          1.]]
        # 13 May

        plug_in_flange =   [[0.9976129, 0.06628157, 0.01937152, - 0.00371728],
         [-0.06732648, 0.99595649, 0.05947957, 0.05443973],
         [-0.01535079, - 0.06064181, 0.99804154, 0.24816427],
         [0., 0., 0., 1.]]
        """plug_in_flange =   [[9.98138050e-01,  5.84703971e-02,  1.73679730e-02,  9.65761986e-04],
                            [-5.92641872e-02,  9.97020109e-01,  4.93827666e-02, - 3.50289245e-02],
                            [-1.44287883e-02, - 5.03201172e-02,        9.98628908e-01,        2.19527732e-01],
                            [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]"""# old
        self.tool = plug_in_flange

    def set_manual_calibration(self):
        self.mtx = np.asarray([[634.14800253,   0.,         629.36554151],
                               [  0.,         634.13672323, 361.22908492],
                               [  0.   ,        0.         ,  1.    ,   ]])
        self.R_cam2gripper = R.from_rotvec([-0.02193857, -0.03329439, -3.14092666]).as_matrix()
        self.t_cam2gripper = [[-11.93768264], [118.74955747], [207.06885955]]
        self.dist = np.asarray([[-0.053358,    0.05795461, -0.00042388,  0.00043828, -0.01121712]])


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

    def re_loss(self, points_3d, image_pixels, img_tcp0):

        ### PREPARE DATA
        proxy = np.concatenate([self.R_cam2gripper, self.t_cam2gripper], axis=-1)

        cam2gripper = np.concatenate([proxy, np.array([[0, 0, 0, 1]])], axis=0)

        # print('cam2gripper', cam2gripper)

        grip2base_list = []
        base2cam_list = []
        rvecs_robot_in_camera = []
        tvecs_robot_in_camera = []

        # print('img_tcp0', img_tcp0)
        ori = R.from_rotvec(np.array(img_tcp0[3:]))
        # print('ori', ori.as_rotvec())
        ori_m = ori.as_matrix()

        mm_data = np.array(img_tcp0[:3]) * 1000
        # print('mm_data',mm_data)
        proxy = np.concatenate([ori_m, mm_data.reshape((3, 1))], axis=-1)
        grip2base = np.concatenate([proxy, np.array([[0, 0, 0, 1]])], axis=0)

        # print('grip2base', grip2base)

        grip2base_list.append(grip2base)

        # evaluation of cameras projection matrices
        cam2base = np.matmul(grip2base, cam2gripper)
        # print('cam2base', cam2base)
        # print('t_cam2base', cam2base[:3, 3])
        base2cam = np.linalg.inv(cam2base)
        base2cam_list.append(base2cam)
        rvec = R.from_matrix(base2cam[:3, :3]).as_rotvec()
        tvec = base2cam[:3, 3] / 1000

        imgpoints2, _ = cv2.projectPoints(points_3d,rvec , tvec, self.mtx, self.dist)
        imgpoints2 = [i[0] for i in imgpoints2]
        #print("image_pixels")
        #print(image_pixels)
        #print('imgpoints2')
        #print(imgpoints2)

        #print("reprojection error")
        reprojection_error = np.absolute(np.asarray(imgpoints2)-image_pixels)
        print(np.sum(reprojection_error))
        #print(np.sum(reprojection_error))

        return np.sum(reprojection_error)


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

    def predict_socket_points(self, return_image = False):
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            ### UNDISTORT IMAGE:
            h, w = color_image.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (
            w, h))  # be careful , mtx, dst are taken from the global nspace
            # undistort
            undst = cv2.undistort(color_image, self.mtx, self.dist, None, newcameramtx)
            # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            color_image_tensor = [F.to_tensor(undst)]
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
            if return_image:
                return flattened_keypoints, color_image
            else:
                return flattened_keypoints

    def insert_plug(self, to_go):
        to_go_rot = R.from_rotvec(to_go[3:]).as_matrix()
        to_go_trans = [[to_go[0]], [to_go[1]], [to_go[2]]]
        proxy = np.concatenate([to_go_rot, to_go_trans], axis=-1)
        to_go = np.concatenate([proxy, np.array([[0, 0, 0, 1]])], axis=0)
        to_go = np.concatenate([proxy, np.array([[0, 0, 0, 1]])], axis=0)
        #print("to_go", to_go)
        tcp_0_inside = to_go @ np.linalg.inv(self.tool)
        shift_in_socket_1 = [[0],
                           [0],
                           [-0.07]]

        shift_in_base_1 = to_go[:3, :3] @ shift_in_socket_1

        tcp_0_near_position_1 = tcp_0_inside[:3,3] + shift_in_base_1.T

        rotvec = R.from_matrix(tcp_0_inside[:3,:3]).as_rotvec()
        posevec_near = np.concatenate((tcp_0_near_position_1[0],rotvec))
        self.robot.movel(posevec_near, vel = .05, acc=0.05)

        tcp_0_inside = to_go @ np.linalg.inv(self.tool)
        shift_in_socket_2 = [[0],
                           [0],
                           [-0.02]]

        shift_in_base_2 = to_go[:3, :3] @ shift_in_socket_2

        tcp_0_near_position_2 = tcp_0_inside[:3,3] + shift_in_base_2.T

        rotvec = R.from_matrix(tcp_0_inside[:3,:3]).as_rotvec()
        posevec_inside = np.concatenate((tcp_0_near_position_2[0], rotvec))
        self.robot.movel(posevec_inside, vel=.005, acc=0.05)
        time.sleep(3)
        self.robot.movel(posevec_near, vel=.025, acc=0.05)
        time.sleep(3)
        # for test [-0.5388262907611292, -1.3727858702289026, -1.6960457007037562, 0.05184531211853027, 0.5155471563339233, -0.033739868794576466]


    def estimate_pose_precise(self, rough_socket_pose, method='triangulation'):
        self.step = 'Precise'

        # firstly, I want decide, how to make movements...
        # I will move around estimated pose of the socket....
        # I will start my journey from rotational movement
        rough_socket_rotation_in_base = R.from_rotvec(rough_socket_pose[3:]).as_matrix()
        robot_pose = np.asarray(self.robot.getl())

        look_positions_in_socket = [[0.05, 0, -0.05],
                                    [0.0, 0.07, -0.045],
                                    [-0.065, 0, -0.06],
                                    [0, -0.05, 0.015]]

        look_angles_in_socket = [[0, 30, 0],
                                 [-25, 0, 0],
                                 [0, -30, 0],
                                 [20, 0, 0]]

        frame_tcp0 = list([])
        frame_points = list([])
        where_id = 0
        wheres = ['left', 'up', 'right', 'down']
        for look in zip(look_positions_in_socket, look_angles_in_socket):
            rotation_in_socket = R.from_euler('xyz', look[1], degrees=True).as_matrix()
            position_in_socket = np.asarray(look[0])
            # calculate rotation
            rotation_in_base = np.asarray(rough_socket_rotation_in_base) @ rotation_in_socket.T
            rotvec_in_base = R.from_matrix(rotation_in_base).as_rotvec()
            # calculate position
            position_in_base = np.asarray(rough_socket_rotation_in_base) @ position_in_socket.T + rough_socket_pose[:3]
            self.robot.movel(np.concatenate((position_in_base, rotvec_in_base)), acc=0.15, vel=0.25)
            time.sleep(0.2)
            frame_tcp0.append(self.robot.get_pose().pose_vector)
            frame_points.append(list([]))
            look_points = []
            # save img
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                break
            save_data(color_image=color_image, robot=self.robot, where=wheres[where_id], env_id=2,
                      img_count=self.img_count)
            self.img_count += 1
            for _ in range(5):
                time.sleep(0.2)
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
            where_id += 1

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

        """### CALCULATE REPROJECTION ERROR
        re_losses = []
        for img_tcp0, image_keypoints in zip(frame_tcp0, frame_points):
            re_loss = self.re_loss(points_3d=np.mean(np.asarray(points), axis=0),
                                   image_pixels=image_keypoints,
                                   img_tcp0=img_tcp0)

            re_losses.append(re_loss)

        if re_losses[0] > 100 or re_losses[1] > 100 or re_losses[2] > 100 or re_losses[3] > 100:
            continue
        """
        std = np.std(points_in3d_robot_base, axis=0)

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

        points_mean = np.mean(points, axis=0)

        pseudo_ground_truth_mean = [ 0.96398465, -0.27205393,  0.67959601]

        print("point detection error: ", points_mean-pseudo_ground_truth_mean)

        x_from_combinations = []
        y_from_combinations = []
        z_from_combinations = []

        for direction_idx in direction_combinations:
            x = points[direction_idx[0]] - points[direction_idx[1]]
            x = x / np.linalg.norm(x)
            x_from_combinations.append(x)

        x = np.mean(x_from_combinations, axis=0)
        print(f'mean x: {x}')

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
        print(f'mean y: {y}')

        z = np.mean(z_from_combinations, axis=0)
        print(f'mean z: {z}')

        centerpoint = np.mean(points, axis=0)

        refpoint = np.asarray([x, y, z]).T @ [[0.02], [-0.08], [-0.12]]
        refpoint = refpoint.T + centerpoint
        refpoint = refpoint[0]

        r = R.from_matrix(np.asarray([x, y, z]).T).as_rotvec()
        reference_pose = np.concatenate((refpoint, r))
        #print(f'estimated reference pose {reference_pose}')
        return reference_pose


    def reproject_and_draw(self, points_3d, images_pixels, img_tcp0s, images = None):
        ### PREPARE

        R_cam2gripper = self.R_cam2gripper
        t_cam2gripper = self.t_cam2gripper
        #print('R_cam2gripper', 't_cam2gripper')
        #print(R_cam2gripper, t_cam2gripper)
        #print('images_pixels')
        #print(images_pixels)
        #print('img_tcp0s')
        #print(img_tcp0s)
        proxy = np.concatenate([R_cam2gripper, t_cam2gripper], axis=-1)
        cam2gripper = np.concatenate([proxy, np.array([[0, 0, 0, 1]])], axis=0)
        grip2base_list = []
        base2cam_list = []
        rvecs_robot_in_camera = []
        tvecs_robot_in_camera = []
        n=0
        for img_tcp0 in img_tcp0s:
            #print(n)
            n+=1
            ori = R.from_rotvec(np.array(img_tcp0[3:]))
            ori_m = ori.as_matrix()

            mm_data = np.array(img_tcp0[:3]) * 1000
            proxy = np.concatenate([ori_m, mm_data.reshape((3, 1))], axis=-1)
            grip2base = np.concatenate([proxy, np.array([[0, 0, 0, 1]])], axis=0)
            grip2base_list.append(grip2base)

            # evaluation of cameras projection matrices
            cam2base = np.matmul(grip2base, cam2gripper)
            base2cam = np.linalg.inv(cam2base)
            base2cam_list.append(base2cam)
            rvec = R.from_matrix(base2cam[:3, :3]).as_rotvec()
            tvec = base2cam[:3, 3] / 1000
            rvecs_robot_in_camera.append(rvec)
            tvecs_robot_in_camera.append(tvec)

        tvecs_robot_in_camera = np.asarray(tvecs_robot_in_camera)
        rvecs_robot_in_camera = np.asarray(rvecs_robot_in_camera)
        r_t_vecs = np.hstack((rvecs_robot_in_camera.ravel(), tvecs_robot_in_camera.ravel()))
        imgpoints_arr = []
        n_keypoints = len(images_pixels[0])
        n_keypoints = int(n_keypoints)

        amount_of_rvecs_params = len(images_pixels) * 3
        #print(amount_of_rvecs_params)
        r_vecs = r_t_vecs[:amount_of_rvecs_params].reshape(-1, 3)
        t_vecs = r_t_vecs[amount_of_rvecs_params:].reshape(-1, 3)
        #print(r_vecs)
        #print(t_vecs)
        for i in range(r_vecs.shape[0]):
            imgpoints2, _ = cv2.projectPoints(points_3d, r_vecs[i], t_vecs[i], self.mtx, self.dist)
            if images:
                self.draw_points(images[i], imgpoints2, i)
            imgpoints_arr.append(imgpoints2)
        re_loss = (np.array(imgpoints_arr).reshape((-1, n_keypoints, 2)) - images_pixels).ravel()
        print(np.mean(re_loss,axis=0))
        return np.mean(re_loss,axis=0)

    def draw_points(self, img, points, id):
        image = img

        for kps in points:

            for idx, kp in enumerate(kps):
                image = cv2.circle(image.copy(), (int(kp[0]), int(kp[1])), 2, (210, 32, 13), 2)
                cv2.imwrite(f'data/reprojection_{id}.jpg', image)
        """
        f, ax = plt.subplots(1, 2, figsize=(40, 20))

        ax[0].imshow(image)
        ax[0].set_title('image', fontsize=12)

        ax[1].imshow(image)
        ax[1].set_title('image', fontsize=12)
        plt.show()

        time.sleep(5)
        self.pause_stream = True
        """

        """
        cv2.namedWindow('Reprojection', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Reprojection', image)
        k = cv2.waitKey(0) & 0xFF
        """

    def estimate_pose_precise_with_orange_plug(self, rough_socket_pose, method='triangulation'):
        self.step = 'Precise'
        # firstly, I want decide, how to make movements...
        # I will move around estimated pose of the socket....
        # I will start my journey from rotational movement
        rough_socket_rotation_in_base = R.from_rotvec(rough_socket_pose[3:]).as_matrix()
        robot_pose = np.asarray(self.robot.getl())

        look_positions_in_socket = [[0.16, -0.01, -0.2],
                                    [-0.02, 0.125, -0.2],
                                    [-0.18, -0.0, -0.16],
                                    [0, -0.03, -0.205]]

        look_angles_in_socket = [[0, 30, 0],
                                 [-25, 0, 0],
                                 [0, -30, 0],
                                 [5, 0, 0]]

        frame_tcp0 = list([])
        frame_tcp0_for_reprojection = list([])
        frame_points = list([])
        frame_points_valid = list([])
        images = list([])
        where_id = 0
        wheres = ['left', 'up', 'right', 'down']
        for look in zip(look_positions_in_socket, look_angles_in_socket):
            rotation_in_socket = R.from_euler('xyz', look[1], degrees=True).as_matrix()
            position_in_socket = np.asarray(look[0])
            # calculate rotation
            rotation_in_base = np.asarray(rough_socket_rotation_in_base) @ rotation_in_socket.T
            rotvec_in_base = R.from_matrix(rotation_in_base).as_rotvec()
            # calculate position
            position_in_base = np.asarray(rough_socket_rotation_in_base) @ position_in_socket.T + rough_socket_pose[:3]
            self.robot.movel(np.concatenate((position_in_base, rotvec_in_base)), acc=0.15, vel=0.25)
            time.sleep(1)
            frame_tcp0.append(self.robot.get_pose().pose_vector)
            frame_points.append(list([]))
            look_points = []
            for _ in range(5):
                time.sleep(0.2)
                while True:
                    try:
                        pred_points, image = self.predict_socket_points(return_image=True)
                        look_points.append(pred_points)


                        #frame_points[-1] = np.append(frame_points[-1], self.predict_socket_points())
                        #print(self.predict_socket_points())
                        break
                    except Exception as e:
                        print(e)
                        continue
                frame_points[-1] = np.mean(look_points, axis=0)

            ### SAVE IMAGE
            save_data(color_image=image, robot=self.robot, where=wheres[where_id], env_id=2,
                      img_count=self.img_count)
            where_id += 1
            self.img_count += 1
            frame_tcp0_for_reprojection.append(self.robot.get_pose().pose_vector)

            unflattened_frame_points = np.reshape(np.array(frame_points[-1]), (-1, int(len(frame_points[-1])/2), 2)).tolist()
            images.append(image)
            valid_idx = [2, 3, 4, 5,
                         7, 8,
                         10, 11, 12, 13,
                         14]
            frame_points_valid.append(np.asarray(unflattened_frame_points[0])[valid_idx])

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

        ### CHECK BY REPROJECTION ERROR
        # when we obtained 3D coordinates, we may reproject them back to image
        # and see what is the difference between initial points predicted by the network on each frame
        reprojection_error = self.reproject_and_draw(points, frame_points_valid, frame_tcp0_for_reprojection, images)

        plane_combinations = [[10, 3, 0],
                              [6, 8, 1],
                              [4, 9, 3],
                              [5, 0, 6],
                              [7, 9, 2]]
        direction_combinations = [[0, 3], [6, 9]]



        x_from_combinations = []


        for direction_idx in direction_combinations:
            x = points[direction_idx[0]] - points[direction_idx[1]]
            x = x / np.linalg.norm(x)
            x_from_combinations.append(x)

        x = np.mean(x_from_combinations, axis=0)    # preliminary X

        x_from_combinations = []
        y_from_combinations = []
        z_from_combinations = []
        for plane_idx in plane_combinations:
            my_plane = plane(points[plane_idx[0]], points[1], points[0])
            z = np.asarray([my_plane.a, my_plane.b, my_plane.c])
            z = z / np.linalg.norm(z)
            #print(f'z for combination: {z}')
            y = np.cross(z, x)
            x = np.cross(y, z)
            x_from_combinations.append(x)
            y_from_combinations.append(y)
            z_from_combinations.append(z)

        x = np.mean(x_from_combinations, axis=0)
        print(f'mean x: {x}, {np.linalg.norm(x)}')

        y = np.mean(y_from_combinations, axis=0)
        print(f'mean y: {y}, {np.linalg.norm(y)}')

        z = np.mean(z_from_combinations, axis=0)
        print(f'mean z: {z}, {np.linalg.norm(z)}')

        print("X@Y")
        xy = x @ y
        print(x @ y)
        print("X@Z")
        xz = x @ z
        print(x @ z)
        print("Z@Y")
        zy = z @ y
        print(z @ y)


        centerpoint = np.mean(points, axis=0)

        # now we have rotation matrix and centerpoint. LETS CALCULATE THE TRANSFORM BETWEEN this coordinate system and reference: robot_tcp_near_near = [0.7102000170196487, -0.2408435343018464, 0.6649601094040635, 1.3089278830058457, 1.2421023952167922, 1.1545812090091068]
        r = R.from_matrix(np.asarray([x, y, z]).T).as_rotvec()

        reference_pose = np.concatenate((centerpoint, r))

        #return reference_pose

        refpoint = np.asarray([x, y, z]).T @ [[0.00], [0.00], [0.0]]
        refpoint = refpoint.T + centerpoint
        refpoint = refpoint[0]

        my_rot_tcp = R.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()      # X + down, - up; Y + ? - ?; Z + ? - ?
        my_rot_base = np.asarray([x, y, z]).T @ my_rot_tcp

        r = R.from_matrix(my_rot_base).as_rotvec()

        reference_pose = np.concatenate((refpoint, r))
        #print(f'estimated reference pose {reference_pose}')
        return reference_pose, reprojection_error, xy, xz, zy

    def estimate_position_rough_2_pts(self):
        self.step = 'Rough'
        poses = [[0.38682893556877224, 0.00394862768850718, 0.867077201101707, 1.7155989934183007, 1.5279110349137486, 0.654730062752815],
                 [0.3868232411252079, -0.10679818301766506, 0.84047548584032, 1.6808484104220096, 1.6368548849865332, 0.8173668762387832]]

        #poses = [[0.28682893556877224, 0.00394862768850718, 0.767077201101707, 1.7155989934183007, 1.5279110349137486, 0.654730062752815],
                 #[0.2868232411252079, -0.10679818301766506, 0.767047548584032, 1.6808484104220096, 1.6368548849865332, 0.8173668762387832]]

        #poses = [[0.3106910530795562, 0.10123962291978998, 1.0616972470718316, 2.8580139942532106, -0.9250232638359467, 0.8612857098330513],
                 #[0.3698221248099694, -0.18425234572932428, 1.0617424446981494, -2.402086131978857, 1.3762507545698124, -0.5008681600667722]]

        frame_tcp0 = []
        frame_points = []

        # frames collection:
        for pose in poses:
            time.sleep(0.1)
            self.robot.movel(pose, acc=0.2, vel=0.25)
            time.sleep(0.2)
            frame_tcp0.append(self.robot.get_pose().pose_vector)
            frame_points.append([])
            for _ in range(5):
                time.sleep(0.1)
                while True:
                    try:
                        frame_points[-1].append(self.predict_corners())
                        # print(frame_points[-1][-1])
                        break
                    except:
                        continue
            frame_points[-1] = np.mean(frame_points[-1], axis=0)

        combinations = [[0,1]]

        points_in3d_robot_base = []
        for combination in combinations:
            points_in3d_robot_base.append(triangulate(frame_points[combination[0]], frame_points[combination[1]],
                                          frame_tcp0[combination[0]], frame_tcp0[combination[1]],
                                          self.R_cam2gripper, self.t_cam2gripper, self.mtx))
            points_in3d_robot_base[-1] = points_in3d_robot_base[-1]/1000

        points_in3d_robot_base = np.mean(points_in3d_robot_base, axis=0)

        points = np.asarray(points_in3d_robot_base)
        centerpoint = np.mean(points, axis=0)
        my_plane = plane(points[2], points[1], points[0])
        z = np.asarray([my_plane.a, my_plane.b, my_plane.c])
        z = z / np.linalg.norm(z)
        x = points[0]-points[1]
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        x = np.cross(y, z)

        refpoint = np.asarray([x, y, z]).T @ [[0.02], [-0.06], [-0.10]] # ex refpoint = np.asarray([x, y, z]).T @ [[0.02], [0.16], [-0.28]]
        refpoint = refpoint.T + centerpoint
        refpoint = refpoint[0]

        r = R.from_matrix(np.asarray([x, y, z]).T).as_rotvec()      # rotation vector of the TCP in base
        r_euler = R.from_matrix(np.asarray([x, y, z]).T).as_euler('zyx', degrees=True)
        my_rot_tcp = R.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()       # ex         my_rot_tcp = R.from_euler('xyz', [0, 45, 0], degrees=True).as_matrix()

        my_rot_base = np.asarray([x, y, z]).T @ my_rot_tcp

        r = R.from_matrix(my_rot_base).as_rotvec()
        reference_pose = np.concatenate((refpoint, r))
        return reference_pose

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
                image = cv2.circle(image.copy(), tuple(kp), 2, (75, 32, 210), 2)

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
    def visualize_by_steps(self):
        if self.robot or not self.robot:
            try:
                n = 0
                cv2.namedWindow('Inference', cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Inference", 7680, 4320)
                # cv2.setMouseCallback('Inference', self.mouse_callback)
                while True:
                    #print(self.robot.getl()[2])
                    if self.pause_stream:
                        time.sleep(4)
                    # Wait for a coherent pair of frames: depth and color
                    frames = self.pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue

                    # Convert images to numpy arrays
                    color_image = np.asanyarray(color_frame.get_data())
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

                    if self.step == "Rough":
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
                            pass
                            #print("No keypoints detected")

                        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                        # draw inference or raw image
                        cv2.imshow('Inference', out.get_image()[:, :, ::-1])
                    elif self.step == "Precise":

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
                    else:
                        cv2.imshow('Inference', color_image)

                    cv2.waitKey(1)
                    # time.sleep(3)
            finally:
                # Stop streaming
                self.pipeline.stop()

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

    def run_single_insertion(self, combination='ResNet50+triangulation', mode='Auto'):
        if combination == 'ResNet50+triangulation':
            pass
        else:
            raise ValueError('Unknown experiment combination yet')

        rough_pose = self.estimate_position_rough_2_pts()
        #self.move_to_ref(rough_pose)
        precise_pose, reprojection_error, xy, xz, zy = self.estimate_pose_precise_with_orange_plug(rough_pose)
        print(precise_pose)
        self.insert_plug(precise_pose)

        return precise_pose, reprojection_error, xy, xz, zy
        #self.move_to_ref(precise_pose)
        #print(precise_pose)


        #target_pose_in_TCP = self.calculate_transform_to_ref_with_plug(precise_pose) # (error)
        #print(target_pose_in_TCP)

        #precise_ori = R.from_rotvec(precise_pose[3:]).as_matrix()
        #precise_position = precise_pose[:3]
        #target_position_in_base = precise_ori @ target_pose_in_TCP[:3]
        #target_orientation_in_base =


    def run_experiment(self, combination='ResNet50+triangulation', mode='Auto'):
        if combination == 'ResNet50+triangulation':
            pass
        else:
            raise ValueError('Unknown experiment combination yet')
        experiment_results = pd.DataFrame()
        # run over all 4 possible ur3 positions
        ur3_pose_id = 0
        for ur3_pose, true_pts in zip(pts.ur3_poses, pts.true_points):
            try:
                try:
                    self.UR3.movej(ur3_pose, acc=0.25, vel=0.45)
                except Exception as e:
                    print(e)
                    time.sleep(10)

                # run N experiments on given ur3_pose from random combinations of points
                for exp_id in range(100):
                    rough_pose = self.estimate_position_rough()
                    error_in_pose = self.get_pose_error(ur3_pose_id)
                    if any([lin_error > 0.03 for lin_error in error_in_pose[:3]]):
                        print('Wrong rough estimation. Start again.')
                        print(error_in_pose)
                        continue
                    self.move_to_ref(rough_pose)
                    precise_pose = self.estimate_pose_precise(rough_pose)
                    self.move_to_ref(precise_pose)
                    error_in_pose = self.get_pose_error(ur3_pose_id)
                    # save results
                    run_result_dict = {'ur3_pose_id': ur3_pose_id,
                                       'exp_id': exp_id,
                                       'pose_x': precise_pose[0],
                                       'pose_y': precise_pose[1],
                                       'pose_z': precise_pose[2],
                                       'pose_roll': precise_pose[3],
                                       'pose_pitch': precise_pose[4],
                                       'pose_yaw': precise_pose[5],
                                       'error_in_pose_x': error_in_pose[0],
                                       'error_in_pose_y': error_in_pose[1],
                                       'error_in_pose_z': error_in_pose[2],
                                       'error_in_pose_roll': error_in_pose[3],
                                       'error_in_pose_pitch': error_in_pose[4],
                                       'error_in_pose_yaw': error_in_pose[5]}

                    experiment_results = experiment_results.append(run_result_dict, ignore_index=True)
                    print('Obtained pose and error...')
                    print(precise_pose)
                    print(error_in_pose)
                    self.move_back_exp()
                experiment_results.to_csv(f'data/experiments/insertion/pose{ur3_pose}_experiment.csv')
                ur3_pose_id += 1
            except:
                experiment_results.to_csv('data/experiments/insertion/not_finished_experiment.csv')
        experiment_results.to_csv('data/experiments/insertion/experiment.csv')

    def calculate_transform_to_ref_with_plug(self, given_pose):
        robot_tcp_near_near = [0.7102000170196487, -0.2408435343018464, 0.6649601094040635, 1.3089278830058457,
                               1.2421023952167922, 1.1545812090091068]
        target_position = np.asarray(robot_tcp_near_near[:3])
        target_orientation = R.from_rotvec(robot_tcp_near_near[3:]).as_matrix()
        print(target_orientation)
        given_position = given_pose[:3]
        given_orientation = R.from_rotvec(given_pose[3:]).as_matrix()

        target_ori_in_given = target_orientation @ np.linalg.inv(given_orientation)

        target_position_in_given = np.linalg.inv(given_orientation) @ (target_position - given_position)

        print(target_position_in_given, R.from_matrix(target_ori_in_given).as_rotvec())

        return np.concatenate((target_position_in_given, R.from_matrix(target_ori_in_given).as_rotvec()))

    def run_experiment_v2(self, combination='ResNet50+triangulation', mode='Auto'):
        if combination == 'ResNet50+triangulation':
            pass
        else:
            raise ValueError('Unknown experiment combination yet')
        experiment_results = pd.DataFrame()
        # run over all 4 possible ur3 positions

        try:

            # run N experiments on given ur3_pose from random combinations of points
            for exp_id in range(100):
                rough_pose = self.estimate_position_rough_2_pts()

                self.move_to_ref(rough_pose)
                precise_pose = self.estimate_pose_precise_with_plug(rough_pose)
                error_in_pose = self.calculate_transform_to_ref_with_plug(precise_pose)
                # save results
                run_result_dict = {'exp_id': exp_id,
                                   'pose_x': precise_pose[0],
                                   'pose_y': precise_pose[1],
                                   'pose_z': precise_pose[2],
                                   'pose_roll': precise_pose[3],
                                   'pose_pitch': precise_pose[4],
                                   'pose_yaw': precise_pose[5],
                                   'error_in_pose_x': error_in_pose[0],
                                   'error_in_pose_y': error_in_pose[1],
                                   'error_in_pose_z': error_in_pose[2],
                                   'error_in_pose_roll': error_in_pose[3],
                                   'error_in_pose_pitch': error_in_pose[4],
                                   'error_in_pose_yaw': error_in_pose[5]}

                experiment_results = experiment_results.append(run_result_dict, ignore_index=True)
                print('Obtained pose and error...')
                print(precise_pose)
                print(error_in_pose)
                self.move_back_exp()
            experiment_results.to_csv(f'data/experiments/insertion/pose{0}_experiment.csv')

        except:
            experiment_results.to_csv('data/experiments/insertion/not_finished_experiment.csv')
        experiment_results.to_csv('data/experiments/insertion/experiment.csv')

    # pose estimators: return points_in_3d_robot_base


def save_data(color_image, robot, img_count=0, env_id = 2, where = None):
    try:
        path, dirs, files = next(os.walk(f'data/big_experiment/my_env_{env_id}/raw/{where}'))


    except Exception as e:

        print(e)

        path = f'data/big_experiment/my_env_{env_id}/raw/{where}'
        print(f"Directory {path} was not found. Creating the directory...")
        os.makedirs(path)
        print(f"Directory {path} was created.")
        path, dirs, files = next(os.walk(f'data/big_experiment/my_env_{env_id}/raw/{where}'))


    file_count = len(files)
    img_count = int(file_count / 2)

    filename_img = f'data/big_experiment/my_env_{env_id}/raw/{where}/frame_env_{env_id}_{where}_' + str(img_count)
    filename_json = f'data/big_experiment/my_env_{env_id}/raw/{where}/meta_env_{env_id}_{where}_' + str(img_count)

    try:
        # save raw
        cv2.imwrite(filename_img+'.jpg', color_image)
        data = {"j": robot.getj(),'l': list(robot.get_pose().pose_vector)}
        with open(filename_json+'.json', 'w') as outfile:
            json.dump(data, outfile)
        #print(str(img_count)+' saved')
    except Exception as e:
        print(e)

if __name__ == "__main__":
    go = ResNetCV()
    stream = Thread(target=go.visualize_by_steps)
    posess = [[0.38682893556877224, 0.00394862768850718, 0.867077201101707, 1.7155989934183007, 1.5279110349137486,
              0.654730062752815],
             [0.3868232411252079, -0.10679818301766506, 0.84047548584032, 1.6808484104220096, 1.6368548849865332,
              0.8173668762387832]]
    stream.start()
    precise_poses = []
    reprojection_errors = []
    dot_products = []
    iniJ = go.UR3.getj()
    target_g = iniJ.copy()
    target_g[0] = target_g[0] - np.deg2rad(10)
    target_g[1] = target_g[1] + np.deg2rad(10)
    for i in range(100):
        # time.wait(5)
        precise_pose, reprojection_error, xy, xz, zy = go.run_single_insertion()
        print("go.run_single_insertion() finished")
        # go.robot.movel(posess[0], acc=0.2, vel=0.25)
        # time.sleep(3)
        # go.UR3.movej(target_g)
        # time.sleep(3)
        # precise_pose, reprojection_error, xy, xz, zy = go.run_single_insertion()
        # print("go.run_single_insertion() finished")
        # go.robot.movel(posess[0], acc=0.2, vel=0.25)
        # time.sleep(3)
        # go.UR3.movej(iniJ)
        # time.sleep(3)
        # precise_poses.append(precise_pose)
        # reprojection_errors.append(reprojection_error)
        # dot_products.append([xy, xz, zy])
        # with open('data/big_experiment/config_16May_test_env_1/precise_poses_v1.011111111111.pickle', 'wb') as f:
        #     pickle.dump(precise_poses, f)
        # with open('data/big_experiment/config_16May_test_env_1/reprojection_errors_v1.01111111111.pickle', 'wb') as f:
        #     pickle.dump(reprojection_errors, f)
        # with open('data/big_experiment/config_16May_test_env_1/dot_poructs_errors_v1.011111111111.pickle', 'wb') as f:
        #     pickle.dump(dot_products, f)

"""RESULT 03 may


"""

"""
Obtained pose and error...
[ 1.00831165 -0.04841585  0.65077565  1.07029205  1.17508615  1.21039984]
[ 0.00037953 -0.00132203 -0.0077237  -0.01567339  0.01592467  0.0502891 ]

[ 1.00839925 -0.05061507  0.65208986  1.06550072  1.19113746  1.21471578]
[ 0.00165823 -0.00131493 -0.00986785 -0.00276827  0.02367117  0.05536058]

[ 1.01104251 -0.04648131  0.64935667  1.06597774  1.14933686  1.21249313]
[-0.00130608  0.00168572 -0.00627366 -0.02972779 -0.00393953  0.04449956]

[ 1.00821112 -0.04970795  0.65145515  1.06850992  1.18405055  1.21849232]
[ 0.00106207 -0.0014132  -0.00896127 -0.00446955  0.01936044  0.04974345]

[ 1.00802238 -0.05051772  0.65213442  1.06636811  1.19211376  1.2134936 ]
[ 0.00185888 -0.00173126 -0.0096478  -0.00274589  0.02499701  0.0557418 ]

[ 1.00622296 -0.05267229  0.65455829  1.07106982  1.21965536  1.22897989]
[ 0.00440441 -0.00345869 -0.01154586  0.0228494   0.04315565  0.05085453]

[ 1.01032116 -0.04689298  0.64852085  1.06383588  1.15113692  1.20490962]
[-0.00217438  0.00087896 -0.00673591 -0.03365253 -0.00167145  0.05019466]

[ 1.00913326 -0.04945952  0.64938385  1.05932404  1.17008184  1.21515839]
[-0.00087225 -0.00060785 -0.00895216 -0.01426727  0.00583239  0.05383595]
"""

"""
[ 1.00831165 -0.04841585  0.65077565  1.07029205  1.17508615  1.21039984]
[ 1.00839925 -0.05061507  0.65208986  1.06550072  1.19113746  1.21471578]
[ 1.01104251 -0.04648131  0.64935667  1.06597774  1.14933686  1.21249313]
[ 1.00821112 -0.04970795  0.65145515  1.06850992  1.18405055  1.21849232]
[ 1.00802238 -0.05051772  0.65213442  1.06636811  1.19211376  1.2134936 ]
[ 1.00622296 -0.05267229  0.65455829  1.07106982  1.21965536  1.22897989]
[ 1.01032116 -0.04689298  0.64852085  1.06383588  1.15113692  1.20490962]
[ 1.00913326 -0.04945952  0.64938385  1.05932404  1.17008184  1.21515839]

"""

"""
[ 1.10820823 -0.02120459  0.58875129  1.37236899  1.84341103  0.71185372]
[ 1.10815865 -0.01761251  0.58505     1.37153372  1.80343875  0.69341737]
[ 1.10766613 -0.02205746  0.59154342  1.38765284  1.8586548   0.7043219 ]
[ 1.10797415 -0.01569308  0.58687417  1.36994943  1.80750121  0.66601079]
[ 1.10743266 -0.01990526  0.58883394  1.37593341  1.83182438  0.68835113]
[ 1.10925326 -0.02513682  0.5920018   1.37481307  1.87914972  0.71665707]
"""

"""
DRY RUN:
[0.6310307353487026, -0.20784389836303357, 0.619183583333346, 1.222559618773729, 1.1568931043178088, 1.1252909777598639]
[0.6270570378249464, -0.21194332514325281, 0.6422423896079679, 1.2760912493074195, 1.2082799566147018, 1.1187670490351114]
[0.6371333704838892, -0.19347198536993987, 0.6066604076378502, 1.210000086914988, 1.094366164865446, 1.1141031176796827]
[0.6325449692927693, -0.19830884895854486, 0.6209788161122202, 1.233815465307381, 1.1395997584383042, 1.1005234039394118]
[0.6220061856090011, -0.22748544278527214, 0.6447308055863539, 1.2516601042345148, 1.2500783416424428, 1.13728074044423]
[0.6299046954867912, -0.20372326395642726, 0.6240257515789795, 1.2384409993638155, 1.1537888044484668, 1.1179358017163152]
[0.62950758358092, -0.20502509355522355, 0.6262334107970023, 1.244460773592647, 1.1615602681449897, 1.1204115315993008]
[0.6309785870680145, -0.20241510533091464, 0.6281850869424381, 1.2519815040840712, 1.1584305688296426, 1.1142042103821743]
[0.6366246942335808, -0.19548307349807836, 0.6137854489384409, 1.223470724615874, 1.1104436986779018, 1.1149849107814866]
[0.635597835146649, -0.19168539545110874, 0.6185528677428185, 1.236430429556117, 1.1156533940841364, 1.0949901180847197]
[0.6324647928401665, -0.20307334158253695, 0.6203905883301082, 1.2312084241181747, 1.1450712029689172, 1.1215569280107418]
[0.6356349295793658, -0.19036469877514667, 0.6189321561401361, 1.236739504063971, 1.1120955846448446, 1.0917441264615213]
[0.6382169440031039, -0.18094273765395355, 0.6210255047232853, 1.252580517912625, 1.0943935535060567, 1.0697129143833877]
[0.634753448517504, -0.19576518330325013, 0.6179027678885993, 1.2331610344024104, 1.1242481931379957, 1.1063657409141079]
[0.6392789434277031, -0.1918662029569491, 0.6085009701961671, 1.2167544566026909, 1.089138322963678, 1.1177511255471868]
[0.6268143048205943, -0.21020661041479885, 0.6304122213490497, 1.2368924283575664, 1.186647972043903, 1.1106082390206315]
[0.6330712110687946, -0.20383091525864747, 0.621827692374271, 1.2355361486665015, 1.1443529573399776, 1.1317235505396215]
[0.6278642147162216, -0.20315570352728088, 0.6316815617055738, 1.2521330913997322, 1.1705284455336824, 1.1017906788785365]
[0.6356612928435397, -0.19155580903117198, 0.6190982226743219, 1.2374587148635428, 1.11410060201996, 1.098839304102262]
[0.6315416154353537, -0.20368186913199696, 0.6219020231350136, 1.2322488312933633, 1.1476501078048673, 1.1227897330532526]
[0.6304704218795198, -0.19737551855183133, 0.6317221470167094, 1.255875260730256, 1.1621371248363768, 1.083259619247061]
[0.6285621362934789, -0.20565699992905512, 0.6246979852296479, 1.2349163573849464, 1.1629436324680091, 1.1157108241729143]
[0.6252766251430073, -0.22237890715459513, 0.6338109409913244, 1.2343367835258234, 1.214605998186988, 1.1476114968633089]
[0.6325364932634805, -0.19268161941662945, 0.6341371988567381, 1.2764994979969595, 1.1533994623996835, 1.0797876570456921]
[0.635134214455654, -0.1926779137525139, 0.6249954585483414, 1.247995018154351, 1.1283746738313751, 1.091553794153225]
[0.6305643993105658, -0.19937452155872776, 0.6283854207678011, 1.2476755200381375, 1.157294023844392, 1.095940834592761]
[0.637887215882231, -0.18901193425582025, 0.6136593904768511, 1.228464993235681, 1.0976628966587834, 1.0984032210944892]
[0.6444372989575772, -0.17620018456920827, 0.6052207032198891, 1.2178617561464535, 1.0523608331818979, 1.0722231807640517]
[0.6303599034959237, -0.20140533235323413, 0.6231428015515706, 1.2353220903307491, 1.149540358405071, 1.1084988253148147]
[0.6331621820969159, -0.20269696268917467, 0.6113072893702602, 1.2126120950158272, 1.1240397562481819, 1.1332261801599164]
[0.6345222020454345, -0.1969155054968329, 0.6208601374290483, 1.2374066463306892, 1.1296635322582305, 1.109719075529485]
[0.6250576482571929, -0.22734831860117385, 0.6438378368291501, 1.2565501745123728, 1.2443155835657833, 1.1493466813299473]
[0.6260807750192658, -0.21350289992002217, 0.6357651107992405, 1.2433968541422213, 1.2039743842781967, 1.1181585352619303]
[0.6242642481893325, -0.21350633301288124, 0.6520894748167013, 1.2860027994835774, 1.2332591052074837, 1.1067897479795474]
[0.6255490421478749, -0.2196441867508135, 0.6379267516720388, 1.2474927911102778, 1.2183173121330888, 1.1346567336848081]
[0.6346692718527366, -0.19435120677179, 0.6190392138862758, 1.2353393863272977, 1.1225863202050625, 1.1009931496289402]
[0.6325577240522794, -0.20249236021067257, 0.6224728360731423, 1.2364397459125787, 1.1460793056350087, 1.1205280267115976]
[0.6342110362800805, -0.19924119199700233, 0.614376168072826, 1.2175402278886085, 1.1234983448769837, 1.1162662173088764]
[0.6272771277622917, -0.21378222735045802, 0.6340939818166368, 1.2461940631151494, 1.199248328654836, 1.1257953826211589]
[0.6377695098503567, -0.18424653299606514, 0.6139628892906385, 1.2361401364027047, 1.0892615997972226, 1.085700270698389]
[0.6304716922433852, -0.205184830449724, 0.6223282372313116, 1.236142964410785, 1.1495803398641, 1.1315010351204033]
[0.6408079260421725, -0.18951737852796421, 0.6054213579863417, 1.2123459373707464, 1.0794439252036307, 1.1105616658149138]
[0.6315444631155042, -0.20179504564857265, 0.6236149397888694, 1.2357851046687567, 1.1512628827482188, 1.1091724886194696]
[0.6308749639587489, -0.20684082626296255, 0.624159687719859, 1.2349565079142937, 1.1636058923098564, 1.1196789222578143]
[0.6334940146692073, -0.2015641722113284, 0.620891314938846, 1.2332302791418208, 1.142979062482205, 1.1151443846240343]
[0.6301060796408964, -0.20368943408342927, 0.6289021525316464, 1.2472859879873712, 1.165554717391841, 1.1091693362024555]
[0.6277070192090034, -0.21086135898292996, 0.6387740190622309, 1.2648245814848411, 1.2011952770017602, 1.115856018621407]
[0.6323883829095864, -0.1989307901992753, 0.6242303746861714, 1.2430200413405468, 1.1429359014586413, 1.1055904825206129]
[0.6322366504215694, -0.2048195985233767, 0.62046736497791, 1.2272328250172657, 1.147639513181926, 1.1250444973105316]
[0.6278689957682696, -0.214272019539035, 0.633659216554552, 1.2410627849704008, 1.2025912193985995, 1.1194247573375466]
[0.6319146906434426, -0.20960209215434486, 0.6229403469186833, 1.2301569229332059, 1.163537623073887, 1.132182620260925]
[0.6347041301045573, -0.19865212573133734, 0.6120699347247763, 1.2192118696568732, 1.1190636921179957, 1.1201095532320537]
[0.6254068410394076, -0.22215004394815938, 0.6444826003250386, 1.2606939346556316, 1.2359161336794153, 1.1329975840474384]
[0.6387913763030855, -0.1806028904006156, 0.6298337170310724, 1.2665001919219483, 1.1166927759100735, 1.0519616239647853]
[0.6296929533782687, -0.2012520732710867, 0.6337584186312435, 1.2649688763073135, 1.171754343483882, 1.0964624955240518]
[0.6296487563807109, -0.20732606250112573, 0.6321566153693045, 1.252398537372034, 1.1815571334786255, 1.1144513674609788]
[0.634774790780805, -0.19228814439557257, 0.6187450391576911, 1.2351715850429306, 1.1198005005426028, 1.0923862928231463]
[0.6301810136470634, -0.20641748115724348, 0.6346416810329998, 1.2573822108466886, 1.188565934863537, 1.0985227437901388]
[0.6304986817488948, -0.2029731422755952, 0.6266491104175499, 1.2476629597419797, 1.1579443458414516, 1.1149234652671047]
[0.6349882477904167, -0.19897393282268028, 0.62078196844527, 1.2410148627318125, 1.1298624362849596, 1.1232802761301506]
[0.6272396090894833, -0.2076322266299836, 0.6586217671372774, 1.3303355458569557, 1.2233620989595226, 1.1017344829792688]
[0.6286503124492188, -0.2132625010352202, 0.6232754011947113, 1.2214960718344947, 1.176073087354241, 1.1313581539616044]
[0.632019553104475, -0.20858217787264785, 0.6195485533372813, 1.2180762697800227, 1.1564431836820694, 1.1290579924015087]
[0.6356233419022624, -0.19304270113251865, 0.6202190954563402, 1.2390774054912905, 1.1209893094670749, 1.097704226047499]
[0.6283162175381692, -0.20926153260215563, 0.6386675566432765, 1.2677920183014637, 1.1966254279337765, 1.1144677198977382]
[0.6278778846431742, -0.2141140753617606, 0.6367729918509577, 1.2567499764373136, 1.2036547724165851, 1.124997597782491]
[0.6282613080092566, -0.2201365749712933, 0.6293029595149778, 1.2309919060704686, 1.2003475557021088, 1.1494627969565805]
[0.6216359639001847, -0.22255819686422038, 0.6907202374115299, 1.364345840930672, 1.3248564473156843, 1.0788590032938208]
[0.6288634541038499, -0.21169265198718606, 0.6413368095755201, 1.271802355730837, 1.2083987669250824, 1.1134217490551601]
[0.6221334820814793, -0.2376690323209076, 0.6678614966570231, 1.2929997740734593, 1.3158519637180899, 1.1364260072911163]
[0.6273476113972904, -0.21852307224483927, 0.6374934758338263, 1.2549204152567803, 1.212354167981038, 1.1383591175779308]
[0.6350437567458531, -0.19565237033455604, 0.6266109400364944, 1.2531211940830504, 1.13612493884497, 1.1055172985084596]
[0.6370220722991523, -0.19630089040624227, 0.6099432141341246, 1.208354682414515, 1.1106734456146572, 1.1090203627756055]
[0.6373544159752699, -0.19202480349333687, 0.6155522757488127, 1.2290162750078566, 1.11212918003964, 1.0936784647770879]
[0.6320760435381981, -0.19582401767216023, 0.6239256875530093, 1.2475572409896007, 1.1334849902997748, 1.1052859237868358]
[0.6258013837822982, -0.22645845539704718, 0.6365185171600896, 1.2378612374591849, 1.2286495037766587, 1.1510795988391875]
[0.6362416526606371, -0.20049720664919832, 0.6093506002860547, 1.2095003940051359, 1.113376955429253, 1.1309185833392605]
[0.6304336702202735, -0.20883930507702822, 0.6241715862801303, 1.2293381352545325, 1.1684071616839553, 1.1222540038172917]
[0.6329966320422706, -0.1955313301872263, 0.6204616160338574, 1.2394196398581656, 1.129444696368413, 1.1023037351376133]
"""