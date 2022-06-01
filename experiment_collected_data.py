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

from insertion_utils import triangulate, triangulate_unflattened
import data.experiments.insertion.input_points as pts

from models.ResNet50_model import get_model as get_ResNet50
from models.detectron2_MaskRCNN import get_model as get_MaskRCNN

from insertion_utils import plane


class CalibrationModel:
    def __init__(self, intrinsic, distortion, name):
        self.intrs = intrinsic
        self.dist = distortion
        self.name = name
        self.t_cam2gripper = None
        self.R_cam2gripper = None

    def fill_extrinsic(self, R_cam2gripper, t_cam2gripper):
        self.t_cam2gripper = t_cam2gripper
        self.R_cam2gripper = R.from_rotvec(R_cam2gripper).as_matrix()

    def estimate_re_error(self, images_path):
        objp = np.zeros((8 * 5, 3), np.float32)
        # step size of the checkboard is 100 mm
        objp[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2) * 100
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        mean_re_error = 0

        for fname in images_path:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (8, 5), None)

            if ret == True:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                ret, rvec, tvec = cv2.solvePnP(objp, corners2, self.intrs, self.dist)
                imgpoints_proj, _ = cv2.projectPoints(objp, rvec, tvec, self.intrs, self.dist)
                error = cv2.norm(corners2, imgpoints_proj, cv2.NORM_L2)/len(corners2)
                mean_re_error +=error

        return mean_re_error

# download objects
with open("data/calib_models_ro", "rb") as f:
    calibs = pickle.load(f)

for calib in calibs:
    print(calib.name)
    if calib.name == 'calibration_v5_frames':
        calib.fill_extrinsic([5.75741543e-04, 1.24335210e-02, 3.14059231e+00], [-26.57405788, 109.38070665,  29.24488396])

    if calib.name == 'calibration_v7_frames':
        calib.fill_extrinsic([0.00746955, -0.04715531, 3.13828964], [-29.3115531,  107.42735831,  22.47290233])

    if calib.name == 'calibration_v9_frames':
        calib.fill_extrinsic([0.02825456, - 0.01947504, - 3.12670634], [-24.69554673, 130.25950539,  34.90702682])

    if calib.name == 'calib_horizontal':
        calib.fill_extrinsic([0.04419481,  0.06909873, - 3.12774781], [-18.4751669,  118.74709488,  26.0659704])

    print(calib.intrs)

valid_calibs = ['calib_horizontal', 'calibration_v5_frames', 'calibration_v7_frames', 'calibration_v9_frames']

class Experiment():
    def __init__(self, env_ids):
        # initalize model
        self.NUM_KEYPOINTS = 19
        self.device = torch.device('cpu')

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


        self.precise_model = get_ResNet50(num_keypoints=self.NUM_KEYPOINTS, weights_path='data/outputs/keypointsrcnn_weights_6_epoch260.pth', load_device=self.device)
        self.precise_model.to(self.device)

        # calibrate camera
        self.mtx, self.t_cam2gripper, self.R_cam2gripper = self.set_manual_calibration()

        # load experimental data
        self.img_fns = []
        self.img_fns_dict = {}
        self.down_img_filenames = []
        self.up_img_filenames = []
        self.left_img_filenames = []
        self.right_img_filenames = []

        for env_id in env_ids:
            for path, dirs, files in os.walk(f"data/big_experiment/env_{env_id}/raw"):
                for file in files:
                    if file[-3:] == 'jpg':
                        if path.split('/')[-1] not in self.img_fns_dict.keys():
                            self.img_fns_dict[path.split('/')[-1]] = [os.path.join(path, file)]
                        else:
                            self.img_fns_dict[path.split('/')[-1]].append(os.path.join(path, file))
                        self.img_fns.append(os.path.join(path, file))

        #print(self.img_fns_dict)

        #self.keypoint_prediction_list, self.bboxes_prediction_list = self.predict_full_dataset()

        self.keypoint_prediction_dict = {}
        self.bboxes_prediction_dict = {}
        for key in self.img_fns_dict.keys():
            self.keypoint_prediction_dict[key], self.bboxes_prediction_dict[key] = self.predict_list(self.img_fns_dict[key])


    def set_manual_calibration(self):
        mtx = [[633.22,   0.,         629.326],
               [  0.,         632.515, 360.888],
               [  0.,           0.,           1. ]]
        """#calib1
        t_cam2gripper = [[-23.],
                              [107.],
                              [26.]]

        R_cam2gripper = [[-0.99926641, -0.03782104, 0.00526198],
                              [0.03777663, -0.99922319, -0.00994069],
                              [0.0056323, -0.00973294, 0.99992018]]
        """
        """#calib Viktor v7
        t_cam2gripper = [[-29.028],
                         [108.07],
                         [22.745]]

        R_cam2gripper = [[-0.99998298, -0.00323863,  0.0048522 ],
                           [ 0.00309125, -0.99954272, -0.03007968],
                           [ 0.0049474 , -0.03006417,  0.99953573]]
        """
        # calib Viktor v5
        t_cam2gripper = [[-23.89993997], [112.4474997],   [29.15469655]]


        R_cam2gripper =[[-9.99995414e-01, -3.01576701e-03, -2.79144817e-04],
       [ 3.01316166e-03, -9.99955807e-01,  8.90537008e-03],
       [-3.05989002e-04,  8.90448813e-03,  9.99960307e-01]]

        return mtx, t_cam2gripper, R_cam2gripper

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

        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_gripper2base=dataposes_rot,
                                                                   t_gripper2base=dataposes_shift,
                                                                   R_target2cam=rvecs,
                                                                   t_target2cam=tvecs,
                                                                   method=cv2.CALIB_HAND_EYE_ANDREFF)
        self.dataposes_rot = dataposes_rot


        print("intrinsic matrix from images")
        print(mtx)

        print("camera in gripper coordinates:")
        print(t_cam2gripper)

        print("rotation vector:")
        print(R.from_matrix(R_cam2gripper).as_rotvec())

        return mtx, t_cam2gripper, R_cam2gripper

    def predict_list(self, filenames):
        keypoint_prediction_list = []
        bboxes_prediction_list = []

        for img_filename in filenames:
            color_image = cv2.imread(img_filename)
            # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            color_image_tensor = [F.to_tensor(color_image)]
            color_image_tensors = list(image.to(self.device) for image in color_image_tensor)

            image = (color_image_tensors[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)

            keypoints, bboxes = self.predict_socket_points(image)

            keypoint_prediction_list.append(keypoints)
            bboxes_prediction_list.append(bboxes)

        return keypoint_prediction_list, bboxes_prediction_list

    def predict_full_dataset(self):
        keypoint_prediction_list = []
        bboxes_prediction_list = []

        for img_filename in self.img_fns:
            color_image = cv2.imread(img_filename)
            # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            color_image_tensor = [F.to_tensor(color_image)]
            color_image_tensors = list(image.to(self.device) for image in color_image_tensor)

            image = (color_image_tensors[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)

            keypoints, bboxes = self.predict_socket_points(image)

            keypoint_prediction_list.append(keypoints)
            bboxes_prediction_list.append(bboxes)

        return keypoint_prediction_list, bboxes_prediction_list


    def predict_socket_points(self, img):
        # Convert images to numpy arrays
        color_image = img
        #color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

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
            raise ValueError('Not all keypoints are detected')

        bboxes = []
        for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            bboxes.append(list(map(int, bbox.tolist())))

        return keypoints, bboxes

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
    def visualize_precise(self, img_filename):

        cv2.namedWindow(img_filename, cv2.WINDOW_AUTOSIZE)


        # Convert images to numpy arrays
        color_image = cv2.imread(img_filename)
        #color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        color_image_tensor = [F.to_tensor(color_image)]
        color_image_tensors = list(image.to(self.device) for image in color_image_tensor)


        image = (color_image_tensors[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)

        keypoints, bboxes = self.predict_socket_points(image)

        color_image = self.draw_precise_keypoints(image, bboxes, keypoints)

        cv2.imshow(img_filename, color_image)
        k = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

    @torch.no_grad()
    def visualize_predicted(self, img_filename, keypoints, bboxes):

        cv2.namedWindow(img_filename, cv2.WINDOW_AUTOSIZE)


        # Convert images to numpy arrays
        color_image = cv2.imread(img_filename)
        #color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        color_image_tensor = [F.to_tensor(color_image)]
        color_image_tensors = list(image.to(self.device) for image in color_image_tensor)


        image = (color_image_tensors[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)



        #keypoints = self.keypoint_prediction_dict[key][idx]
        #bboxes = self.bboxes_prediction_dict[key]

        color_image = self.draw_precise_keypoints(image, bboxes, keypoints)

        cv2.imshow(img_filename, color_image)
        k = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

    def run_experiment(self, combination='ResNet50+triangulation', mode='Auto', n_runs = 100):
        if combination == 'ResNet50+triangulation':
            pass
        else:
            raise ValueError('Unknown experiment combination yet')
        experiment_results = pd.DataFrame()

        # generate random combinations of frames
        combinations = []
        for i in range(n_runs):
            idx = random.sample(range(100), 4)
            combinations.append(idx)

        # triangulate points
        triangulated_points_list = []

        for combination in combinations:
            #print(combination)
            tcp0_list = []
            frame_points_list = []
            frame_bboxes_list = []
            for idx, key in zip(combination, self.img_fns_dict.keys()):
                #print(self.img_fns_dict[key][idx])
                filename_id = self.img_fns_dict[key][idx].split('/')[-1].split('.')[0].split('_')[-1]
                #print(filename_id)
                #load tcp0 data
                json_filename = self.img_fns_dict[key][idx].split('.')[0].split('/')[-1]
                json_filename = json_filename[5:]
                json_filename = "meta"+json_filename+".json"
                json_path = "/".join(self.img_fns_dict[key][idx].split('.')[0].split('/')[:-1])
                json_full_name = '/'.join([json_path, json_filename])
                with open(json_full_name) as json_file:
                    json_data = json.load(json_file)
                    tcp0_list.append(json_data['l'])

                # retrieve_prediction
                #frame_points_list.append(self.keypoint_prediction_dict[key][idx])
                frame_bboxes_list.append(self.bboxes_prediction_dict[key][idx])

                frame_points_list.append([float(coordiante) for coordiantes in self.keypoint_prediction_dict[key][idx][0] for coordiante in coordiantes])
                #print([coordiante for coordiantes in self.keypoint_prediction_dict[key][idx][0] for coordiante in coordiantes])

                #self.visualize_predicted(self.img_fns_dict[key][idx], self.keypoint_prediction_dict[key][idx], self.bboxes_prediction_dict[key][idx])

            # run triangulation
            triangulation_idxs = [[0, 1],
                                [1, 2],
                                [2, 3],
                                [3, 0],
                                [1, 3],
                                [0, 2]]

            points_in3d_robot_base = []
            for triangulation_idx in triangulation_idxs:
                points_in3d_robot_base.append(
                    triangulate(frame_points_list[triangulation_idx[0]], frame_points_list[triangulation_idx[1]],
                                tcp0_list[triangulation_idx[0]], tcp0_list[triangulation_idx[1]],
                                self.R_cam2gripper, self.t_cam2gripper, self.mtx))
                points_in3d_robot_base[-1] = points_in3d_robot_base[-1] / 1000
            print(np.mean(np.asarray(points_in3d_robot_base),axis=0))
            triangulated_points_list.append(np.mean(np.asarray(points_in3d_robot_base),axis=0))
        print('total: mean, std')
        print(np.mean(np.asarray(triangulated_points_list), axis=0))
        print(np.std(np.asarray(triangulated_points_list), axis=0))
        true = np.mean(np.asarray(triangulated_points_list), axis=0)
        errors = []
        for prediction in triangulated_points_list:
            errors.append(np.asarray(list(map(abs, prediction - true))))

        mean_errors = np.mean(errors, axis=0)
        for idx, mean_error in enumerate(mean_errors):
            print(idx, mean_error)


        #for img_filename in self.img_fns:
            #self.visualize_precise(img_filename)

    # pose estimators: return points_in_3d_robot_base

    def experiment_series_calibrations(self):
        for calib in calibs:
            pass

go = Experiment([1])
go.run_experiment()

# results for calibration

"""
#calib 1
 [0.00208565 0.00070694 0.00074757] 2
 [0.00197124 0.0006425  0.0007345 ] 3
 [0.00196008 0.00059219 0.00075397] 4
 [0.00182657 0.00060183 0.00074379] 5 
 [0.00219104 0.00071952 0.0007573 ] 7
 [0.0017901  0.00058676 0.00075572] 8
 [0.00189482 0.00065259 0.0008035 ] 10
 [0.00214337 0.00065814 0.0008262 ] 11
 [0.001725   0.00057457 0.00079802] 12
 [0.00181675 0.00056305 0.00082209] 13
 [0.00193651 0.00056058 0.00090502] 14
 
# calib Viktor v7
2 [0.00213315 0.00055339 0.00056164]
3 [0.00190465 0.00050112 0.0005157 ]
4 [0.00172369 0.00044183 0.00047065]
5 [0.00155794 0.00048807 0.00044997]
7 [0.00300786 0.00054191 0.00061499]
8 [0.00159031 0.00047226 0.00046024]
10 [0.0022484  0.00055372 0.00051257]
11 [0.00205358 0.00047374 0.00050283]
12 [0.00163509 0.00047594 0.00050498]
13 [0.00157606 0.00045998 0.00052292]
14 [0.00188364 0.00043226 0.00060689]

# calib Viktor v5
2 [0.00127093 0.00053285 0.00048216]
3 [0.00133445 0.00053744 0.00047974]
4 [0.0014503  0.00062773 0.0005177 ]
5 [0.00132445 0.00059852 0.00050302]
7 [0.00129707 0.00055117 0.00049087]
8 [0.00137524 0.00059698 0.00051994]
10 [0.00112714 0.00052448 0.00050418]
11 [0.00115603 0.00049615 0.00050188]
12 [0.00132253 0.00052498 0.00052107]
13 [0.0014891  0.00057788 0.00056779]
14 [0.00125592 0.00048839 0.00052767]
 """



