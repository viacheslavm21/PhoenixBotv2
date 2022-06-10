### Refactored copy of dirty_calib_test.py

import torch
import cv2, numpy as np, matplotlib.pyplot as plt, pandas as pd, urx, json, os, pickle

import torchvision
from torchvision.transforms import functional as F

from scipy.spatial.transform import Rotation as R

import torchvision.models.detection.mask_rcnn
import random
import time

from insertion_utils import triangulate, triangulate_unflattened

from models.ResNet50_model import get_model as get_ResNet50

from insertion_utils import plane

from calib_utils import CalibrationModel, download_and_fix_calib_objs

def multiplicate_and_concatenate(array, n: int):
    out = np.asarray([])
    for i in range(n):
        out = np.concatenate((out, array))
    return out

class Experiment():
    def __init__(self, img_folder="big_experiment/test", weights_path='data/outputs/keypointsrcnn_weights_10_epoch50.pth',  save_or_read='save'):
        # initalize model
        self.NUM_KEYPOINTS = 19
        self.device = torch.device('cpu')

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.precise_model = get_ResNet50(num_keypoints=self.NUM_KEYPOINTS,
                                          weights_path=weights_path,
                                          load_device=self.device)
        self.precise_model.to(self.device)

        # calibrate camera
        self.mtx, self.dist, self.t_cam2gripper, self.R_cam2gripper = self.set_manual_calibration()

        # load experimental data
        self.img_fns = []
        self.img_fns_dict = {}
        self.down_img_filenames = []
        self.up_img_filenames = []
        self.left_img_filenames = []
        self.right_img_filenames = []
        print("looking at: ", img_folder)
        for path, dirs, files in os.walk(img_folder):
            print("looking at: ", img_folder)
            for file in files:
                if file[-3:] == 'jpg':
                    if path.split('/')[-1] not in self.img_fns_dict.keys():
                        self.img_fns_dict[path.split('/')[-1]] = [os.path.join(path, file)]
                    else:
                        self.img_fns_dict[path.split('/')[-1]].append(os.path.join(path, file))
                    self.img_fns.append(os.path.join(path, file))

        print(self.img_fns_dict)
        if save_or_read == 'read':
        # self.ke   ypoint_prediction_list, self.bboxes_prediction_list = self.predict_full_dataset()

            with open(f'data/keypoint_prediction_dict_from_{img_folder[-5:]}.pickle', 'rb') as f:
                self.keypoint_prediction_dict = pickle.load(f)
            with open(f'data/bboxes_prediction_dict_from_{img_folder[-5:]}.pickle', 'rb') as f:
                self.bboxes_prediction_dict = pickle.load(f)

        if save_or_read == 'save':
            self.keypoint_prediction_dict = {}
            self.bboxes_prediction_dict = {}

            for key in self.img_fns_dict.keys():
                self.keypoint_prediction_dict[key], self.bboxes_prediction_dict[key] = self.predict_list(
                    self.img_fns_dict[key])

            print(self.keypoint_prediction_dict)
            with open(f'data/keypoint_prediction_dict_from_{img_folder[-5:]}.pickle', 'wb') as f:
                pickle.dump(self.keypoint_prediction_dict, f)
            with open(f'data/bboxes_prediction_dict_from_{img_folder[-5:]}.pickle', 'wb') as f:
                pickle.dump(self.bboxes_prediction_dict, f)

    def set_manual_calibration(self):
        mtx =  np.asarray([[633.22, 0., 629.326],
               [0., 632.515, 360.888],
               [0., 0., 1.]])

        dist = np.asarray([[-0.053358, 0.05795461, -0.00042388, 0.00043828, -0.01121712]])
        # calib Viktor v5
        t_cam2gripper = [[-23.89993997], [112.4474997], [29.15469655]]

        R_cam2gripper = [[-9.99995414e-01, -3.01576701e-03, -2.79144817e-04],
                         [3.01316166e-03, -9.99955807e-01, 8.90537008e-03],
                         [-3.05989002e-04, 8.90448813e-03, 9.99960307e-01]]

        return mtx, dist, t_cam2gripper, R_cam2gripper

    def calibrate(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((8 * 5, 3), np.float32)

        # step size of the checkboard is 100 mm
        objp[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2) * 100

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 22]

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

        # check_calib_frames(images)

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
            # print(corners2)
            ret, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)
            rvecs.append(rvec)
            tvecs.append(tvec)

        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_gripper2base=dataposes_rot,
                                                            t_gripper2base=dataposes_shift,
                                                            R_target2cam=rvecs,
                                                            t_target2cam=tvecs,
                                                            method=cv2.CALIB_HAND_EYE_ANDREFF)
        self.dataposes_rot = dataposes_rot

        #print("intrinsic matrix from images")
        #print(mtx)

        #print("camera in gripper coordinates:")
        #print(t_cam2gripper)

        #print("rotation vector:")
        #print(R.from_matrix(R_cam2gripper).as_rotvec())

        return mtx, t_cam2gripper, R_cam2gripper

    def predict_list(self, filenames):
        keypoint_prediction_list = []
        bboxes_prediction_list = []

        for img_filename in filenames:
            color_image = cv2.imread(img_filename)
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
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        ### UNDISTORT IMAGE:
        h, w = color_image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (
        w, h))  # be careful , mtx, dst are taken from the global nspace
        # undistort
        #print(type(dist))
        #print(type(mtx))
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
                        # frame_points[-1] = np.append(frame_points[-1], self.predict_socket_points())
                        # print(self.predict_socket_points())
                        break
                    except Exception as e:
                        print(e)
                        continue
                frame_points[-1] = np.mean(look_points, axis=0)

        combinations = []
        # print(frame_points)

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
                points_in3d_robot_base[-1] = points_in3d_robot_base[-1] / 1000

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
        # print(f'mean x: {x}')

        for plane_idx in plane_combinations:
            my_plane = plane(points[plane_idx[0]], points[1], points[0])
            z = np.asarray([my_plane.a, my_plane.b, my_plane.c])
            z = z / np.linalg.norm(z)
            # print(f'z for combination: {z}')
            y = np.cross(z, x)
            x = np.cross(y, z)
            y_from_combinations.append(y)
            z_from_combinations.append(z)

        y = np.mean(y_from_combinations, axis=0)
        # print(f'mean y: {y}')

        z = np.mean(z_from_combinations, axis=0)
        # print(f'mean z: {z}')

        centerpoint = np.mean(points, axis=0)

        refpoint = np.asarray([x, y, z]).T @ [[0.02], [-0.08], [-0.12]]
        refpoint = refpoint.T + centerpoint
        refpoint = refpoint[0]

        r = R.from_matrix(np.asarray([x, y, z]).T).as_rotvec()
        reference_pose = np.concatenate((refpoint, r))
        # print(f'estimated reference pose {reference_pose}')
        return reference_pose

    def draw_precise_keypoints(self, image, bboxes, keypoints, image_original=None, bboxes_original=None,
                               keypoints_original=None):
        fontsize = 2
        for bbox in bboxes:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            image = cv2.rectangle(image.copy(), start_point, end_point, (0, 255, 0), 2)

        for kps in keypoints:
            for idx, kp in enumerate(kps):
                image = cv2.circle(image.copy(), tuple(kp), 2, (255, 0, 0), 2)

        return image

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
        #print(reprojection_error)
        #print(np.sum(reprojection_error))

        return np.sum(reprojection_error)

    def get_pose_error(self, ur3_pose_id):
        pose_in_ref = self.robot.getl()
        r = R.from_rotvec(pose_in_ref[3:])
        pose_in_ref[3:] = r.as_euler('xyz', degrees=False)
        pose_in_ref = np.asarray(pose_in_ref)
        pose_error = pose_in_ref - pts.true_poses[ur3_pose_id]

        pose_error_xyz_in_tcp = np.asarray(R.from_rotvec(self.robot.getl()[3:]).as_matrix()) @ pose_error[:3]
        pose_error_rpy_in_tcp = np.asarray(R.from_rotvec(self.robot.getl()[3:]).as_matrix()) @ pose_error[3:]

        pose_error_in_tcp = np.concatenate((pose_error_xyz_in_tcp, pose_error_rpy_in_tcp))

        return pose_error_in_tcp

    def draw_reprojected(self, img, points_3d, img_tcp0):

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

        imgpoints2, _ = cv2.projectPoints(points_3d, rvec, tvec, self.mtx, self.dist)
        # for e,_ in enumerate(images_pixels[i]):
        #    images_pixels[i][e] = [images_pixels[i][e]]

        # print('images_pixels[i]', images_pixels[i])
        # print('imgpoints2', imgpoints2)
        # draw_reprojected(images[i], images_pixels[i])
        # print('points')
        # print(img)
        image = cv2.imread(img)
        #print(imgpoints2)
        for kps in imgpoints2:

            for idx, kp in enumerate(kps):
                image = cv2.circle(image.copy(), (int(kp[0]), int(kp[1])), 2, (20, 32, 213), 2)

        cv2.namedWindow('Reprojection', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Reprojection', image)
        k = cv2.waitKey(0) & 0xFF

    @torch.no_grad()
    def visualize_precise(self, img_filename, only_valid = False):

        cv2.namedWindow(img_filename, cv2.WINDOW_AUTOSIZE)

        # Convert images to numpy arrays
        color_image = cv2.imread(img_filename)
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        color_image_tensor = [F.to_tensor(color_image)]
        color_image_tensors = list(image.to(self.device) for image in color_image_tensor)

        image = (color_image_tensors[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)

        keypoints, bboxes = self.predict_socket_points(image)

        if only_valid:
            #print(keypoints)
            #print(keypoints[0])
            keypoints_valid = []
            for idx, point in enumerate(keypoints[0]):
                #print(idx, point)
                if idx in [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14]:
                    keypoints_valid.append(point)
            keypoints = [keypoints_valid]

        color_image = self.draw_precise_keypoints(image, bboxes, keypoints)

        cv2.imshow(img_filename, color_image)
        k = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

    @torch.no_grad()
    def visualize_predicted(self, img_filename, keypoints, bboxes):

        cv2.namedWindow(img_filename, cv2.WINDOW_AUTOSIZE)

        # Convert images to numpy arrays
        color_image = cv2.imread(img_filename)
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        color_image_tensor = [F.to_tensor(color_image)]
        color_image_tensors = list(image.to(self.device) for image in color_image_tensor)

        image = (color_image_tensors[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)

        # keypoints = self.keypoint_prediction_dict[key][idx]
        # bboxes = self.bboxes_prediction_dict[key]

        color_image = self.draw_precise_keypoints(image, bboxes, keypoints)

        cv2.imshow(img_filename, color_image)
        k = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

    def run_experiment_plot(self, combination='ResNet50+triangulation', mode='Auto', n_runs=100, combination_type='random'):
        if combination == 'ResNet50+triangulation':
            pass
        else:
            raise ValueError('Unknown experiment combination yet')
        experiment_results = pd.DataFrame()

        # print(self.mtx)
        # print(self.R_cam2gripper)
        # print(self.t_cam2gripper)

        if combination_type == 'random':
            # generate random combinations of frames
            combinations = []
            for i in range(n_runs):
                idx = random.sample(range(100), 4)
                combinations.append(idx)

        elif combination_type == 'corresponding':
            combinations = []
            for i in range(n_runs):
                combinations.append([i,i,i,i])

        # triangulate points
        triangulated_points_list = []

        for combination in combinations:
            # print(combination)
            tcp0_list = []
            frame_points_list = []
            frame_bboxes_list = []
            for idx, key in zip(combination, self.img_fns_dict.keys()):
                # print(self.img_fns_dict[key][idx])
                filename_id = self.img_fns_dict[key][idx].split('/')[-1].split('.')[0].split('_')[-1]
                # print(filename_id)
                # load tcp0 data
                json_filename = self.img_fns_dict[key][idx].split('.')[0].split('/')[-1]
                json_filename = json_filename[5:]
                json_filename = "meta" + json_filename + ".json"
                json_path = "/".join(self.img_fns_dict[key][idx].split('.')[0].split('/')[:-1])
                json_full_name = '/'.join([json_path, json_filename])
                with open(json_full_name) as json_file:
                    json_data = json.load(json_file)
                    tcp0_list.append(json_data['l'])

                # retrieve_prediction
                # frame_points_list.append(self.keypoint_prediction_dict[key][idx])
                frame_bboxes_list.append(self.bboxes_prediction_dict[key][idx])

                frame_points_list.append(
                    [float(coordiante) for coordiantes in self.keypoint_prediction_dict[key][idx][0] for coordiante in
                     coordiantes])
                # print([coordiante for coordiantes in self.keypoint_prediction_dict[key][idx][0] for coordiante in coordiantes])

                # self.visualize_predicted(self.img_fns_dict[key][idx], self.keypoint_prediction_dict[key][idx], self.bboxes_prediction_dict[key][idx])

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
            # print(np.mean(np.asarray(points_in3d_robot_base), axis=0))
            triangulated_points_list.append(np.mean(np.asarray(points_in3d_robot_base), axis=0))
            triangulated_points_mean_by_point = np.mean(triangulated_points_list,axis=1)
        print("        print(triangulated_points_list)")
        print(len(triangulated_points_mean_by_point))
        points_x = np.asarray(triangulated_points_mean_by_point)[:, 0]
        points_z = np.asarray(triangulated_points_mean_by_point)[:, 1]
        points_y = np.asarray(triangulated_points_mean_by_point)[:, 2]

        #plot_error(points_x)
        #plot_error(points_y)
        #plot_error(points_z)
        # print('total: mean, std')
        # print(np.mean(np.asarray(triangulated_points_list), axis=0))
        # print(np.std(np.asarray(triangulated_points_list), axis=0))
        true = np.mean(np.asarray(triangulated_points_list), axis=0)
        valid_true = []
        for idx, point in enumerate(true):
            if idx in [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14]:
                valid_true.append(point)

        print('true', np.mean(valid_true, axis=0))
        errors = []
        errors_x = []
        errors_y = []
        errors_z = []
        for prediction in triangulated_points_list:
            errors.append(np.asarray(list(map(abs, prediction - true))))

            for idx, last_error in enumerate(errors[-1]):
                if idx in [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14]:
                    # print(idx, mean_error)
                    errors_x.append(last_error[0])
                    errors_y.append(last_error[1])
                    errors_z.append(last_error[2])

        print("max errors_x", np.max(np.asarray(errors_x)))
        print("max errors_y", np.max(np.asarray(errors_y)))
        print("max errors_z", np.max(np.asarray(errors_z)))

        mean_errors = np.mean(errors, axis=0)
        valid_mean_errors = []
        for idx, mean_error in enumerate(mean_errors):
            if idx in [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14]:
                # print(idx, mean_error)
                valid_mean_errors.append(mean_error)

        return valid_mean_errors

    def run_experiment(self, combination='ResNet50+triangulation', mode='Auto', n_runs=1000, combination_type='random'):
        if combination == 'ResNet50+triangulation':
            pass
        else:
            raise ValueError('Unknown experiment combination yet')
        experiment_results = pd.DataFrame()


        #print(self.mtx)
        #print(self.R_cam2gripper)
        #print(self.t_cam2gripper)

        if combination_type == 'random':
            # generate random combinations of frames
            combinations = []
            for i in range(n_runs):
                idx = random.sample(range(100), 4)
                idx = random.sample(range(100), 4)
                combinations.append(idx)

        elif combination_type == 'corresponding':
            combinations = []
            for i in range(n_runs):
                combinations.append([i, i, i, i])
        # triangulate points
        triangulated_points_list = []

        for combination in combinations:
            # print(combination)
            tcp0_list = []
            frame_points_list = []
            frame_bboxes_list = []
            for idx, key in zip(combination, self.img_fns_dict.keys()):
                # print(self.img_fns_dict[key][idx])
                #filename_id = self.img_fns_dict[key][idx].split('/')[-1].split('.')[0].split('_')[-1]
                # print(filename_id)
                # load tcp0 data
                json_filename = self.img_fns_dict[key][idx].split('.')[0].split('/')[-1]
                json_filename = json_filename[5:]
                json_filename = "meta" + json_filename + ".json"
                json_path = "/".join(self.img_fns_dict[key][idx].split('.')[0].split('/')[:-1])
                json_full_name = '/'.join([json_path, json_filename])
                with open(json_full_name) as json_file:
                    json_data = json.load(json_file)
                    tcp0_list.append(json_data['l'])

                # retrieve_prediction
                # frame_points_list.append(self.keypoint_prediction_dict[key][idx])
                frame_bboxes_list.append(self.bboxes_prediction_dict[key][idx])

                frame_points_list.append(
                    [float(coordiante) for coordiantes in self.keypoint_prediction_dict[key][idx][0] for coordiante in
                     coordiantes])
                # print([coordiante for coordiantes in self.keypoint_prediction_dict[key][idx][0] for coordiante in coordiantes])

                # self.visualize_predicted(self.img_fns_dict[key][idx], self.keypoint_prediction_dict[key][idx], self.bboxes_prediction_dict[key][idx])

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
            #print(np.mean(np.asarray(points_in3d_robot_base), axis=0))
            triangulated_points_list.append(np.mean(np.asarray(points_in3d_robot_base), axis=0))


        points_x = np.asarray(triangulated_points_list)[:,0]
        points_z = np.asarray(triangulated_points_list)[:, 1]
        points_y = np.asarray(triangulated_points_list)[:, 2]
        #print('total: mean, std')
        #print(np.mean(np.asarray(triangulated_points_list), axis=0))
        #print(np.std(np.asarray(triangulated_points_list), axis=0))
        true = np.mean(np.asarray(triangulated_points_list), axis=0)
        #print("true",true)
        valid_true = []
        for idx, point in enumerate(true):
            if idx in [2,3,4,5,7,8,10,11,12,13,14]:
                valid_true.append(point)

        print('true', np.mean(valid_true, axis=0))
        errors = []
        errors_x = []
        errors_y = []
        errors_z = []
        for prediction in triangulated_points_list:
            errors.append(np.asarray(list(map(abs, prediction - true))))
            for idx, last_error in enumerate(errors[-1]):
                if idx in [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14]:
                    # print(idx, mean_error)
                    errors_x.append(last_error[0])
                    errors_y.append(last_error[1])
                    errors_z.append(last_error[2])

        print("max errors_x", np.max(np.asarray(errors_x)))
        print("max errors_y", np.max(np.asarray(errors_y)))
        print("max errors_z", np.max(np.asarray(errors_z)))

        mean_errors = np.mean(errors, axis=0)
        valid_mean_errors = []
        for idx, mean_error in enumerate(mean_errors):
            if idx in [2, 3, 4, 5,
                       7, 8,
                       10, 11, 12, 13,
                       14]:
                #print(idx, mean_error)
                valid_mean_errors.append(mean_error)

        return valid_mean_errors

    def run_pose_detection_experiment(self, combination='ResNet50+triangulation', mode='Auto', n_runs=1000, combination_type='random', show_max_error = False, show_all = False):
        if combination == 'ResNet50+triangulation':
            pass
        else:
            raise ValueError('Unknown experiment combination yet')
        experiment_results = pd.DataFrame()

        # Define combinations for triangulation

        if combination_type == 'random':
            # generate random combinations of frames
            combinations = []
            for i in range(n_runs):
                idx = random.sample(range(100), 4)
                combinations.append(idx)

        elif combination_type == 'corresponding':
            combinations = []
            for i in range(n_runs):
                combinations.append([i, i, i, i])
        # triangulate points
        triangulated_points_list = []
        reference_poses_list = []

        x_list = []
        y_list = []
        z_list = []
        center_point_list = []

        ### ONE RUN:

        combinations_images = []
        combinations_keypoint_prediction_list = []
        combinations_keypoint_pixel_prediction_list = []
        combinations_tcp0_list = []

        for combination in combinations:

            # Retrieve kp predictions and tcp coords
            tcp0_list = []
            frame_points_list = []
            valid_frame_points_list = []
            frame_bboxes_list = []
            combination_images = []

            for idx, key in zip(combination, self.img_fns_dict.keys()):
                combination_images.append(self.img_fns_dict[key][idx])
                # load tcp0 data
                json_filename = self.img_fns_dict[key][idx].split('.')[0].split('/')[-1]
                json_filename = json_filename[5:]
                json_filename = "meta" + json_filename + ".json"
                json_path = "/".join(self.img_fns_dict[key][idx].split('.')[0].split('/')[:-1])
                json_full_name = '/'.join([json_path, json_filename])
                with open(json_full_name) as json_file:
                    json_data = json.load(json_file)
                    tcp0_list.append(json_data['l'])

                # retrieve_prediction
                frame_bboxes_list.append(self.bboxes_prediction_dict[key][idx])
                frame_points_list.append(
                    [float(coordiante) for coordiantes in self.keypoint_prediction_dict[key][idx][0] for coordiante in
                     coordiantes])
                valid_idx = [2, 3, 4, 5,
                             7, 8,
                             10, 11, 12, 13,
                             14]
                valid_frame_points_list.append(np.reshape(np.array(frame_points_list[-1]), (-1, 19, 2))[0][valid_idx])




                # self.visualize_predicted(self.img_fns_dict[key][idx], self.keypoint_prediction_dict[key][idx], self.bboxes_prediction_dict[key][idx])

            # run triangulation
            triangulation_idxs = [[0, 1],
                                  [1, 2],
                                  [2, 3],
                                  [3, 0],
                                  [1, 3],
                                  [0, 2]]


            points_in3d_robot_base = []
            reference_poses = []
            x_s = []
            y_s = []
            z_s = []
            center_point_s = []
            triangulation_points_list = []
            # fill prediction by triangulation idx
            for triangulation_idx in triangulation_idxs:
                points_in3d_robot_base.append(
                    triangulate(frame_points_list[triangulation_idx[0]], frame_points_list[triangulation_idx[1]],
                                tcp0_list[triangulation_idx[0]], tcp0_list[triangulation_idx[1]],
                                self.R_cam2gripper, self.t_cam2gripper, self.mtx))
                points_in3d_robot_base[-1] = points_in3d_robot_base[-1] / 1000
                # calculate pose error:

                # pose:
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

                for direction_idx in direction_combinations:
                    x = points[direction_idx[0]] - points[direction_idx[1]]
                    x = x / np.linalg.norm(x)
                    x_from_combinations.append(x)

                x = np.mean(x_from_combinations, axis=0)
                # print(f'mean x: {x}')

                x_from_combinations = []
                y_from_combinations = []
                z_from_combinations = []

                for plane_idx in plane_combinations:
                    my_plane = plane(points[plane_idx[0]], points[plane_idx[1]], points[plane_idx[2]])

                    ### EXPERIMETNAL PLANE
                    point1 = np.mean(np.asarray([points[0], points[1], points[4]]), axis=0)
                    point2 = np.mean(np.asarray([points[2], points[3], points[5]]), axis=0)
                    point3 = np.mean(np.asarray([points[7], points[8], points[10]]), axis=0)
                    #my_plane = plane(point1, point2, point3)

                    z = np.asarray([my_plane.a, my_plane.b, my_plane.c])
                    z = z / np.linalg.norm(z)
                    # print(f'z for combination: {z}')
                    y = np.cross(z, x)
                    x = np.cross(y, z)
                    x_from_combinations.append(x)
                    y_from_combinations.append(y)
                    z_from_combinations.append(z)

                # print(f'mean x: {x}')

                y = np.mean(y_from_combinations, axis=0)
                # print(f'mean y: {y}')

                z = np.mean(z_from_combinations, axis=0)
                # print(f'mean z: {z}')

                x_s.append(x)
                y_s.append(y)
                z_s.append(z)

                centerpoint = np.mean(points, axis=0)
                center_point_s.append(centerpoint)

                refpoint = np.asarray([x, y, z]).T @ [[0.000001], [-0.000001], [-0.000001]]
                refpoint = refpoint.T + centerpoint
                refpoint = refpoint[0]

                r = R.from_matrix(np.asarray([x, y, z]).T).as_rotvec()
                reference_pose = np.concatenate((refpoint, r))
                reference_poses.append(reference_pose)
                triangulation_points_list.append(points)

            ### CALCULATE REPROJECTION ERROR
            re_losses = []
            for image, img_tcp0, image_keypoints in zip(combination_images, tcp0_list,
                                                        valid_frame_points_list):

                re_loss = self.re_loss(points_3d=np.mean(np.asarray(triangulation_points_list), axis=0),
                             image_pixels=image_keypoints,
                             img_tcp0=img_tcp0)

                re_losses.append(re_loss)

            if any([re_loss > 50 for re_loss in re_losses]):
                continue

            combinations_images.append(combination_images)
            combinations_tcp0_list.append(tcp0_list)
            combinations_keypoint_pixel_prediction_list.append(valid_frame_points_list)
            x_list.append(np.mean(np.asarray(x_s), axis=0))
            y_list.append(np.mean(np.asarray(y_s), axis=0))
            z_list.append(np.mean(np.asarray(z_s), axis=0))
            reference_poses_list.append(np.mean(np.asarray(reference_poses), axis=0))
            triangulated_points_list.append(np.mean(np.asarray(points_in3d_robot_base), axis=0))
            combinations_keypoint_prediction_list.append(np.mean(np.asarray(triangulation_points_list), axis=0))

        ### CALCULATE TRUE = mean(all_runs), and ERRORS
        print("N success: ", len(x_list))
        true = np.mean(np.asarray(triangulated_points_list), axis=0) # points
        # vectors:!!!
        true_x = np.mean(np.asarray(x_list), axis=0)
        true_y = np.mean(np.asarray(y_list), axis=0)
        true_z = np.mean(np.asarray(z_list), axis=0)
        true_pose = np.mean(np.asarray(reference_poses_list), axis=0)

        print('true_x', true_x)
        print('true_y', true_y)
        print('true_z', true_z)
        print('true_pose', true_pose)

        errors = []
        errors_x = []
        errors_y = []
        errors_z = []
        errors_vec_x = []
        errors_vec_y = []
        errors_vec_z = []
        errors_pose = []

        for x, y, z in zip(x_list, y_list, z_list):
            #errors_vec_x.append(np.abs(true_x - x))
            #errors_vec_y.append(np.abs(true_y - y))
            #errors_vec_z.append(np.abs(true_z - z))
            errors_vec_x.append(np.abs(angle_between_3d_vectors(true_x, x)))
            errors_vec_y.append(np.abs(angle_between_3d_vectors(true_y, y)))
            errors_vec_z.append(np.abs(angle_between_3d_vectors(true_z, z)))
        for prediction, pose_prediction in zip(triangulated_points_list, reference_poses_list):
            errors.append(np.asarray(list(map(abs, prediction - true))))

            errors_pose.append(np.asarray(list(map(abs, pose_prediction - true_pose))))

            errors_x.append(errors_pose[-1][0])
            errors_y.append(errors_pose[-1][1])
            errors_z.append(errors_pose[-1][2])

        if show_all:
            for idx in range(n_runs):
                for image, img_tcp0, image_keypoints in zip(combinations_images[idx], combinations_tcp0_list[idx], combinations_keypoint_pixel_prediction_list[idx]):
                    self.visualize_precise(image, only_valid=True)
                    self.draw_reprojected(image, combinations_keypoint_prediction_list[idx], img_tcp0)

                    self.re_loss(points_3d=combinations_keypoint_prediction_list[idx],
                                 image_pixels=image_keypoints,
                                 img_tcp0=img_tcp0)

                print("error_x vec", errors_vec_x[idx])
                print("error_y vec", errors_vec_y[idx])
                print("error_z vec", errors_vec_z[idx])

                print("error_x pose", errors_x[idx])
                print("error_y pose", errors_y[idx])
                print("error_z pose", errors_z[idx])

        if show_max_error == True:
            x_vec_max_idx = np.argmax(np.asarray(errors_vec_x), axis=0)
            y_vec_max_idx = np.argmax(np.asarray(errors_vec_y), axis=0)
            z_vec_max_idx = np.argmax(np.asarray(errors_vec_z), axis=0)

            x_pose_max_idx = np.argmax(np.asarray(errors_x))
            x_pose_max_idx = np.argmax(np.asarray(errors_y))
            x_pose_max_idx = np.argmax(np.asarray(errors_z))

            for image, img_tcp0 in zip(combinations_images[z_vec_max_idx], combinations_tcp0_list[z_vec_max_idx]):
                self.visualize_precise(image, only_valid=True)
                self.draw_reprojected(image, combinations_keypoint_prediction_list[z_vec_max_idx], img_tcp0)

            for image_keypoints, img_tcp0 in zip(combinations_keypoint_pixel_prediction_list[z_vec_max_idx], combinations_tcp0_list[z_vec_max_idx]):

                self.re_loss(points_3d=combinations_keypoint_prediction_list[z_vec_max_idx],
                             image_pixels=image_keypoints,
                             img_tcp0=img_tcp0)

            print("error_x vec", errors_vec_x[z_vec_max_idx])
            print("error_y vec", errors_vec_y[z_vec_max_idx])
            print("error_z vec", errors_vec_z[z_vec_max_idx])

            print("error_x pose", errors_x[z_vec_max_idx])
            print("error_y pose", errors_y[z_vec_max_idx])
            print("error_z pose", errors_z[z_vec_max_idx])

        print("max errors_x vec", np.max(np.asarray(errors_vec_x), axis=0))
        print("max errors_y vec", np.max(np.asarray(errors_vec_y), axis=0))
        print("max errors_z vec", np.max(np.asarray(errors_vec_z), axis=0))

        print("max errors_x pose", np.max(np.asarray(errors_x)))
        print("max errors_y pose", np.max(np.asarray(errors_y)))
        print("max errors_z pose", np.max(np.asarray(errors_z)))


        mean_errors = np.mean(errors, axis=0)
        mean_pose_error = np.mean(errors_pose, axis=0)
        mean_error_vec_x = np.mean(errors_vec_x, axis=0)
        mean_error_vec_y = np.mean(errors_vec_y, axis=0)
        mean_error_vec_z = np.mean(errors_vec_z, axis=0)

        print("mean_error_vec_x",mean_error_vec_x)
        print("mean_error_vec_y",mean_error_vec_y)
        print("mean_error_vec_z",mean_error_vec_z)

        return mean_pose_error, errors_x, errors_y, errors_z, errors_vec_x, errors_vec_y, errors_vec_z


    def experiment_series_calibrations(self, calibs, valid_calibs):
        print("Before all: self.R_cam2gripper:", self.R_cam2gripper)
        for calib in calibs:
            if calib.name in valid_calibs:
                print("-----------")
                print(calib.name)
                self.calib_model = calib
                self.dist = calib.dist
                self.mtx = calib.intrs
                self.R_cam2gripper = calib.R_cam2gripper
                self.t_cam2gripper = calib.t_cam2gripper
                print("After all: self.R_cam2gripper", self.R_cam2gripper)
                print('POINTS EXPERIMENT')
                valid_mean_errors = self.run_experiment(combination_type='random')
                #plot_error(np.asarray(valid_mean_errors)[:, 0])
                #plot_error(np.asarray(valid_mean_errors)[:, 1])
                #plot_error(np.asarray(valid_mean_errors)[:, 2])
                mean_error_over_keypoints = np.mean(valid_mean_errors, axis=0)
                print("mean_error_over_keypoints", mean_error_over_keypoints)

                print('POSE EXPERIMENT')
                # calculating pose:
                pose_error_mean, errors_x, errors_y, errors_z, errors_vec_x, errors_vec_y, errors_vec_z = self.run_pose_detection_experiment(combination_type='random', show_max_error = True)
                plot_error(np.asarray(errors_x)*1000, 'Linear error along z, mm')
                plot_error(np.asarray(errors_y)*1000, 'Linear error along y, mm')
                plot_error(np.asarray(errors_z)*1000, 'Linear error along x, mm')
                plot_error(np.asarray(errors_vec_x), 'Angular error around z, deg')
                plot_error(np.asarray(errors_vec_y), 'Angular error around y, deg')
                plot_error(np.asarray(errors_vec_z), 'Angular error around x, deg')

                print('mean_pose_error', pose_error_mean)


def angle_between_3d_vectors(vector1, vector2):
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)

    dot_product = np.dot(unit_vector1, unit_vector2)
    angle = np.arccos(dot_product) * 57.296
    return angle

def plot_error(error, title):
    n, bins, patches = plt.hist(error, 25, density=False, facecolor='#4178A4', alpha=0.75)
    plt.xlabel('Error')
    plt.ylabel('Count')
    plt.title(title)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    calibs, valid_calibs = download_and_fix_calib_objs()

    for calib in calibs:
        if calib.name == 'calib_horizontal':
            mtx = calib.intrs
            dist = calib.dist
            print(mtx)
            print(dist)
            calibs.append(CalibrationModel(intrinsic=mtx, name='optimize_from_horizontal', distortion=dist))
            valid_calibs.append('optimize_from_horizontal')
            R_cam2gripper = [ 3.16455887e-02,  2.02689425e-02, -3.09905387e+00]
            t_cam2gripper = [ -2.34608647e+01, 1.09706787e+02,  2.64404465e+01]
    calibs[-1].fill_extrinsic(R_cam2gripper, t_cam2gripper)
    #torch.multiprocessing.set_start_method('spawn')
    mtx = np.asarray([[634.14800253,   0.,         629.36554151],
           [  0. ,        634.13672323, 361.22908492],
           [  0. ,          0.,           1.        ]])
    dist = np.asarray([[-0.053358 ,   0.05795461, -0.00042388,  0.00043828, -0.01121712]])

    calib_model = CalibrationModel(intrinsic=mtx, distortion=dist, name='final_setup_calib')
    calib_model.fill_extrinsic(R_cam2gripper=[-0.02193857, -0.03329439, -3.14092666],
                               t_cam2gripper=[-11.93768264, 118.74955747, 207.06885955])
    go = Experiment([1], save_or_read='read')

    go.experiment_series_calibrations([calib_model], ['final_setup_calib'])