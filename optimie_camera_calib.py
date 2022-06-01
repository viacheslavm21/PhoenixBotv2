from typing import List

import torch

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]

# import sys
# sys.path.append('vision-main/references/detection')
# import transforms, utils, engine, train
# from utils import collate_fn

import pyrealsense2 as rs
import cv2, numpy as np, matplotlib.pyplot as plt, pandas as pd, urx, json, os, pickle

from PIL import Image
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.transforms import functional as F

from scipy.spatial.transform import Rotation as R
from scipy.optimize import differential_evolution, least_squares
from threading import Thread

import torchvision.models.detection.mask_rcnn
import random
import time
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from insertion_utils import triangulate, triangulate_unflattened
import data.experiments.insertion.input_points as pts

from models.ResNet50_model import get_model as get_ResNet50

from insertion_utils import plane

from calib_utils import CalibrationModel, download_and_fix_calib_objs
from ba_utils import bundle_optimize, bundle_optimize_view


def multiplicate_and_concatenate(array, n: int):
    out = np.asarray([])
    for i in range(n):
        out = np.concatenate((out, array))
    return out


class CameraOpt():
    def __init__(self, env_name, initial_calib_model, save_or_read='save'):

        self.NUM_KEYPOINTS = 19
        self.device = torch.device('cpu')

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.precise_model = get_ResNet50(num_keypoints=self.NUM_KEYPOINTS,
                                          weights_path='data/outputs/keypointsrcnn_weights_10_epoch50.pth',
                                          load_device=self.device)
        self.precise_model.to(self.device)

        self.R_cam2gripper = calib_model.R_cam2gripper
        self.t_cam2gripper = calib_model.t_cam2gripper
        self.mtx = calib_model.intrs
        self.dist = calib_model.dist


        # load experimental data
        self.img_fns = []
        self.img_fns_dict = {}
        self.down_img_filenames = []
        self.up_img_filenames = []
        self.left_img_filenames = []
        self.right_img_filenames = []

        for path, dirs, files in os.walk(f"data/big_experiment/{env_name}/raw"):
            for file in files:
                if file[-3:] == 'jpg':
                    if path.split('/')[-1] not in self.img_fns_dict.keys():
                        self.img_fns_dict[path.split('/')[-1]] = [os.path.join(path, file)]
                    else:
                        self.img_fns_dict[path.split('/')[-1]].append(os.path.join(path, file))
                    self.img_fns.append(os.path.join(path, file))

        # print(self.img_fns_dict)
        if save_or_read == 'read':
            # self.keypoint_prediction_list, self.bboxes_prediction_list = self.predict_full_dataset()

            with open(f'data/keypoint_prediction_dict_{env_name}.pickle', 'rb') as f:
                self.keypoint_prediction_dict = pickle.load(f)
            with open(f'data/bboxes_prediction_dict_{env_name}.pickle', 'rb') as f:
                self.bboxes_prediction_dict = pickle.load(f)

        if save_or_read == 'save':
            self.keypoint_prediction_dict = {}
            self.bboxes_prediction_dict = {}

            for key in self.img_fns_dict.keys():
                self.keypoint_prediction_dict[key], self.bboxes_prediction_dict[key] = self.predict_list(
                    self.img_fns_dict[key])
            with open(f'data/keypoint_prediction_dict_{env_name}.pickle', 'wb') as f:
                pickle.dump(self.keypoint_prediction_dict, f)
            with open(f'data/bboxes_prediction_dict_{env_name}.pickle', 'wb') as f:
                pickle.dump(self.bboxes_prediction_dict, f)

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
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        ### UNDISTORT IMAGE:
        h, w = color_image.shape[:2]
        #print(mtx)
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))  # be careful , mtx, dst are taken from the global nspace
        # undistort
        # print(type(dist))
        # print(type(mtx))
        undst = cv2.undistort(color_image, self.mtx, self.dist, None, newcameramtx)

        # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

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

    def reproject_and_draw(self, points_3d, calib_model: CalibrationModel):
        ### PREPARE img_points and img_tcp0s
        re_images = []
        tcp0_list = []
        frame_points_list = []
        frame_bboxes_list = []
        keys = multiplicate_and_concatenate(list(self.img_fns_dict.keys()), int(100 / 4))
        combination = multiplicate_and_concatenate(list(range(100)), int(4))
        for idx, key in zip(combination, keys):
            # load tcp0 data
            idx = int(idx)
            json_filename = self.img_fns_dict[key][idx].split('.')[0].split('/')[-1]
            json_filename = json_filename[5:]
            json_filename = "meta" + json_filename + ".json"
            json_path = "/".join(self.img_fns_dict[key][idx].split('.')[0].split('/')[:-1])
            json_full_name = '/'.join([json_path, json_filename])
            re_images.append(self.img_fns_dict[key][idx])
            with open(json_full_name) as json_file:
                json_data = json.load(json_file)
                tcp0_list.append(json_data['l'])
                # print(json_data['l'])

            # retrieve_prediction
            # frame_points_list.append(self.keypoint_prediction_dict[key][idx])
            frame_bboxes_list.append(self.bboxes_prediction_dict[key][idx])

            frame_points_list.append(
                [float(coordiante) for coordiantes in self.keypoint_prediction_dict[key][idx][0] for coordiante
                 in
                 coordiantes])

        # valid pixel coords
        frame_points_list_valid = []
        for points_2d in frame_points_list:
            # unflatten the keypoints:
            points_2d = np.reshape(np.array(points_2d), (-1, int(len(points_2d) / 2), 2)).tolist()
            # print(points_2d[0])
            valid_points = []
            for idx, point in enumerate(points_2d[0]):
                if idx in [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14]:
                    valid_points.append(point)
            frame_points_list_valid.append(valid_points)
        img_tcp0s = tcp0_list
        imgs_points = frame_points_list_valid
        print((R.from_matrix(calib_model.R_cam2gripper).as_rotvec(), np.asarray(calib_model.t_cam2gripper).ravel()))
        x0 = np.concatenate(
            (R.from_matrix(calib_model.R_cam2gripper).as_rotvec(), np.asarray(calib_model.t_cam2gripper).ravel()))
        points_3d = np.asarray(points_3d)

        """
        x0 = np.concatenate((R.from_matrix(calib_model.R_cam2gripper).as_rotvec(), np.asarray(calib_model.t_cam2gripper).ravel()))
                res = least_squares(self.re_loss_opt, x0, verbose=1, x_scale='jac', ftol=1e-4, method='lm', max_nfev=10000,
                            args=(objpoints_in_robot, calib_model.intrs, calib_model.dist, imgs_points, img_tcp0s))
           def re_loss_opt(self, params, points_3d, mtx, dist, images_pixels, img_tcp0s):
        """
        params = x0
        images_pixels = imgs_points
        R_cam2gripper = params[:3]
        t_cam2gripper = [[params[3]], [params[4]], [params[5]]]
        R_cam2gripper = R.from_rotvec(R_cam2gripper).as_matrix()
        proxy = np.concatenate([R_cam2gripper, t_cam2gripper], axis=-1)
        cam2gripper = np.concatenate([proxy, np.array([[0, 0, 0, 1]])], axis=0)
        grip2base_list = []
        base2cam_list = []
        rvecs_robot_in_camera = []
        tvecs_robot_in_camera = []

        for img_tcp0 in img_tcp0s:
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
        r_vecs = r_t_vecs[:amount_of_rvecs_params].reshape(-1, 3)
        t_vecs = r_t_vecs[amount_of_rvecs_params:].reshape(-1, 3)
        for i in range(r_vecs.shape[0]):
            imgpoints2, _ = cv2.projectPoints(points_3d, r_vecs[i], t_vecs[i], calib_model.intrs, calib_model.dist)
            self.draw_points(re_images[i], imgpoints2)
            imgpoints_arr.append(imgpoints2)

    def draw_points(self, img, points):
        #print('points')
        #print(img)
        image = cv2.imread(img)
        for kps in points:

            for idx, kp in enumerate(kps):
                image = cv2.circle(image.copy(), (int(kp[0]), int(kp[1])), 2, (210, 32, 13), 2)

        cv2.namedWindow('Reprojection', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Reprojection', image)
        k = cv2.waitKey(0) & 0xFF

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

        imgpoints2, _ = cv2.projectPoints(points_3d, rvec, tvec, self.mtx, self.dist)
        imgpoints2 = [i[0] for i in imgpoints2]
        # print("image_pixels")
        # print(image_pixels)
        # print('imgpoints2')
        # print(imgpoints2)

        # print("reprojection error")
        reprojection_error = np.absolute(np.asarray(imgpoints2) - image_pixels)
        # print(reprojection_error)
        # print(np.sum(reprojection_error))

        return np.sum(reprojection_error)

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
        # print(imgpoints2)
        for kps in imgpoints2:

            for idx, kp in enumerate(kps):
                image = cv2.circle(image.copy(), (int(kp[0]), int(kp[1])), 2, (20, 32, 213), 2)

        cv2.namedWindow('Reprojection', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Reprojection', image)
        k = cv2.waitKey(0) & 0xFF

    @torch.no_grad()
    def visualize_precise(self, img_filename, only_valid=False):

        cv2.namedWindow(img_filename, cv2.WINDOW_AUTOSIZE)

        # Convert images to numpy arrays
        color_image = cv2.imread(img_filename)
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        color_image_tensor = [F.to_tensor(color_image)]
        color_image_tensors = list(image.to(self.device) for image in color_image_tensor)

        image = (color_image_tensors[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)

        keypoints, bboxes = self.predict_socket_points(image)

        if only_valid:
            # print(keypoints)
            # print(keypoints[0])
            keypoints_valid = []
            for idx, point in enumerate(keypoints[0]):
                # print(idx, point)
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


    def re_loss_opt(self, params, points_3d, mtx, dist, images_pixels, img_tcp0s):

        R_cam2gripper = params[:3]
        t_cam2gripper = [[params[3]],[params[4]], [params[5]]]
        R_cam2gripper = R.from_rotvec(R_cam2gripper).as_matrix()
        proxy = np.concatenate([R_cam2gripper, t_cam2gripper], axis=-1)
        cam2gripper = np.concatenate([proxy, np.array([[0, 0, 0, 1]])], axis=0)
        grip2base_list = []
        base2cam_list = []
        rvecs_robot_in_camera = []
        tvecs_robot_in_camera = []

        for img_tcp0 in img_tcp0s:
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
        r_vecs = r_t_vecs[:amount_of_rvecs_params].reshape(-1, 3)
        t_vecs = r_t_vecs[amount_of_rvecs_params:].reshape(-1, 3)
        for i in range(r_vecs.shape[0]):

            imgpoints2, _ = cv2.projectPoints(points_3d, r_vecs[i], t_vecs[i], mtx, dist)

            imgpoints_arr.append(imgpoints2)

        return (np.array(imgpoints_arr).reshape((-1, n_keypoints, 2)) - images_pixels).ravel()


    def optimize_calib_re_least_squares(self, calib_model: CalibrationModel):
        """
        :param imgs_points: flattened keypoints
        :param img_tcp0s:
        :param calib_model:
        :return: optimized hand-eye calibration
        """

        # PREDICT 3D Keypoints over 1000 combinations
        combinations = []
        for i in range(10000):
            idx = random.sample(range(100), 20)
            combinations.append(idx)

        # triangulate points
        ba_optimized_points_list = []
        reference_poses_list = []

        keys = multiplicate_and_concatenate(list(self.img_fns_dict.keys()), int(20 / 4))

        triangulated_points_list = []
        for combination_id, combination in enumerate(combinations):

            tcp0_list = []
            frame_points_list = []
            frame_bboxes_list = []
            for idx, key in zip(combination, keys):
                filename_id = self.img_fns_dict[key][idx].split('/')[-1].split('.')[0].split('_')[-1]
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

            points_in3d_robot_base = np.mean(points_in3d_robot_base, axis=0)
            # I will run ba only with valid points
            # valid 3d coords
            points_in3d_robot_base_valid = []
            for idx, point in enumerate(points_in3d_robot_base):
                # print(idx, point)
                if idx in [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14]:
                    points_in3d_robot_base_valid.append(point)

            triangulated_points_list.append(points_in3d_robot_base)

        points_3d = np.mean(np.asarray(triangulated_points_list), axis=0)
        valid_points_3d = []
        for idx, point in enumerate(points_3d):
            if idx in [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14]:
                valid_points_3d.append(point)

        ### PREPARED points in 3D = valid_points_3d

        ### PREPARE img_points and img_tcp0s

        tcp0_list = []
        frame_points_list = []
        frame_bboxes_list = []
        keys = multiplicate_and_concatenate(list(self.img_fns_dict.keys()), int(400 / 4))
        combination = np.concatenate((np.asarray(list(range(100))), np.asarray(list(range(100))), np.asarray(list(range(100))), np.asarray(list(range(100)))))
        for idx, key in zip(combination, keys):

            # load tcp0 data
            idx = int(idx)
            json_filename = self.img_fns_dict[key][idx].split('.')[0].split('/')[-1]
            json_filename = json_filename[5:]
            json_filename = "meta" + json_filename + ".json"
            json_path = "/".join(self.img_fns_dict[key][idx].split('.')[0].split('/')[:-1])
            json_full_name = '/'.join([json_path, json_filename])

            with open(json_full_name) as json_file:
                json_data = json.load(json_file)
                tcp0_list.append(json_data['l'])
                # print(json_data['l'])

            # retrieve_prediction
            # frame_points_list.append(self.keypoint_prediction_dict[key][idx])
            frame_bboxes_list.append(self.bboxes_prediction_dict[key][idx])

            frame_points_list.append(
                [float(coordiante) for coordiantes in self.keypoint_prediction_dict[key][idx][0] for coordiante
                 in
                 coordiantes])

        # valid pixel coords
        frame_points_list_valid = []
        for points_2d in frame_points_list:
            # unflatten the keypoints:
            points_2d = np.reshape(np.array(points_2d), (-1, int(len(points_2d) / 2), 2)).tolist()
            # print(points_2d[0])
            valid_points = []
            for idx, point in enumerate(points_2d[0]):
                if idx in [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14]:
                    valid_points.append(point)
            frame_points_list_valid.append(valid_points)

        img_tcp0s = tcp0_list
        imgs_points = frame_points_list_valid
        print((R.from_matrix(calib_model.R_cam2gripper).as_rotvec(), np.asarray(calib_model.t_cam2gripper).ravel()))
        x0 = np.concatenate((R.from_matrix(calib_model.R_cam2gripper).as_rotvec(), np.asarray(calib_model.t_cam2gripper).ravel()))
        objpoints_in_robot = np.asarray(valid_points_3d)
        t0 = time.time()
        res = least_squares(self.re_loss_opt, x0, verbose=1, x_scale='jac', ftol=1e-12, method='lm', max_nfev=10000,
                            args=(objpoints_in_robot, calib_model.intrs, calib_model.dist, imgs_points, img_tcp0s))
        t1 = time.time()
        new_calib_model = CalibrationModel(intrinsic=calib_model.intrs, distortion=calib_model.dist, name='optimized_stupid')
        new_calib_model.fill_extrinsic(R_cam2gripper=res.x[:3], t_cam2gripper=res.x[3:])
        self.reproject_and_draw(objpoints_in_robot, new_calib_model)

        print("Optimization took {0:.0f} seconds".format(t1 - t0))

        n_keypoints = len(imgs_points[0])
        n_keypoints = int(n_keypoints)
        amount_of_points_coords = n_keypoints * 3

        print(f"res:{res}")

        #print(f"res:{res.x[:amount_of_points_coords].reshape(-1,3)}")

        return res.x[:amount_of_points_coords].reshape(-1, 3)

    def loss_positioning(self, params):
        # print('entered')
        self.R_cam2gripper = R.from_rotvec(params[:3]).as_matrix()
        self.t_cam2gripper = [[params[3]], [params[4]], [params[5]]]
        # print(self.R_cam2gripper, self.t_cam2gripper)
        valid_mean_errors = self.run_experiment()
        mean_error_over_keypoints = np.mean(valid_mean_errors, axis=0)
        loss = sum(mean_error_over_keypoints)
        print(loss)
        return loss

    def optimize_calib_genetic(self, calibs):
        for calib in calibs:
            if calib.name == 'calib_horizontal':
                self.mtx = calib.intrs
                self.R_cam2gripper = calib.R_cam2gripper
                self.t_cam2gripper = calib.t_cam2gripper
        R_cam2gripper, t_cam2gripper, mtx = R.from_matrix(self.R_cam2gripper).as_rotvec(), \
                                            np.asarray(self.t_cam2gripper).T[0], self.mtx
        print("Initial calibration", R_cam2gripper, t_cam2gripper, mtx)

        params = np.concatenate((R_cam2gripper, t_cam2gripper))
        print(params)
        p_bnds = [(params[0] - 0.05, params[0] + 0.05),
                  (params[1] - 0.05, params[1] + 0.05),
                  (params[2] - 0.05, params[2] + 0.05),
                  (params[3] - 5, params[3] + 5),
                  (params[4] - 10, params[4] + 5),
                  (params[5] - 10, params[5] + 5)]
        print(p_bnds)
        # reprojection_loss(params)
        # запуск оптимизации (генетический алгоритм)
        i = [0, ]
        res_de = differential_evolution(self.loss_positioning, p_bnds, x0=params, maxiter=100, tol=1e-05, atol=0.,
                                        disp=True, workers=1, popsize=10)  # callback=cb_de,
        print(res_de)


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
    mtx = [[634.14800253,   0.,         629.36554151],
           [  0. ,        634.13672323, 361.22908492],
           [  0. ,          0.,           1.        ]]
    dist = [[-0.053358 ,   0.05795461, -0.00042388,  0.00043828, -0.01121712]]

    calib_model = CalibrationModel(intrinsic=mtx, distortion=dist, name='final_setup_calib')
    calib_model.fill_extrinsic(R_cam2gripper=[-0.02193857, -0.03329439, -3.14092666],
                               t_cam2gripper=[-11.93768264, 118.74955747, 207.06885955])

    go = CameraOpt('config_13May_env_1', save_or_read='read', initial_calib_model = calib_model)
    go.optimize_calib_re_least_squares(calib_model)
