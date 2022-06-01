### TEST OF REPROJECTION

import torch

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]

import pyrealsense2 as rs
import cv2, numpy as np, matplotlib.pyplot as plt, pandas as pd, urx, json, os, pickle

from PIL import Image
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.transforms import functional as F

from scipy.spatial.transform import Rotation as R
from scipy.optimize import differential_evolution
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

from calib_utils import CalibrationModel
from ba_utils import bundle_optimize


def download_and_fix_calib_objs():
    with open("/home/viacheslav/PycharmProjects/PhoenixBot/data/calib_models_ro", "rb") as f:
        calibs = pickle.load(f)

    for calib in calibs:
        # print(calib.name)
        if calib.name == 'calibration_v5_frames':
            calib.fill_extrinsic([5.75741543e-04, 1.24335210e-02, 3.14059231e+00],
                                 [-26.57405788, 109.38070665, 29.24488396])

        if calib.name == 'calibration_v7_frames':
            calib.fill_extrinsic([0.00746955, -0.04715531, 3.13828964], [-29.3115531, 107.42735831, 22.47290233])

        if calib.name == 'calibration_v9_frames':
            calib.fill_extrinsic([0.02825456, - 0.01947504, - 3.12670634], [-24.69554673, 130.25950539, 34.90702682])

        if calib.name == 'calib_horizontal':
            calib.fill_extrinsic([0.04419481, 0.06909873, - 3.12774781], [-18.4751669, 118.74709488, 26.0659704])

        # print(calib.intrs)

    valid_calibs = ['calib_horizontal', 'calibration_v5_frames', 'calibration_v7_frames', 'calibration_v9_frames']

    for calib in calibs:
        if calib.name == 'calib_horizontal':
            mtx = calib.intrs
            dist = calib.dist
            print(mtx)
            calibs.append(CalibrationModel(intrinsic=mtx, name='optimize_from_horizontal', distortion=dist))
            valid_calibs.append('optimize_from_horizontal')
            R_cam2gripper = [ 3.16455887e-02,  2.02689425e-02, -3.09905387e+00]
            t_cam2gripper = [ -2.34608647e+01, 1.09706787e+02,  2.64404465e+01]
    calibs[-1].fill_extrinsic(R_cam2gripper, t_cam2gripper)

    return calibs, valid_calibs

calibs, valid_calibs = download_and_fix_calib_objs()


class Experiment():
    def __init__(self, calib_model_name = 'optimize_from_horizontal'):

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


        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # Start streaming
        EXPOSURE_VALUE = 100
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
        self.device = torch.device('cpu')

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.precise_model = get_ResNet50(num_keypoints=self.NUM_KEYPOINTS,
                                          weights_path='data/outputs/keypointsrcnn_weights_10_epoch50.pth',
                                          load_device=self.device)
        self.precise_model.to(self.device)

        # camera calibration:
        calibs, valid_calibs = download_and_fix_calib_objs()
        for calib in calibs:
            if calib.name in valid_calibs:
                if calib.name == calib_model_name:
                    print(calib.name)
                    self.calib_model = calib
                    self.mtx = calib.intrs
                    self.R_cam2gripper = calib.R_cam2gripper
                    self.t_cam2gripper = calib.t_cam2gripper
                    self.dist = calib.dist

        self.true_3d_coords = [[ 0.96518636, -0.26507636,  0.69802452],
                                 [ 0.96399071, -0.28112136,  0.6971856 ],
                                 [ 0.96538993, -0.26112733,  0.68778541],
                                 [ 0.96480728, -0.26782145,  0.68725604],
                                 [ 0.96409182, -0.27730866,  0.68691323],
                                 [ 0.96367408, -0.28385627,  0.68682852],
                                 [ 0.96600849, -0.24819457,  0.68309264],
                                 [ 0.9650757 , -0.26426931,  0.68128601],
                                 [ 0.96388222, -0.28027113,  0.68063055],
                                 [ 0.96360128, -0.29633147,  0.68090438],
                                 [ 0.96520017, -0.25983636,  0.67437966],
                                 [ 0.96437056, -0.26777366,  0.67422841],
                                 [ 0.96390079, -0.27590752,  0.6738275 ],
                                 [ 0.96334759, -0.2836043 ,  0.67355824],
                                 [ 0.96414219, -0.27165068,  0.66712591],
                                 [ 0.96462228, -0.25833122,  0.66130493],
                                 [ 0.96261506, -0.28287719,  0.65936054],
                                 [ 0.96312134, -0.25241723,  0.65674694],
                                 [ 0.95880171, -0.28956061,  0.65328157]]
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


    def draw_precise_keypoints(self, image, bboxes, keypoints, image_original=None, bboxes_original=None,
                               keypoints_original=None):
        fontsize = 2
        for bbox in bboxes:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            image = cv2.rectangle(image.copy(), start_point, end_point, (0, 255, 0), 2)

        for kps in keypoints:
            for idx, kp in enumerate(kps):
                image = cv2.circle(image.copy(), (int(kp[0]), int(kp[1])), 2, (255, 0, 0), 2)

        return image

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
                # color_image_tensor = color_image_tensor.float()
                # print(color_image_tensor.size())
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
                # time.sleep(3)
        finally:
            # Stop streaming
            self.pipeline.stop()


    @torch.no_grad()
    def visualize_reproject(self):
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
                # color_image_tensor = color_image_tensor.float()
                # print(color_image_tensor.size())
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

                tcp0 = self.robot.getl()
                mtx = self.mtx
                dist = self.dist
                points_3d = self.true_3d_coords
                proxy = np.concatenate([self.R_cam2gripper, self.t_cam2gripper], axis=-1)
                # print('proxy',proxy)
                cam2gripper = np.concatenate([proxy, np.array([[0, 0, 0, 1]])], axis=0)
                ori = R.from_rotvec(np.array(tcp0[3:]))
                # print('ori', ori.as_rotvec())
                ori_m = ori.as_matrix()

                mm_data = np.array(tcp0[:3]) * 1000
                # print('mm_data',mm_data)
                proxy = np.concatenate([ori_m, mm_data.reshape((3, 1))], axis=-1)
                grip2base = np.concatenate([proxy, np.array([[0, 0, 0, 1]])], axis=0)

                # evaluation of cameras projection matrices
                cam2base = np.matmul(grip2base, cam2gripper)
                # print('cam2base', cam2base)
                # print('t_cam2base', cam2base[:3, 3])
                base2cam = np.linalg.inv(cam2base)

                rvec = R.from_matrix(base2cam[:3, :3]).as_rotvec()
                tvec = base2cam[:3, 3] / 1000
                imgpoints2, _ = cv2.projectPoints(np.asarray(points_3d), rvec, tvec, mtx, dist)


                bboxes = []
                for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                    bboxes.append(list(map(int, bbox.tolist())))

                color_image = self.draw_precise_keypoints(image, bboxes, imgpoints2)

                cv2.imshow('Inference', color_image)

                cv2.waitKey(1)
                # time.sleep(3)
        finally:
            # Stop streaming
            self.pipeline.stop()

    def ba_loss_draw(self, tcp0, images_pixels):
        mtx = self.mtx
        dist = self.dist
        points_3d = self.true_3d_coords
        proxy = np.concatenate([self.R_cam2gripper, self.t_cam2gripper], axis=-1)
        # print('proxy',proxy)
        cam2gripper = np.concatenate([proxy, np.array([[0, 0, 0, 1]])], axis=0)
        ori = R.from_rotvec(np.array(tcp0[3:]))
        # print('ori', ori.as_rotvec())
        ori_m = ori.as_matrix()

        mm_data = np.array(tcp0[:3]) * 1000
        # print('mm_data',mm_data)
        proxy = np.concatenate([ori_m, mm_data.reshape((3, 1))], axis=-1)
        grip2base = np.concatenate([proxy, np.array([[0, 0, 0, 1]])], axis=0)

        # evaluation of cameras projection matrices
        cam2base = np.matmul(grip2base, cam2gripper)
        # print('cam2base', cam2base)
        # print('t_cam2base', cam2base[:3, 3])
        base2cam = np.linalg.inv(cam2base)

        rvec = R.from_matrix(base2cam[:3, :3]).as_rotvec()
        tvec = base2cam[:3, 3] / 1000
        imgpoints2, _ = cv2.projectPoints(points_3d, rvec, tvec, mtx, dist)


        return imgpoints2

go = Experiment()

go.visualize_reproject()