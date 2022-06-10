"""
Model fine-tuning
"""
import time
import click
import pyrealsense2 as rs
import numpy as np
import urx
import cv2
import torch
import os
from threading import Thread
from time import sleep
import json
from models.ResNet50_model import get_model as get_ResNet50
from calib_test import Experiment, CalibrationModel
from optimize_camera_calib import CameraOpt
from scipy.spatial.transform import Rotation as R

EXPOSURE_VALUE = 150

env_name = 'my_finetune'

ROBOT_IP = "192.168.88.100"

robot = urx.Robot(ROBOT_IP, use_rt=True)

pipeline = rs.pipeline()
config = rs.config()

prof = pipeline.start(config)

s = prof.get_device().query_sensors()[1]
s.set_option(rs.option.exposure, EXPOSURE_VALUE)

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The script requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

NUM_KEYPOINTS = 19

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

precise_model = get_ResNet50(num_keypoints=NUM_KEYPOINTS,
                                  weights_path='data/outputs/keypointsrcnn_weights_10_epoch50.pth',
                                  load_device=device)
precise_model.to(device)

def save_data(pool, img_count=0, where = None):

    filename_img = f'data/finetunes/{env_name}/raw/{where}/frame_env_{env_name}_{where}_' + str(img_count)
    filename_json = f'data/finetunes/{env_name}/raw/{where}/meta_env_{env_name}_{where}_' + str(img_count)

    try:
        # save raw
        cv2.imwrite(filename_img+'.jpg', pool.color_image)
        data = {"j": robot.getj(),'l': list(robot.get_pose().pose_vector)}
        with open(filename_json+'.json', 'w') as outfile:
            json.dump(data, outfile)
        print(str(img_count)+' saved')

    except FileNotFoundError:
        path = f'data/finetunes/{env_name}/raw/{where}'
        print(f"Directory {path} was not found. Creating the directory...")
        os.makedirs(path)
        print(f"Directory {path} was created.")
        save_data(pool, img_count, where)
    except Exception as e:
        print(e)


def mouse_callback(event, x, y, flags, param):
    pool = param
    if event == cv2.EVENT_LBUTTONDOWN:

        if pool.side == "Left":
            if pool.stage == 1:
                pool.begin_left = robot.getl()[:3]
                pool.ori_left = robot.getl()[3:]
            elif pool.stage == 2:
                pool.end_left = robot.getl()[:3]

        if pool.side == "Right":
            if pool.stage == 1:
                pool.begin_right = robot.getl()[:3]
                pool.ori_right = robot.getl()[3:]
            elif pool.stage == 2:
                pool.end_right = robot.getl()[:3]

        if pool.side == "Up":
            if pool.stage == 1:
                pool.begin_up = robot.getl()[:3]
                pool.ori_up = robot.getl()[3:]
            elif pool.stage == 2:
                pool.end_up = robot.getl()[:3]

        if pool.side == "Down":
            if pool.stage == 1:
                pool.begin_down = robot.getl()[:3]
                pool.ori_down = robot.getl()[3:]
            elif pool.stage == 2:
                pool.end_down = robot.getl()[:3]


def run_experiment(*args):
    pool, optimize_camera_calibration, debug = args

    # Step 1: Set boundaries for viewing

    # Left
    pool.side = "Left"
    pool.stage = 1
    pool.text = f'Set boundary {pool.stage} for {pool.side} side.'

    while True:
        if pool.begin_left != []:
            break

    pool.stage = 2
    pool.text = f'Set boundary {pool.stage} for {pool.side} side.'

    while True:
        if pool.end_left != []:
            break

    # Right
    pool.side = "Right"
    pool.stage = 1
    pool.text = f'Set boundary {pool.stage} for {pool.side} side.'

    while True:
        if pool.begin_right != []:
            break

    pool.stage = 2
    pool.text = f'Set boundary {pool.stage} for {pool.side} side.'

    while True:
        if pool.end_right != []:
            break

    # Up
    pool.side = "Up"
    pool.stage = 1
    pool.text = f'Set boundary {pool.stage} for {pool.side} side.'

    while True:
        if pool.begin_up != []:
            break

    pool.stage = 2
    pool.text = f'Set boundary {pool.stage} for {pool.side} side.'

    while True:
        if pool.end_up != []:
            break

    # Down
    pool.side = "Down"
    pool.stage = 1
    pool.text = f'Set boundary {pool.stage} for {pool.side} side.'

    while True:
        if pool.begin_down != []:
            break

    pool.stage = 2
    pool.text = f'Set boundary {pool.stage} for {pool.side} side.'

    while True:
        if pool.end_down != []:
            break

    pool.text = f'Wait until image collection is finished.'

    #print(pool.begin_down, pool.end_down, pool.begin_up, pool.end_up)

    n = 2


    begin_up = pool.begin_up
    end_up = pool.end_up
    img_count = 0
    print("Up boundaries")
    print(np.concatenate((begin_up,pool.ori_up)),
          np.concatenate((end_up,pool.ori_up)))

    if debug:
        begin_up = [0.76317143, 0.04606918, 0.61435412]
        end_up = [0.76316919, -0.02990835, 0.59236572]

    for y in np.linspace(begin_up[1],end_up[1],n):
        for z in np.linspace(begin_up[2],end_up[2],n):
            ori_up = pool.ori_up
            if debug:
                ori_up = [1.40903893,  1.36964326,  0.97971153]
            #for i in range(3):
            #    ori_up[i] = ori_up[i] + np.random.uniform(-0.05, 0.05)

            x = np.linspace(begin_up[0], end_up[0], n * 10)[img_count]
            pose_vec = [x, y, z, ori_up[0], ori_up[1], ori_up[2]]
            robot.movel(pose_vec, vel=0.25, acc=0.35)
            save_data(pool, img_count, 'up')
            img_count += 1
            sleep(1.0)

    begin_down = pool.begin_down
    end_down = pool.end_down
    img_count = 0
    print("Down boundaries")
    print(np.concatenate((begin_down,pool.ori_down)),
          np.concatenate((end_down,pool.ori_down)))

    if debug:
        begin_down = [0.76320912, - 0.0298756,   0.52913839]
        end_down = [0.76313912, 0.03570664, 0.493664]

    for y in np.linspace(begin_down[1], end_down[1], n):
        for z in np.linspace(begin_down[2], end_down[2], n):
            x = np.linspace(begin_down[0], end_down[0], n*10)[img_count]
            ori_down = pool.ori_down
            if debug:
                ori_down = [1.2142219, 1.15684038, 1.13323442]
            pose_vec = [x, y, z, ori_down[0], ori_down[1], ori_down[2]]
            robot.movel(pose_vec, vel=0.25, acc=0.35)
            save_data(pool, img_count, 'down')
            img_count += 1
            sleep(1.0)

    begin_right = pool.begin_right
    end_right = pool.end_right
    img_count = 0
    print("Right boundaries")
    print(np.concatenate((begin_right,pool.ori_right)),
          np.concatenate((end_right,pool.ori_right)))

    if debug:
        begin_right = [0.76320812, - 0.08503204,  0.53461637]
        end_right = [0.76315605, - 0.13947306, 0.55563164]


    for y in np.linspace(begin_right[1], end_right[1], n):
        for z in np.linspace(begin_right[2], end_right[2], n):
            ori_right = pool.ori_right
            if debug:
                ori_right = [1.10240255, 1.50591011, 1.31402318]
            #for i in range(3):
            #    ori_right[i] = ori_right[i] + np.random.uniform(-0.05, 0.05)

            x = np.linspace(begin_right[0], end_right[0], n * 10)[img_count]
            pose_vec = [x, y, z, ori_right[0], ori_right[1], ori_right[2]]
            robot.movel(pose_vec, vel=0.25, acc=0.35)
            save_data(pool, img_count, 'right')
            img_count += 1
            sleep(1.0)

    begin_left = pool.begin_left
    end_left = pool.end_left
    img_count = 0
    print("Left boundaries")
    print(np.concatenate((begin_left,pool.ori_left)),
          np.concatenate((end_left,pool.ori_left)))

    if debug:
        begin_left = [0.76315976, 0.03132591, 0.57389706]
        end_left = [0.76314426, 0.09697606, 0.53464401]

    for y in np.linspace(begin_left[1], end_left[1], n):
        for z in np.linspace(begin_left[2], end_left[2], n):
            ori_left = pool.ori_left
            if debug:
                ori_left = [1.4065642, 1.05634515, 0.89878902]
            #for i in range(3):
            #    ori_left[i] = ori_left[i] + np.random.uniform(-0.05, 0.05)

            x = np.linspace(begin_left[0], end_left[0], n * 10)[img_count]
            pose_vec = [x, y, z, ori_left[0], ori_left[1], ori_left[2]]
            robot.movel(pose_vec,  vel=0.25, acc=0.35)
            save_data(pool, img_count, 'left')
            img_count+=1
            sleep(1.0)


    pool.text = f'Image collection is finished.'
    time.sleep(2)

    # pool.text = f'Optimizing camera calibration.'

    mtx = np.asarray([[634.14800253, 0., 629.36554151],
                           [0., 634.13672323, 361.22908492],
                           [0., 0., 1., ]])
    R_cam2gripper = [ 1.08406012e-02,  4.97508834e-02,  3.13081739e+00]
    t_cam2gripper = [-1.15373626e+01,  1.14326648e+02,  2.13237716e+02]

    dist = np.asarray([[-0.053358, 0.05795461, -0.00042388, 0.00043828, -0.01121712]])

    calib_model = CalibrationModel(intrinsic=mtx, distortion=dist, name='finetune_calib')
    calib_model.fill_extrinsic(R_cam2gripper=R_cam2gripper,
                               t_cam2gripper=t_cam2gripper)

    go = Experiment(img_folder="data/finetunes/my_finetune", save_or_read='save')

    if optimize_camera_calibration:
        camera_opt = CameraOpt(env_name="data/finetunes/my_finetune", pred_filename_b="data/bboxes_prediction_dict_from_etune.pickle", pred_filename_k="data/keypoint_prediction_dict_from_etune.pickle",save_or_read='read', initial_calib_model=calib_model)
        #opt_hand_eye = camera_opt.optimize_calib_re_least_squares(calib_model)
        opt_hand_eye = camera_opt.optimize_calib_re_least_squares(calib_model)#, experiment=go)


        print("opt_hand_eye", opt_hand_eye)
        calib_model.fill_extrinsic(R_cam2gripper=opt_hand_eye[0],
                                   t_cam2gripper=opt_hand_eye[1])


    # pool.text = f'Camera calibration is optimized.'
    # time.sleep(2)
    pool.text = f'Predicting socket points.'
    go.experiment_series_calibrations([calib_model], ['finetune_calib'])

class FramePool:
    def __init__(self):
        self.color_image = None
        self.color_image_no_text = None
        self.text = None

        self.begin_left = []
        self.end_left = []
        self.ori_left = []

        self.begin_right = []
        self.end_right = []
        self.ori_right = []

        self.begin_up = []
        self.end_up = []
        self.ori_up = []

        self.begin_down = []
        self.end_down = []
        self.ori_down = []

        self.stage = None
        self.side = None

    def draw_text(self, color_image):
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX

        # org
        org = (50, 50)

        # fontScale
        fontScale = 1

        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2
        color_image = cv2.putText(color_image, self.text, org, font, fontScale, color, thickness)
        return  color_image

    def update(self):
        try:
            while True:
                time.sleep(1/120)
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                self.color_image_no_text = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

                if self.text:
                    color_image = self.draw_text(color_image)

                self.color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        finally:
            # Stop streaming
            pipeline.stop()


@click.command()
@click.option('-e',
              '--exposure-value',
              help="Set camera exposure",
              type=click.INT,
              is_flag=False,
              default=150,
              show_default=True)
@click.option('-rndori',
              '--randomize-orientation',
              help="Add random term to orientation during scanning",
              type=click.BOOL,
              is_flag=True,
              default=False,
              show_default=True)
@click.option('-camopt',
              '--optimize-camera-calibration',
              help="TBD",
              type=click.BOOL,
              is_flag=True,
              default=True,
              show_default=True)
@click.option('-d',
              '--debug',
              help="Debug mode",
              type=click.BOOL,
              is_flag=True,
              default=True,
              show_default=True)
def main(exposure_value: int, randomize_orientation: bool, optimize_camera_calibration: bool, debug: bool):
    pool = FramePool()
    args = [pool, optimize_camera_calibration, debug]
    moving_thread = Thread(target=run_experiment, daemon=True, args=args)
    moving_thread.start()
    frame_update_thread = Thread(target=pool.update, daemon=True)
    frame_update_thread.start()
    while True:
        time.sleep(1/60)
        # Show images
        cv2.namedWindow('Demo', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('Demo', mouse_callback, param=pool)
        #cv2.resizeWindow('Demo', 640, 480)
        cv2.imshow('Demo', pool.color_image)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
