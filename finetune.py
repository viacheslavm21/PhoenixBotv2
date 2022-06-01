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
from dirty_calib_test import Experiment, CalibrationModel
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



def run_experiment(args):

    go = Experiment([1], save_or_read='save')

    pool = args

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

    n = 10

    begin_down = pool.begin_down
    end_down = pool.end_down
    img_count = 0
    for y in np.linspace(begin_down[1],end_down[1],n):
        for z in np.linspace(begin_down[2],end_down[2],n):
            ori_down = pool.ori_down

            for i in range(3):
                ori_down[i] = ori_down[i] + np.random.uniform(-0.05, 0.05)
            pose_vec = [begin_down[0],y,z,ori_down[0],ori_down[1],ori_down[2]]
            robot.movel(pose_vec, vel=0.25, acc=0.35)
            save_data(pool, img_count, 'down')
            img_count += 1
            sleep(1.0)

    begin_right = pool.begin_right
    end_right = pool.end_right

    img_count = 0
    for y in np.linspace(begin_right[1],end_right[1],n):
        for z in np.linspace(begin_right[2],end_right[2],n):
            ori_right = pool.ori_right

            for i in range(3):
                ori_right[i] = ori_right[i] + np.random.uniform(-0.05, 0.05)
            pose_vec = [begin_right[0],y,z,ori_right[0],ori_right[1],ori_right[2]]
            robot.movel(pose_vec, vel=0.25, acc=0.35)
            save_data(pool, img_count, 'right')
            img_count += 1
            sleep(1.0)


    begin_left = pool.begin_left
    end_left = pool.end_left

    img_count = 0
    for y in np.linspace(begin_left[1],end_left[1],n):
        for z in np.linspace(begin_left[2],end_left[2],n):
            ori_left = pool.ori_left
            for i in range(3):
                ori_left[i] = ori_left[i] + np.random.uniform(-0.05, 0.05)
            pose_vec = [begin_left[0],y,z,ori_left[0],ori_left[1],ori_left[2]]
            robot.movel(pose_vec,  vel=0.25, acc=0.35)
            save_data(pool, img_count, 'left')
            img_count+=1
            sleep(1.0)

    begin_up = pool.begin_up
    end_up = pool.end_up

    img_count = 0
    for y in np.linspace(begin_up[1],end_up[1],n):
        for z in np.linspace(begin_up[2],end_up[2],n):
            ori_up = pool.ori_up
            for i in range(3):
                ori_up[i] = ori_up[i] + np.random.uniform(-0.05, 0.05)
            pose_vec = [begin_up[0],y,z,ori_up[0],ori_up[1],ori_up[2]]
            robot.movel(pose_vec, vel=0.25, acc=0.35)
            save_data(pool, img_count, 'up')
            img_count += 1
            sleep(1.0)

    pool.text = f'Image collection is finished.'
    time.sleep(2)

    # pool.text = f'Optimizing camera calibration.'

    calib_model = CalibrationModel()

    mtx = np.asarray([[634.14800253, 0., 629.36554151],
                           [0., 634.13672323, 361.22908492],
                           [0., 0., 1., ]])
    R_cam2gripper = R.from_rotvec([-0.02193857, -0.03329439, -3.14092666]).as_matrix()
    t_cam2gripper = [[-11.93768264], [118.74955747], [207.06885955]]
    dist = np.asarray([[-0.053358, 0.05795461, -0.00042388, 0.00043828, -0.01121712]])

    calib_model = CalibrationModel(intrinsic=mtx, distortion=dist, name='finetune_calib')
    calib_model.fill_extrinsic(R_cam2gripper=R_cam2gripper,
                               t_cam2gripper=t_cam2gripper)

    # pool.text = f'Camera calibration is optimized.'
    # time.sleep(2)

    pool.text = f'Predicting socket points.'
    go.experiment_series_calibrations([calib_model], ['finetune_calib'])

    """
        begin_down = [0.8572998862773443, 0.04669753164975192, 0.55424054601676]
    end_down = [0.8572726748339248, -0.025224131112807004, 0.5728942086892206]

    img_count = 0
    for y in np.linspace(begin_down[1],end_down[1],n):
        for z in np.linspace(begin_down[2],end_down[2],n):
            ori_down = [1.265474089573716, 1.1185596344806958, 1.1163144977202624]

            for i in range(3):
                ori_down[i] = ori_down[i] + np.random.uniform(-0.05, 0.05)
            pose_vec = [begin_down[0],y,z,ori_down[0],ori_down[1],ori_down[2]]
            robot.movel(pose_vec, vel=0.25, acc=0.35)
            save_data(img_count, 'down')
            img_count += 1
            sleep(1.0)

    begin_right = [0.8580309371569254, -0.12110425235186006, 0.5916651774851597]
    end_right = [0.8580110126067423, -0.19531582008672357, 0.5667821685406375]

    img_count = 0
    for y in np.linspace(begin_right[1],end_right[1],n):
        for z in np.linspace(begin_right[2],end_right[2],n):
            ori_right = [1.0304238386050677, 1.527112162054058, 1.446301452005834]

            for i in range(3):
                ori_right[i] = ori_right[i] + np.random.uniform(-0.05, 0.05)
            pose_vec = [begin_right[0],y,z,ori_right[0],ori_right[1],ori_right[2]]
            robot.movel(pose_vec, vel=0.25, acc=0.35)
            save_data(img_count, 'right')
            img_count += 1
            sleep(1.0)


    begin_left = [0.8655120626548728, 0.08127379720628629, 0.57827360810733]
    end_left = [0.8655158757027868, 0.1341425776240957, 0.5506184797376877]

    img_count = 0
    for y in np.linspace(begin_left[1],end_left[1],n):
        for z in np.linspace(begin_left[2],end_left[2],n):
            ori_left = [ 1.3883282782993385, 0.8758122517997624, 0.8735844784399638]
            for i in range(3):
                ori_left[i] = ori_left[i] + np.random.uniform(-0.05, 0.05)
            pose_vec = [begin_left[0],y,z,ori_left[0],ori_left[1],ori_left[2]]
            robot.movel(pose_vec,  vel=0.25, acc=0.35)
            save_data(img_count, 'left')
            img_count+=1
            sleep(1.0)

    begin_up = [0.8363403845934372, -0.026240635691044466, 0.6973899081056515]
    end_up = [0.8363190728110094, 0.041234593738884454, 0.7220265810802895]

    img_count = 0
    for y in np.linspace(begin_up[1],end_up[1],n):
        for z in np.linspace(begin_up[2],end_up[2],n):
            ori_up = [1.550006780007305, 1.432632747485671, 0.8623123515022114]
            for i in range(3):
                ori_up[i] = ori_up[i] + np.random.uniform(-0.05, 0.05)
            pose_vec = [begin_up[0],y,z,ori_up[0],ori_up[1],ori_up[2]]
            robot.movel(pose_vec, vel=0.25, acc=0.35)
            save_data(img_count, 'up')
            img_count += 1
            sleep(1.0)
    """

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
@click.option('-camopt',
              '--optimize-camera-calibration',
              help="TBD",
              type=click.BOOL,
              is_flag=True,
              default=False,
              show_default=True)
def main(exposure_value: int, optimize_camera_calibration: bool):

    pool = FramePool()
    args = [pool]
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
