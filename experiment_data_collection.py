"""
Collecting data for experiment.
The script moves the robot in four sectors (relative to the socket): Left, Up, Right, Down
Images are captured and saved to corresponding folder
Robot joints and TCP coords are saved for each image
Keypoint predictions are also saved
"""

"""
This script starts streaming from the realsense camera and saves a picture every 5 sec
It also saves robot joint angles at moments of capturing pics.
РАЗ:
Это положение ЮР10 -= референскное
robot_tcp_inside = [0.74756322620669, -0.24124910971400937, 0.6571589923566876, 1.2859884030912865, 1.218086356838451, 1.1689460432621404] 
robot_joints_inside = [-3.7331581751452845, -1.361729923878805, 1.6842083930969238, -3.387172047291891, -1.021761719380514, -3.168464485798971]

Положение ЮР10 заподлецо с плоскостью сокета!!!
array([[-0.04987552],
       [ 0.00171541],
       [ 0.0030806 ]])
robot_tcp_near_near = [0.74756322620669-0.04987552, -0.24124910971400937+0.00171541, 0.6571589923566876+0.0030806, 1.2859884030912865, 1.218086356838451, 1.1689460432621404] 
Это ЮР3 в референсном полодении
socket_tcp_when_inside_was_measured = [0.06839188025398572, -0.25581951297427913, 0.500467729263251, 0.1224361512161317, 2.1149005464648805, -2.2813784552385434]
socket_joints_when_inside_was_measured = [-0.5414903799640101, -1.365213696156637, -1.6896942297564905, 0.05110299587249756, 0.5073274374008179, -0.03999644914735967]

ДВА: (то, что сейчас)
Это положение ЮР10 -= референскное
robot_tcp_inside = [0.6976284902891156, -0.23950485111314057, 0.6603168087177494, 1.285755333884626, 1.218086162165414, 1.1688770055777193]
robot_joints_inside = [2.5083203315734863, -1.4403899351703089, 1.7717814445495605, -3.3943570295916956, -0.9800494352923792, 3.111675262451172]

Положение ЮР10 заподлецо с плоскостью сокета!!!

robot_tcp_near_near = [0.7102000170196487, -0.2408435343018464, 0.6649601094040635, 1.3089278830058457, 1.2421023952167922, 1.1545812090091068]
robot_joints_near_near = [2.5182175636291504, -1.4311183134662073, 1.7462811470031738, -3.340491596852438, -0.990058724080221, 3.091484546661377]
Это ЮР3 в референсном полодении
socket_tcp_when_inside_was_measured = [0.06837427442206023, -0.25582019562876734, 0.5004765450793814, 0.1224733064660375, 2.1149185634237417, -2.281451671868745]
socket_joints_when_inside_was_measured = [-0.5415023008929651, -1.365201775227682, -1.6896584669696253, 0.051126956939697266, 0.507315456867218, -0.04006845155824834]
"""

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


EXPOSURE_VALUE = 150

env_id = 2

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
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

glob_img_count = 0

NUM_KEYPOINTS = 19

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

precise_model = get_ResNet50(num_keypoints=NUM_KEYPOINTS,
                                  weights_path='data/outputs/keypointsrcnn_weights_6_epoch260.pth',
                                  load_device=device)
precise_model.to(device)

def save_data(img_count=0, where = None):

    filename_img = f'data/big_experiment/config_23May_env_{env_id}/raw/{where}/frame_env_{env_id}_{where}_'+str(img_count)
    filename_json = f'data/big_experiment/config_23May_env_{env_id}/raw/{where}/meta_env_{env_id}_{where}_' + str(img_count)
    try:
        # save raw
        cv2.imwrite(filename_img+'.jpg', color_image)
        data = {"j": robot.getj(),'l': list(robot.get_pose().pose_vector)}
        with open(filename_json+'.json', 'w') as outfile:
            json.dump(data, outfile)
        print(str(img_count)+' saved')
    except Exception as e:
        print(e)

def run_experiment_may13():

    n = 10
    begin_up = [1.0174524446009747, -0.05234927656133891, 0.7918454271895835]
    end_up = [1.0174612601223068, 0.02893121462363131, 0.7462322025464855]
    img_count = 0
    for y in np.linspace(begin_up[1],end_up[1],n):
        for z in np.linspace(begin_up[2],end_up[2],n):
            ori_up = [2.211353712743308, 2.179004420655926, 0.027381350647709152]
            for i in range(3):
                ori_up[i] = ori_up[i] + np.random.uniform(-0.05, 0.05)
            pose_vec = [begin_up[0],y,z,ori_up[0],ori_up[1],ori_up[2]]
            robot.movel(pose_vec, vel=0.25, acc=0.35)
            save_data(img_count, 'up')
            img_count += 1
            sleep(0.4)

    begin_down = [1.0006994337712027, -0.06153509057735719, 0.6960766424821022]
    end_down = [1.0007086402790202, 0.03392554109797587, 0.7305328555444366]
    img_count = 0
    for y in np.linspace(begin_down[1],end_down[1],n):
        for z in np.linspace(begin_down[2],end_down[2],n):
            ori_down = [2.0228498514388646, 1.9933333553407875, 0.3766183178014909]

            for i in range(3):
                ori_down[i] = ori_down[i] + np.random.uniform(-0.05, 0.05)
            pose_vec = [begin_down[0],y,z,ori_down[0],ori_down[1],ori_down[2]]
            robot.movel(pose_vec, vel=0.25, acc=0.35)
            save_data(img_count, 'down')
            img_count += 1
            sleep(0.4)

    begin_right = [1.0110305889971907, -0.07687466675312797, 0.733018045582954]
    end_right = [1.0239468622641739, -0.1279101162682663, 0.7624208991043048]
    img_count = 0
    for y in np.linspace(begin_right[1],end_right[1],n):
        for z in np.linspace(begin_right[2],end_right[2],n):
            ori_right = [1.472706548414238, 2.6386450495354845, 0.2612600546986725]

            for i in range(3):
                ori_right[i] = ori_right[i] + np.random.uniform(-0.05, 0.05)
            pose_vec = [begin_right[0],y,z,ori_right[0],ori_right[1],ori_right[2]]
            robot.movel(pose_vec, vel=0.25, acc=0.35)
            save_data(img_count, 'right')
            img_count += 1
            sleep(0.4)


    begin_left = [1.0001611698011799, 0.015899107795441764, 0.720301902189386]
    end_left = [1.0400880076633088, 0.07439111408426344, 0.7662294638239011]
    img_count = 0
    for y in np.linspace(begin_left[1],end_left[1],n):
        for z in np.linspace(begin_left[2],end_left[2],n):
            ori_left = [2.5473499141863867, 1.6301700454327308, 0.10375688223816304]
            for i in range(3):
                ori_left[i] = ori_left[i] + np.random.uniform(-0.05, 0.05)
            pose_vec = [begin_left[0],y,z,ori_left[0],ori_left[1],ori_left[2]]
            robot.movel(pose_vec,  vel=0.25, acc=0.35)
            save_data(img_count, 'left')
            img_count+=1
            sleep(0.4)


def run_experiment():

    n = 10

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


moving_thread = Thread(target=run_experiment)
moving_thread.start()

if __name__ == "__main__":
    #run_experiment()
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Show images
            cv2.namedWindow('Demo', cv2.WINDOW_AUTOSIZE)
            #cv2.setMouseCallback('Demo', mouse_callback)
            #cv2.resizeWindow('Demo', 640, 480)
            cv2.imshow('Demo', color_image)
            cv2.waitKey(1)

    finally:
        # Stop streaming
        pipeline.stop()
