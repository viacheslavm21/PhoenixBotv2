# bundle adjustment functions for coordinate refinement

from __future__ import print_function

import time
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import cv2
import numpy as np
from typing import List
from scipy.spatial.transform import Rotation as R

# I need: 3d points in base.
# for each frame: 3d image points, rvec, tvec
from calib_utils import CalibrationModel

def draw_reprojected(img, points):
    print('points')
    print(img)
    image = cv2.imread(img)
    for kps in points:

        for idx, kp in enumerate(kps):
            print(kp)
            image = cv2.circle(image.copy(),(int(kp[0]),int(kp[1])), 1, (210, 32, 13), 1)

    cv2.namedWindow('Reprojection', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Reprojection', image)
    k = cv2.waitKey(0) & 0xFF

def ba_loss_draw(params, mtx, dist, images_pixels, images):
    imgpoints_arr = []
    #print(images_pixels)
    n_keypoints = len(images_pixels[0])
    #print("n_keypoints", n_keypoints)
    n_keypoints = int(n_keypoints)
    amount_of_points_coords = n_keypoints*3
    amount_of_rvecs_params = amount_of_points_coords + len(images_pixels) * 3
    # globally defined: amount_of_points_coords,amount_of_rvecs_params
    #print('params', params)
    points_3d = params[:amount_of_points_coords].reshape(-1,3)
    r_vecs = params[amount_of_points_coords:amount_of_rvecs_params].reshape(-1,3)
    t_vecs = params[amount_of_rvecs_params:].reshape(-1,3)
    for i in range(r_vecs.shape[0]):
        #print('points_3d', points_3d)
        #print('r_vecs[i]', r_vecs[i])
        #print('t_vecs[i]', t_vecs[i])
        #print('mtx',mtx)
        #print('dist', dist)
        imgpoints2, _ = cv2.projectPoints(points_3d, r_vecs[i], t_vecs[i], mtx, dist)
        #for e,_ in enumerate(images_pixels[i]):
        #    images_pixels[i][e] = [images_pixels[i][e]]

        #print('images_pixels[i]', images_pixels[i])
        #print('imgpoints2', imgpoints2)
        #draw_reprojected(images[i], images_pixels[i])
        draw_reprojected(images[i], imgpoints2)
        #print('imgpoints2', imgpoints2)
        imgpoints_arr.append(imgpoints2)
    return (np.array(imgpoints_arr).reshape((-1,n_keypoints,2)) - images_pixels).ravel()


def ba_loss(params, mtx, dist, images_pixels):#, images):
    imgpoints_arr = []
    #print(images_pixels)
    n_keypoints = len(images_pixels[0])
    #print("n_keypoints", n_keypoints)
    n_keypoints = int(n_keypoints)
    amount_of_points_coords = n_keypoints*3
    amount_of_rvecs_params = amount_of_points_coords + len(images_pixels) * 3
    # globally defined: amount_of_points_coords,amount_of_rvecs_params
    #print('params', params)
    points_3d = params[:amount_of_points_coords].reshape(-1,3)
    r_vecs = params[amount_of_points_coords:amount_of_rvecs_params].reshape(-1,3)
    t_vecs = params[amount_of_rvecs_params:].reshape(-1,3)
    for i in range(r_vecs.shape[0]):
        #print('points_3d', points_3d)
        #print('r_vecs[i]', r_vecs[i])
        #print('t_vecs[i]', t_vecs[i])
        #print('mtx',mtx)
        #print('dist', dist)
        imgpoints2, _ = cv2.projectPoints(points_3d, r_vecs[i], t_vecs[i], mtx, dist)
        #draw_reprojected(images[i], imgpoints2)
        #print('imgpoints2', imgpoints2)
        imgpoints_arr.append(imgpoints2)

    return (np.array(imgpoints_arr).reshape((-1,n_keypoints,2)) - images_pixels).ravel()

def bundle_optimize_view(objpoints_in_robot: List, imgs_points: List[List], img_tcp0s: List[List], calib_model: CalibrationModel, images: List):
    ### PREPARE DATA
    proxy = np.concatenate([calib_model.R_cam2gripper, calib_model.t_cam2gripper], axis=-1)

    cam2gripper = np.concatenate([proxy, np.array([[0, 0, 0, 1]])], axis=0)

    print('cam2gripper', cam2gripper)

    grip2base_list = []
    base2cam_list = []
    rvecs_robot_in_camera = []
    tvecs_robot_in_camera = []

    for img_tcp0 in img_tcp0s:
        print('img_tcp0', img_tcp0)
        ori = R.from_rotvec(np.array(img_tcp0[3:]))
        #print('ori', ori.as_rotvec())
        ori_m = ori.as_matrix()

        mm_data = np.array(img_tcp0[:3]) * 1000
        #print('mm_data',mm_data)
        proxy = np.concatenate([ori_m, mm_data.reshape((3, 1))], axis=-1)
        grip2base = np.concatenate([proxy, np.array([[0, 0, 0, 1]])], axis=0)

        #print('grip2base', grip2base)

        grip2base_list.append(grip2base)

        # evaluation of cameras projection matrices
        cam2base = np.matmul(grip2base, cam2gripper)
        #print('cam2base', cam2base)
        #print('t_cam2base', cam2base[:3, 3])
        base2cam = np.linalg.inv(cam2base)
        base2cam_list.append(base2cam)
        rvec = R.from_matrix(base2cam[:3, :3]).as_rotvec()
        tvec = base2cam[:3, 3]/1000
        #print(rvec, tvec)
        rvecs_robot_in_camera.append(rvec)
        tvecs_robot_in_camera.append(tvec)

    objpoints_in_robot = np.asarray(objpoints_in_robot)
    tvecs_robot_in_camera = np.asarray(tvecs_robot_in_camera)
    rvecs_robot_in_camera = np.asarray(rvecs_robot_in_camera)
    print('objpoints_in_robot', objpoints_in_robot)
    #print('len objpoints_in_robot', len(objpoints_in_robot))
    print('tvecs_robot_in_camera', tvecs_robot_in_camera)
    print('rvecs_robot_in_camera', rvecs_robot_in_camera)
    # nado: objpoints_in_robot, images_pixels, rvecs_robot_in_camera, tvecs_robot_in_camera

    # loss calculation dry run
    x0 = np.hstack((objpoints_in_robot.ravel(), rvecs_robot_in_camera.ravel(), tvecs_robot_in_camera.ravel()))
    # np.linalg.norm(fun(x0,  calibr_models[0].intrs, calibr_models[0].dist,images_pixels_refined))
    f0 = ba_loss_draw(x0, calib_model.intrs, calib_model.dist, imgs_points, images)
    plt.plot(f0)
    plt.show()
    #print("x0 looks like", x0)
    t0 = time.time()
    res = least_squares(ba_loss, x0, verbose=2, x_scale='jac', ftol=1e-4, method='lm', max_nfev=10000,
                        args=(calib_model.intrs, calib_model.dist, imgs_points))
    t1 = time.time()
    print("Optimization took {0:.0f} seconds".format(t1 - t0))

    n_keypoints = len(imgs_points[0])
    n_keypoints = int(n_keypoints)
    amount_of_points_coords = n_keypoints * 3

    #print(f"res:{res.x[:amount_of_points_coords].reshape(-1,3)}")
    plt.plot(res.fun)
    plt.show()
    #print("res.x, looks like", res.x)

    f_res = ba_loss_draw(res.x, calib_model.intrs, calib_model.dist, imgs_points, images)

    return res.x[:amount_of_points_coords].reshape(-1, 3)

def bundle_optimize(objpoints_in_robot: List, imgs_points: List[List], img_tcp0s: List[List], calib_model: CalibrationModel, images: List):
    ### PREPARE DATA
    proxy = np.concatenate([calib_model.R_cam2gripper, calib_model.t_cam2gripper], axis=-1)
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
        rvec = R.from_matrix(base2cam[:3,:3]).as_rotvec()
        tvec = base2cam[:3, 3]/1000
        rvecs_robot_in_camera.append(rvec)
        tvecs_robot_in_camera.append(tvec)

    objpoints_in_robot = np.asarray(objpoints_in_robot)
    tvecs_robot_in_camera = np.asarray(tvecs_robot_in_camera)
    rvecs_robot_in_camera = np.asarray(rvecs_robot_in_camera)

    # loss calculation dry run
    x0 = np.hstack((objpoints_in_robot.ravel(), rvecs_robot_in_camera.ravel(), tvecs_robot_in_camera.ravel()))
    t0 = time.time()
    res = least_squares(ba_loss, x0, verbose=0, x_scale='jac', ftol=1e-4, method='lm', max_nfev=10000,
                        args=(calib_model.intrs, calib_model.dist, imgs_points))
    t1 = time.time()
    #print("Optimization took {0:.0f} seconds".format(t1 - t0))

    n_keypoints = len(imgs_points[0])
    n_keypoints = int(n_keypoints)
    amount_of_points_coords = n_keypoints * 3

    #print(f"res:{res.x[:amount_of_points_coords].reshape(-1,3)}")

    return res.x[:amount_of_points_coords].reshape(-1, 3)