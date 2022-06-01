import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

def calibrate_hand_eye_initial_pnp():
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


    print("intrinsic matrix from images")
    print(mtx)

    print("camera in gripper coordinates:")
    print(t_cam2gripper)

    print("rotation vector:")
    print(R.from_matrix(R_cam2gripper).as_rotvec())

    return t_cam2gripper, R.from_matrix(R_cam2gripper).as_rotvec()

def compute_re_projection_loss(hand_eye):
    pass


def optimize_genetic(initial_hand_eye):
    pass
