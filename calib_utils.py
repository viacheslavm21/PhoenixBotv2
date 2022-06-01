
import cv2, numpy as np,pickle

from scipy.spatial.transform import Rotation as R


class CalibrationModel:
    def __init__(self, intrinsic, distortion, name):
        self.intrs = np.asarray(intrinsic)
        self.dist = np.asarray(distortion)
        self.name = name
        self.t_cam2gripper = None
        self.R_cam2gripper = None

    def fill_extrinsic(self, R_cam2gripper, t_cam2gripper):
        """
        :param R_cam2gripper: rotvec 1x3 rad
        :param t_cam2gripper:  translation vector 1x3 mm
        :return: None
        """
        self.t_cam2gripper = [[t_cam2gripper[0]], [t_cam2gripper[1]], [t_cam2gripper[2]]]
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
                error = cv2.norm(corners2, imgpoints_proj, cv2.NORM_L2) / len(corners2)
                mean_re_error += error

        return mean_re_error


# download objects
def download_and_fix_calib_objs():
    with open("data/calib_models_ro", "rb") as f:
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
    return calibs, valid_calibs
