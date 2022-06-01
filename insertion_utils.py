import numpy as np, cv2
from  scipy.spatial.transform import Rotation as R
# utility functions



class plane():
  def __init__(self,p1,p2,p3):
    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    self.a, self.b, self.c = cp
    self.d = np.dot(cp, p3)

def triangulate(frame1_points, frame2_points, frame1_tcp0, frame2_tcp0, R_cam2gripper, t_cam2gripper, mtx):
    datas = [frame1_tcp0, frame2_tcp0]
    corn = []
    #print(frame1_points, frame2_points)
    #print(frame1_tcp0, frame2_tcp0)
    frame1_unflattened_points = np.reshape(np.array(frame1_points), (-1, int(len(frame1_points)/2), 2)).tolist()
    corn.append(frame1_unflattened_points)
    frame2_unflattened_points = np.reshape(np.array(frame2_points), (-1, int(len(frame2_points)/2), 2)).tolist()
    corn.append(frame2_unflattened_points)

    grip2base = []
    for dname in datas:
        pose_vector = dname

        ori = R.from_rotvec(np.array(pose_vector[3:]))
        ori_m = ori.as_matrix()

        mm_data = np.array(pose_vector[:3]) * 1000
        proxy = np.concatenate([ori_m, mm_data.reshape((3, 1))], axis=-1)
        grip2bas = np.concatenate([proxy, np.array([[0, 0, 0, 1]])], axis=0)

        grip2base.append(grip2bas)

    #print('R_cam2gripper', R_cam2gripper)
    #print('t_cam2gripper', t_cam2gripper)
    proxy = np.concatenate([R_cam2gripper, t_cam2gripper], axis=-1)
    #print('proxy', proxy)
    cam2gripper = np.concatenate([proxy, np.array([[0, 0, 0, 1]])], axis=0)
    #print('cam2gripper', cam2gripper)

    # evaluation of cameras projection matrices
    cam2base1 = np.matmul(grip2base[0], cam2gripper)
    #print('cam2base1',cam2base1)
    base2cam1 = np.linalg.inv(cam2base1)
    base2cam1 = np.delete(base2cam1, (3), axis=0)

    cam2base2 = np.matmul(grip2base[1], cam2gripper)
    base2cam2 = np.linalg.inv(cam2base2)
    base2cam2 = np.delete(base2cam2, (3), axis=0)
    # these proj matrices
    proj1 = np.matmul(mtx, base2cam1)
    proj2 = np.matmul(mtx, base2cam2)

    # method to get 3D coord of points column-wise [x,y,z,1]
    points = cv2.triangulatePoints(proj1, proj2, np.array(corn[0]), np.array(corn[1]))

    pointsT = points / points[3]
    pointsT = pointsT[:-1].T
    return (pointsT)

def triangulate_unflattened(frame1_points, frame2_points, frame1_tcp0, frame2_tcp0, R_cam2gripper, t_cam2gripper, mtx):
    datas = [frame1_tcp0, frame2_tcp0]
    corn = []

    corn.append(frame1_points)
    corn.append(frame2_points)

    grip2base = []
    for dname in datas:
        pose_vector = dname

        ori = R.from_rotvec(np.array(pose_vector[3:]))
        ori_m = ori.as_matrix()

        mm_data = np.array(pose_vector[:3]) * 1000
        proxy = np.concatenate([ori_m, mm_data.reshape((3, 1))], axis=-1)
        grip2bas = np.concatenate([proxy, np.array([[0, 0, 0, 1]])], axis=0)

        grip2base.append(grip2bas)

    proxy = np.concatenate([R_cam2gripper, t_cam2gripper], axis=-1)
    cam2gripper = np.concatenate([proxy, np.array([[0, 0, 0, 1]])], axis=0)

    # evaluation of cameras projection matrices
    cam2base1 = np.matmul(grip2base[0], cam2gripper)
    base2cam1 = np.linalg.inv(cam2base1)
    base2cam1 = np.delete(base2cam1, (3), axis=0)

    cam2base2 = np.matmul(grip2base[1], cam2gripper)
    base2cam2 = np.linalg.inv(cam2base2)
    base2cam2 = np.delete(base2cam2, (3), axis=0)
    # these proj matrices
    proj1 = np.matmul(mtx, base2cam1)
    proj2 = np.matmul(mtx, base2cam2)

    # method to get 3D coord of points column-wise [x,y,z,1]
    points = cv2.triangulatePoints(proj1, proj2, np.array(corn[0]), np.array(corn[1]))

    pointsT = points / points[3]
    pointsT = pointsT[:-1].T
    return (pointsT)