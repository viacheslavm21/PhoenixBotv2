import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import data.experiments.insertion.input_points as pts

results = []
for i in range(4):
    results.append(pd.read_csv(f'data/experiments/insertion/pose{i}_experiment.csv'))
true_poses = pts.true_poses

SHOW_POSE = 2

true_data_x = []
true_data_y = []
true_data_z = []

for poseid in range(4):
    true_data_x.append(np.ones(len(results[poseid])) * np.mean(results[poseid]['pose_x']))
    true_data_y.append(np.ones(len(results[poseid])) * np.mean(results[poseid]['pose_y']))
    true_data_z.append(np.ones(len(results[poseid])) * np.mean(results[poseid]['pose_z']))
    error_x = np.asarray(true_data_x[-1]) - results[poseid]['pose_x']
    error_y = np.asarray(true_data_y[-1]) - results[poseid]['pose_y']
    error_z = np.asarray(true_data_z[-1]) - results[poseid]['pose_z']

    mse_x = mean_squared_error(true_data_x[-1], results[poseid]['pose_x'])
    mse_y = mean_squared_error(true_data_y[-1], results[poseid]['pose_y'])
    mse_z = mean_squared_error(true_data_z[-1], results[poseid]['pose_z'])

    print(mse_x, mse_y, mse_z)
    print(np.sqrt(mse_x), np.sqrt(mse_y), np.sqrt(mse_z))
    print(np.std(error_x),np.std(error_y),np.std(error_z))

    n, bins, patches = plt.hist(error_x, 25, density=True, facecolor='g', alpha=0.75)
    plt.xlabel('Error')
    plt.ylabel('Probability')
    plt.title('error_in_pose_x')
    plt.grid(True)
    plt.show()

    n, bins, patches = plt.hist(error_y, 25, density=True, facecolor='g', alpha=0.75)
    plt.xlabel('Error')
    plt.ylabel('Probability')
    plt.title('error_in_pose_y')
    plt.grid(True)
    plt.show()

    n, bins, patches = plt.hist(error_z, 25, density=True, facecolor='g', alpha=0.75)
    plt.xlabel('Error')
    plt.ylabel('Probability')
    plt.title('error_in_pose_z')
    plt.grid(True)
    plt.show()

input()
print('mean error_in_pose_x', np.mean(results[SHOW_POSE]['error_in_pose_x']))
print('std error_in_pose_x', np.std(results[SHOW_POSE]['error_in_pose_x']))

n, bins, patches = plt.hist(results[SHOW_POSE]['error_in_pose_x'], 25, density=True, facecolor='g', alpha=0.75)
plt.xlabel('Error')
plt.ylabel('Probability')
plt.title('error_in_pose_x')
plt.grid(True)
plt.show()

print('mean error_in_pose_y', np.mean(results[SHOW_POSE]['error_in_pose_y']))
print('std error_in_pose_y', np.std(results[SHOW_POSE]['error_in_pose_y']))

n, bins, patches = plt.hist(results[SHOW_POSE]['error_in_pose_y'], 25, density=True, facecolor='g', alpha=0.75)
plt.xlabel('Error')
plt.ylabel('Probability')
plt.title('error_in_pose_y')
plt.grid(True)
plt.show()

print('mean error_in_pose_z', np.mean(results[SHOW_POSE]['error_in_pose_z']))
print('std error_in_pose_z', np.std(results[SHOW_POSE]['error_in_pose_z']))

n, bins, patches = plt.hist(results[SHOW_POSE]['error_in_pose_z'], 25, density=True, facecolor='g', alpha=0.75)
plt.xlabel('Error')
plt.ylabel('Probability')
plt.title('error_in_pose_z')
plt.grid(True)
plt.show()



n, bins, patches = plt.hist(results[SHOW_POSE]['error_in_pose_yaw'], 25, density=True, facecolor='g', alpha=0.75)
plt.xlabel('Error')
plt.ylabel('Probability')
plt.title('error_in_pose_yaw')
plt.grid(True)
plt.show()

n, bins, patches = plt.hist(results[SHOW_POSE]['error_in_pose_pitch'], 25, density=True, facecolor='g', alpha=0.75)
plt.xlabel('Error')
plt.ylabel('Probability')
plt.title('error_in_pose_pitch')
plt.grid(True)
plt.show()

n, bins, patches = plt.hist(results[SHOW_POSE]['error_in_pose_roll'], 25, density=True, facecolor='g', alpha=0.75)
plt.xlabel('Error')
plt.ylabel('Probability')
plt.title('error_in_pose_roll')
plt.grid(True)
plt.show()

plt.plot(results[SHOW_POSE]['error_in_pose_x'])
plt.ylabel('error_in_pose_x')
plt.show()

plt.plot(results[SHOW_POSE]['error_in_pose_y'])
plt.ylabel('error_in_pose_y')
plt.show()

plt.plot(results[SHOW_POSE]['error_in_pose_z'])
plt.ylabel('error_in_pose_z')
plt.show()