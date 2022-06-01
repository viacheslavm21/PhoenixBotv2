from typing import List

from fitter import Fitter
import urx, json, numpy as np, pickle
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# To find the best data distribution:
#https://medium.com/the-researchers-guide/finding-the-best-distribution-that-fits-your-data-using-pythons-fitter-library-319a5a0972e9

def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions), axis=0)

def get_error(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return y_true - predictions

def get_abs_error(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.abs(y_true - predictions)

class AnalyzeInsertion:
    def __init__(self, datafilenames: List):
        self.poses = []
        for datafilename in datafilenames:
            with open(datafilename, 'rb') as f:
                poses = np.asarray(pickle.load(f))
                if self.poses != []:   # do not follow the recommendation
                    self.poses = np.concatenate((self.poses, poses))
                else:
                    self.poses = np.asarray(poses)

        self.X = self.poses[:, 0]
        self.Y = self.poses[:, 1]
        self.Z = self.poses[:, 2]
        self.RX = self.poses[:, 3]
        self.RY = self.poses[:, 4]
        self.RZ = self.poses[:, 5]

        self.gt = np.mean(np.asarray(self.poses), axis=0)  # ground truth

        self.errors = get_error(self.gt, self.poses)

        self.e_X = self.errors[:, 0]
        self.e_Y = self.errors[:, 1]
        self.e_Z = self.errors[:, 2]
        self.e_RX = self.errors[:, 3]
        self.e_RY = self.errors[:, 4]
        self.e_RZ = self.errors[:, 5]

        self.font = {'family': 'serif',
                    'color': 'darkred',
                    'weight': 'normal',
                    'size': 16,
                    }

    def print_stats(self):
        print("Lenght:", len(self.poses))
        print("MEAN")
        print(np.mean(np.asarray(self.poses), axis=0)[:3]*1000, np.mean(np.asarray(self.poses), axis=0)[3:]*57.3)
        print("STD")
        print(np.std(np.asarray(self.poses), axis=0)[:3]*1000, np.std(np.asarray(self.poses), axis=0)[3:]*57.3)
        print("Mean Absolute Error")
        print(mae(np.mean(np.asarray(self.poses), axis=0), self.poses)[:3]*1000, mae(np.mean(np.asarray(self.poses), axis=0), self.poses)[3:]*57.3)

    def print_data(self):
        print(self.poses)

    def plot_one_hist(self, data, title):
        plt.rcParams["figure.figsize"] = (10, 7)
        n, bins, patches = plt.hist(data, 25, density=False, facecolor='#4178A4', edgecolor='black', alpha=0.95)

        plt.xlabel(title, fontsize=28)
        plt.ylabel('Number of repeats', fontsize=28)
        plt.xticks(fontsize=28, rotation=0)
        plt.yticks(fontsize=28, rotation=0)
        #plt.title(f"Historgam of {title}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"random/error_hitograms/E_3_{title}.png")
        plt.show()

    def plot_one_hist_density(self, data, title):
        sigma = np.std(data)
        mu = np.mean(data)
        plt.rcParams["figure.figsize"] = (10, 7)
        n, bins, patches = plt.hist(data, 25, density=True, facecolor='#4178A4', edgecolor='black', alpha=0.75)
        plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *

                 np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),

                 linewidth=2, color='r')
        plt.xlabel(title, fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.xticks(fontsize=16, rotation=0)
        plt.yticks(fontsize=16, rotation=0)
        #plt.title(f"Historgam of {title}")
        plt.grid(True)
        plt.show()

    def fit(self, data):
        f = Fitter(data)
        """,
                   distributions=['gamma',
                                  'lognorm',
                                  "beta",
                                  "burr",
                                  "norm"])"""
        f.fit()
        print(f.summary())



class AnalyzeCorrelations(AnalyzeInsertion):
    def __init__(self, datafilenames: List, reprojectionFilenames: List, dotProductFilenames: List):
        super(AnalyzeCorrelations, self).__init__(datafilenames)
        self.re_errors = []
        for reprojectionFilename in reprojectionFilenames:
            with open(reprojectionFilename, 'rb') as f:
                re_errors = np.asarray(pickle.load(f))
                if self.re_errors != []:   # do not follow the recommendation
                    self.re_errors = np.concatenate((self.re_errors, re_errors))
                else:
                    self.re_errors = np.asarray(re_errors)

        self.dot_products = []
        for dotProductFilename in dotProductFilenames:
            with open(dotProductFilename, 'rb') as f:
                dot_products = np.asarray(pickle.load(f))
                if self.dot_products != []:   # do not follow the recommendation
                    self.dot_products = np.concatenate((self.dot_products, dot_products))
                else:
                    self.dot_products = np.asarray(dot_products)

        #self._sort_by_error_value()

    def _sort_by_error_value(self):
        print(np.sort(self.re_errors))

    def plot_scatter_re_projection_vs_positioning(self):

        for i in range(6):
            data2 = list(map(abs, self.re_errors))
            data1 = list(map(abs,self.errors[:,i]))
            covariance = np.cov(data1, data2)
            print(f"Covariance between re-projection error and {i + 1}th positioning DoF")
            print(covariance)
            corr, _ = pearsonr(data1, data2)
            print(f'Pearsons correlation between re-projection error and {i + 1}th positioning DoF: %.3f' % corr)
            # plot
            plt.scatter(data1, data2)
            plt.title(f"Correlation between re-projection error and {i+1}th positioning DoF")
            plt.savefig(f'data/correlations/Correlation between re-projection error and {i+1}th positioning DoF')
            plt.show()

    def plot_scatter_dot_products_vs_positioning(self):
        for n, dot_name in enumerate(['X and Y', 'X and Z', 'Z and Y']):
            for i in range(6):
                data1 = list(map(abs, self.errors[:, i]))
                data2 = list(map(abs, self.dot_products[:,n]))

                covariance = np.cov(data1, data2)
                print(f"Covariance between dot product {dot_name} and {i + 1}th positioning DoF")
                print(covariance)
                corr, _ = pearsonr(data1, data2)
                print(f'Pearsons correlation between dot product {dot_name} and {i + 1}th positioning DoF: %.3f' % corr)
                # plot
                plt.scatter(data1, data2)
                plt.title(f"Correlation between dot product {dot_name} and {i + 1}th positioning DoF")
                plt.show()


insertion_poses = AnalyzeInsertion(["data/big_experiment/config_13May_env_2/precise_poses_v1.2.pickle"])
#insertion_poses.fit(insertion_poses.X)

reprojection_correlation = AnalyzeCorrelations(["data/big_experiment/config_16May_test_env_1/precise_poses_v1.0.pickle"], ["data/big_experiment/config_16May_test_env_1/reprojection_errors_v1.0.pickle"],['data/big_experiment/config_16May_test_env_1/dot_poructs_errors_v1.0.pickle'])
reprojection_correlation.plot_scatter_re_projection_vs_positioning()
reprojection_correlation.plot_scatter_dot_products_vs_positioning()


#insertion_poses.print_data()
"""insertion_poses.print_stats()
insertion_poses.plot_one_hist(insertion_poses.e_X*1000, "Linear error along Z, mm")  # (!) Z
insertion_poses.plot_one_hist(insertion_poses.e_Y*1000, "Linear error along Y, mm")
insertion_poses.plot_one_hist(insertion_poses.e_Z*1000, "Linear error along X, mm")
insertion_poses.plot_one_hist(insertion_poses.e_RX*57.23, "Angular error around Z, deg")
insertion_poses.plot_one_hist(insertion_poses.e_RY*57.23, "Angular error around Y, deg")
insertion_poses.plot_one_hist(insertion_poses.e_RZ*57.23, "Angular error around X, deg")"""

# "data/big_experiment/config_13May_env_2/precise_poses_v1.1.pickle" - 374 samples
# "data/big_experiment/config_14May_tast_env_1/precise_poses_v1.1.pickle" - 65 samples
# "data/big_experiment/config_13May_env_2/precise_poses_v1.2.pickle" - 56 samples

