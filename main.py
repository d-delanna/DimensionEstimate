import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
import time  # time.perf_counter()


class MNISTDimensionEstimate:
    def __init__(self, radius=0.1, num_centers=10, metric=1, max_radius_scalar=10, save=False):
        self._metric = metric
        self._radius = radius
        self._num_centers = num_centers
        self._max_radius_scalar = max_radius_scalar
        self._labels = [k for k in range(10)]
        self.sorted_data = self._read_in_data()

        self.sorted_dimension_estimates = []
        self.sorted_dim_estimates_std = []
        for label in self._labels:
            data = self.sorted_data[label]
            dimension_estimates = self._estimate_dimension(data)
            self.sorted_dimension_estimates.append(np.mean(dimension_estimates, axis=0))
            self.sorted_dim_estimates_std.append(np.std(dimension_estimates, axis=0))
            print(f'\nLABEL {label} AVG: {self.sorted_dimension_estimates[label]}')
            print(f'LABEL {label} STD DEV: {self.sorted_dim_estimates_std[label]}')

        if save:
            self._plot_estimates()

    def _read_in_data(self):
        mnist_data = pd.read_csv('train.csv')
        sorted_data = [mnist_data.loc[mnist_data['label'] == label].drop(columns='label') for label in self._labels]
        return sorted_data

    def _estimate_dimension(self, data):
        dimension_estimates = np.empty(shape=(self._num_centers, self._max_radius_scalar))  # (centers, radii)

        for center_idx in range(self._num_centers):
            start = time.perf_counter()
            distance_df = self._compute_distance_matrix(center_idx, data)
            dimension_estimates[center_idx] = self._apply_density_formula(distance_df)
            end = time.perf_counter()
            print(end - start)
        return dimension_estimates

    def _compute_distance_matrix(self, center_idx, data):
        center = data.iloc[center_idx]  # jth row in X
        factor = 28**(2/self._metric) * 255
        return data.apply(lambda x: np.linalg.norm((x - center), ord=self._metric), axis=1) / factor

    def _apply_density_formula(self, distance_df):
        """ In theory, dimension = log_2 ( |B_2r(center)| / |B_r(center)| ). """
        ratios = np.empty(shape=self._max_radius_scalar)

        for radius_scalar in range(1, self._max_radius_scalar + 1):
            size_ball2 = (distance_df <= 2 * self._radius * radius_scalar).sum()
            size_ball1 = (distance_df <= self._radius * radius_scalar).sum()

            ratios[radius_scalar - 1] = math.log2(size_ball2 / size_ball1)

        return ratios

    def _plot_estimates(self):
        x_axis = [self._radius * radius_scalar for radius_scalar in range(1, self._max_radius_scalar + 1)]

        # fig, ax = plt.subplots()
        for label in self._labels:
            #ax.plot(x_axis, self.sorted_dimension_estimates[label], label=label)
            plt.errorbar(x_axis, self.sorted_dimension_estimates[label], yerr=self.sorted_dim_estimates_std[label], label=label)
        plt.xlabel('Radius')
        plt.ylabel('Dimension Estimate')
        plt.legend()
        plt.savefig(f'plots/(r,n,p)=({self._radius},{self._num_centers},{self._metric}).jpg')
        plt.clf()

    def plot_examples(self):
        plt.figure(figsize=(10, 5))
        for center_idx in range(2):
            for label in self._labels:
                plt.subplot(center_idx + 1, len(self._labels), label + 1)
                img = np.array(self.sorted_data[label])[center_idx].reshape(28, 28)
                plt.imshow(img, cmap='gray')
                plt.title(f"Label: {label}")
                plt.axis('off')
        plt.tight_layout()
        plt.savefig('examples.jpg')
        plt.clf()


if __name__ == '__main__':
    for num_centers in [1, 10, 25, 100]:
        MNISTDimensionEstimate(radius=0.05, num_centers=num_centers, metric=2, save=True)
