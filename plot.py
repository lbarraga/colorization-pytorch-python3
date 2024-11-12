import numpy as np
import matplotlib.pyplot as plt
import os

# Directory containing the .npy files
checkpoints_dir = './checkpoints/siggraph_caffemodel/'

# Load the mean and standard deviation .npy files
psnrs_mean_file = os.path.join(checkpoints_dir, 'psnrs_mean_11_12_1522.npy')
psnrs_std_file = os.path.join(checkpoints_dir, 'psnrs_std_11_12_1522.npy')

psnrs_mean = np.load(psnrs_mean_file)
psnrs_std = np.load(psnrs_std_file)

# Number of random points to assign
num_points = np.round(10**np.arange(-.1, 2.8, .1))
num_points[0] = 0
num_points = np.unique(num_points.astype('int'))
num_points_hack = 1. * num_points
num_points_hack[0] = .4

# Plot the mean PSNR with shaded area for the standard deviation
plt.plot(num_points_hack, psnrs_mean, 'bo-', label='Current')
plt.fill_between(num_points_hack, psnrs_mean - psnrs_std, psnrs_mean + psnrs_std, color='blue', alpha=0.2)

# Customize the plot
plt.xscale('log')
plt.xticks([.4, 1, 2, 5, 10, 20, 50, 100, 200, 500],
           ['Auto', '1', '2', '5', '10', '20', '50', '100', '200', '500'])
plt.xlabel('Number of points')
plt.ylabel('PSNR [dB]')
plt.legend(loc=0)
plt.xlim((num_points_hack[0], num_points_hack[-1]))
plt.title('PSNR with Standard Deviation')
plt.show()