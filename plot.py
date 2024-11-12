import csv
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_results(csv_file):
    num_points = []
    psnr_mean = []
    psnr_std = []

    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            num_points.append(float(row['num_points']))
            psnr_mean.append(float(row['psnr_mean']))
            psnr_std.append(float(row['psnr_std']))

    num_points = np.array(num_points)
    psnr_mean = np.array(psnr_mean)
    psnr_std = np.array(psnr_std)

    num_points_hack = 1. * num_points
    num_points_hack[0] = .4

    plt.figure()
    plt.plot(num_points_hack, psnr_mean, 'o-', label=os.path.basename(csv_file))
    plt.fill_between(num_points_hack, psnr_mean - psnr_std, psnr_mean + psnr_std, alpha=0.2)
    plt.xscale('log')
    plt.xticks([.4, 1, 2, 5, 10, 20, 50, 100, 200, 500],
               ['Auto', '1', '2', '5', '10', '20', '50', '100', '200', '500'])
    plt.xlabel('Number of points')
    plt.ylabel('PSNR [db]')
    plt.legend(loc=0)
    plt.xlim((.4, 500))
    plt.title(f'Results for {os.path.basename(csv_file)}')
    plt.show()

def plot_all_csvs_in_folder(folder):
    for file_name in os.listdir(folder):
        if file_name.endswith('.csv'):
            plot_results(os.path.join(folder, file_name))

if __name__ == '__main__':
    folder = './checkpoints/siggraph_caffemodel'
    plot_all_csvs_in_folder(folder)