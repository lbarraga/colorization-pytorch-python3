import csv
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_results(csv_file, ax):
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

    ax.plot(num_points_hack, psnr_mean, 'o-', label=os.path.basename(csv_file))
    ax.fill_between(num_points_hack, psnr_mean - psnr_std, psnr_mean + psnr_std, alpha=0.2)

def plot_all_csvs_in_folder(folder):
    fig, ax = plt.subplots()
    for file_name in os.listdir(folder):
        if file_name.endswith('.csv'):
            plot_results(os.path.join(folder, file_name), ax)

    ax.set_xscale('log')
    ax.set_xticks([.4, 1, 2, 5, 10])
    ax.set_xticklabels(['Auto', '1', '2', '5', '10'])
    ax.set_xlabel('Number of points')
    ax.set_ylabel('PSNR [db]')
    ax.legend(loc=0)
    ax.set_xlim((.4, 500))
    ax.set_title('Results for all CSV files')
    plt.show()

if __name__ == '__main__':
    folder = './checkpoints/siggraph_caffemodel'
    plot_all_csvs_in_folder(folder)