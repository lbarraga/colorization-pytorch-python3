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

    ax.plot(num_points_hack, psnr_mean, 'o-', label=os.path.basename(csv_file[:-4]))
    ax.fill_between(num_points_hack, psnr_mean - psnr_std, psnr_mean + psnr_std, alpha=0.2)

def plot_all_csvs_in_folder(folder):
    fig, ax = plt.subplots()
    for file_name in os.listdir(folder):
        if file_name.endswith('.csv'):
            plot_results(os.path.join(folder, file_name), ax)

    ax.set_xscale('log')
    plt.xticks([.4, 1, 2, 5, 10, 20, 50, 100, 200, 500],
               ['Auto', '1', '2', '5', '10', '20', '50', '100', '200', '500'])
    ax.set_xlabel('Number Color Hints')
    ax.set_ylabel('CIEDE2000 Ratio [0-1]')
    ax.legend(loc=0)
    ax.set_xlim((.4, 500))
    ax.set_title('')
    plt.show()

if __name__ == '__main__':
    folder = './checkpoints/siggraph_caffemodel'
    plot_all_csvs_in_folder(folder)