import csv
from options.train_options import TrainOptions
from models import create_model

import torch
import torchvision
import torchvision.transforms as transforms

from util import util
import numpy as np
import progressbar as pb
import shutil

import datetime as dt
import matplotlib.pyplot as plt

if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt.load_model = True
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.display_id = -1  # no visdom display
    opt.phase = 'test'
    opt.dataroot = './dataset/ilsvrc2012/%s/' % opt.phase
    opt.loadSize = 256
    opt.how_many = 1000 # not set by cli args?
    opt.aspect_ratio = 1.0
    opt.sample_Ps = [6, ]
    opt.load_model = True

    # number of random points to assign
    num_points = np.round(10**np.arange(-.1, 2.8, .1))
    num_points[0] = 0
    num_points = np.unique(num_points.astype('int'))
    N = len(num_points)

    dataset = torchvision.datasets.ImageFolder(opt.dataroot,
                                               transform=transforms.Compose([
                                                   transforms.Resize((opt.loadSize, opt.loadSize)),
                                                   transforms.ToTensor()]))
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    time = dt.datetime.now()
    str_now = '%02d_%02d_%02d%02d' % (time.month, time.day, time.hour, time.minute)

    shutil.copyfile('./checkpoints/%s/latest_net_G.pth' % opt.name, './checkpoints/%s/%s_net_G.pth' % (opt.name, str_now))

    psnrs = np.zeros((opt.how_many, N))

    bar = pb.ProgressBar(maxval=opt.how_many)
    bar.start()
    for i, data_raw in enumerate(dataset_loader):
        data_raw[0] = data_raw[0].cuda()
        data_raw[0] = util.crop_mult(data_raw[0], mult=8)

        for nn in range(N):
            data = util.get_colorization_data(data_raw, opt, ab_thresh=0., num_points=num_points[nn])

            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()

            psnrs[i, nn] = util.calculate_psnr_np(util.tensor2im(visuals['real']), util.tensor2im(visuals['fake_reg']))

        if i == opt.how_many - 1:
            break

        bar.update(i)

    # Save results
    psnrs_mean = np.mean(psnrs, axis=0)
    psnrs_std = np.std(psnrs, axis=0) / np.sqrt(opt.how_many)

    print(psnrs_mean)
    print("====")
    print(psnrs_std)

    np.save('./checkpoints/%s/psnrs_mean_%s' % (opt.name, str_now), psnrs_mean)
    np.save('./checkpoints/%s/psnrs_std_%s' % (opt.name, str_now), psnrs_std)
    np.save('./checkpoints/%s/psnrs_%s' % (opt.name, str_now), psnrs)
    print(', '.join(['%.2f' % psnr for psnr in psnrs_mean]))

    # Save results to CSV
    csv_file = './checkpoints/%s/results_%s.csv' % (opt.name, str_now)
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['num_points', 'psnr_mean', 'psnr_std'])
        for nn in range(N):
            writer.writerow([num_points[nn], psnrs_mean[nn], psnrs_std[nn]])

    num_points_hack = 1. * num_points
    num_points_hack[0] = .4

    plt.plot(num_points_hack, psnrs_mean, 'bo-', label=str_now)
    plt.fill_between(num_points_hack, psnrs_mean - psnrs_std, psnrs_mean + psnrs_std, color='blue', alpha=0.2)

    plt.xscale('log')
    plt.xticks([.4, 1, 2, 5, 10, 20, 50, 100, 200, 500],
               ['Auto', '1', '2', '5', '10', '20', '50', '100', '200', '500'])
    plt.xlabel('Number of points')
    plt.ylabel('CIEDE2000 score')
    plt.legend(loc=0)
    plt.xlim((num_points_hack[0], num_points_hack[-1]))
    plt.savefig('./checkpoints/%s/sweep_%s.png' % (opt.name, str_now))