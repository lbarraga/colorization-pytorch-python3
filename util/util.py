from __future__ import print_function

import math
import os
from collections import OrderedDict
from random import shuffle
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans


# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.clip((np.transpose(image_numpy, (1, 2, 0))), 0, 1) * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_subset_dict(in_dict, keys):
    if (len(keys)):
        subset = OrderedDict()
        for key in keys:
            subset[key] = in_dict[key]
    else:
        subset = in_dict
    return subset


# Color conversion code
def rgb2xyz(rgb):  # rgb from [0,1]
    # xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
    # [0.212671, 0.715160, 0.072169],
    # [0.019334, 0.119193, 0.950227]])

    mask = (rgb > .04045).type(torch.FloatTensor)
    if (rgb.is_cuda):
        mask = mask.cuda()

    rgb = (((rgb + .055) / 1.055) ** 2.4) * mask + rgb / 12.92 * (1 - mask)

    x = .412453 * rgb[:, 0, :, :] + .357580 * rgb[:, 1, :, :] + .180423 * rgb[:, 2, :, :]
    y = .212671 * rgb[:, 0, :, :] + .715160 * rgb[:, 1, :, :] + .072169 * rgb[:, 2, :, :]
    z = .019334 * rgb[:, 0, :, :] + .119193 * rgb[:, 1, :, :] + .950227 * rgb[:, 2, :, :]
    out = torch.cat((x[:, None, :, :], y[:, None, :, :], z[:, None, :, :]), dim=1)

    # if(torch.sum(torch.isnan(out))>0):
    # print('rgb2xyz')
    # embed()
    return out


def xyz2rgb(xyz):
    # array([[ 3.24048134, -1.53715152, -0.49853633],
    #        [-0.96925495,  1.87599   ,  0.04155593],
    #        [ 0.05564664, -0.20404134,  1.05731107]])

    r = 3.24048134 * xyz[:, 0, :, :] - 1.53715152 * xyz[:, 1, :, :] - 0.49853633 * xyz[:, 2, :, :]
    g = -0.96925495 * xyz[:, 0, :, :] + 1.87599 * xyz[:, 1, :, :] + .04155593 * xyz[:, 2, :, :]
    b = .05564664 * xyz[:, 0, :, :] - .20404134 * xyz[:, 1, :, :] + 1.05731107 * xyz[:, 2, :, :]

    rgb = torch.cat((r[:, None, :, :], g[:, None, :, :], b[:, None, :, :]), dim=1)
    rgb = torch.max(rgb, torch.zeros_like(rgb))  # sometimes reaches a small negative number, which causes NaNs

    mask = (rgb > .0031308).type(torch.FloatTensor)
    if (rgb.is_cuda):
        mask = mask.cuda()

    rgb = (1.055 * (rgb ** (1. / 2.4)) - 0.055) * mask + 12.92 * rgb * (1 - mask)

    # if(torch.sum(torch.isnan(rgb))>0):
    # print('xyz2rgb')
    # embed()
    return rgb


def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
    if (xyz.is_cuda):
        sc = sc.cuda()

    xyz_scale = xyz / sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if (xyz_scale.is_cuda):
        mask = mask.cuda()

    xyz_int = xyz_scale ** (1 / 3.) * mask + (7.787 * xyz_scale + 16. / 116.) * (1 - mask)

    L = 116. * xyz_int[:, 1, :, :] - 16.
    a = 500. * (xyz_int[:, 0, :, :] - xyz_int[:, 1, :, :])
    b = 200. * (xyz_int[:, 1, :, :] - xyz_int[:, 2, :, :])
    out = torch.cat((L[:, None, :, :], a[:, None, :, :], b[:, None, :, :]), dim=1)

    # if(torch.sum(torch.isnan(out))>0):
    # print('xyz2lab')
    # embed()

    return out


def lab2xyz(lab):
    y_int = (lab[:, 0, :, :] + 16.) / 116.
    x_int = (lab[:, 1, :, :] / 500.) + y_int
    z_int = y_int - (lab[:, 2, :, :] / 200.)
    if (z_int.is_cuda):
        z_int = torch.max(torch.Tensor((0,)).cuda(), z_int)
    else:
        z_int = torch.max(torch.Tensor((0,)), z_int)

    out = torch.cat((x_int[:, None, :, :], y_int[:, None, :, :], z_int[:, None, :, :]), dim=1)
    mask = (out > .2068966).type(torch.FloatTensor)
    if (out.is_cuda):
        mask = mask.cuda()

    out = (out ** 3.) * mask + (out - 16. / 116.) / 7.787 * (1 - mask)

    sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
    sc = sc.to(out.device)

    out = out * sc

    # if(torch.sum(torch.isnan(out))>0):
    # print('lab2xyz')
    # embed()

    return out


def rgb2lab(rgb, opt):
    lab = xyz2lab(rgb2xyz(rgb))
    l_rs = (lab[:, [0], :, :] - opt.l_cent) / opt.l_norm
    ab_rs = lab[:, 1:, :, :] / opt.ab_norm
    out = torch.cat((l_rs, ab_rs), dim=1)
    # if(torch.sum(torch.isnan(out))>0):
    # print('rgb2lab')
    # embed()
    return out


def lab2rgb(lab_rs, opt):
    l = lab_rs[:, [0], :, :] * opt.l_norm + opt.l_cent
    ab = lab_rs[:, 1:, :, :] * opt.ab_norm
    lab = torch.cat((l, ab), dim=1)
    out = xyz2rgb(lab2xyz(lab))
    # if(torch.sum(torch.isnan(out))>0):
    # print('lab2rgb')
    # embed()
    return out


def get_colorization_data(data_raw, opt, ab_thresh=5., p=.125, num_points=None):
    data = {}

    data_lab = rgb2lab(data_raw[0], opt)
    data['A'] = data_lab[:, [0, ], :, :]
    data['B'] = data_lab[:, 1:, :, :]

    if ab_thresh > 0:  # mask out grayscale images
        thresh = 1. * ab_thresh / opt.ab_norm
        mask = torch.sum(torch.abs(
            torch.max(torch.max(data['B'], dim=3)[0], dim=2)[0] - torch.min(torch.min(data['B'], dim=3)[0], dim=2)[0]),
            dim=1) >= thresh
        data['A'] = data['A'][mask, :, :, :]
        data['B'] = data['B'][mask, :, :, :]
        # print('Removed %i points'%torch.sum(mask==0).numpy())
        if (torch.sum(mask) == 0):
            return None

    return add_color_patches(data, opt, p=p, num_points=num_points)


def add_color_patches(data, opt, p=0.125, num_points=None, samp='?'):
    N, C, H, W = data['B'].shape

    data['hint_B'] = torch.zeros_like(data['B'])
    data['mask_B'] = torch.zeros_like(data['A'])

    for nn in range(N):
        # Determine the number of points to add using a ternary operator

        point_count = 11 if num_points is None else num_points

        P = 6
        # points = add_color_patches_hybrid(H, W, P, point_count, data, opt)
        # points = add_color_patches_superpixel_kmeans(H, W, P, point_count, data)
        # points = add_color_patches_rand_uniform(H, W, P, point_count)
        points = add_color_patches_kmeans(H, W, P, point_count, data, opt)
        # points = add_color_patches_spill_the_bucket(data, point_count, opt)
        # points = add_color_patches_blob_detector(data, point_count, opt)

        for h, w in points:
            mean_height = torch.mean(data['B'][nn, :, h:h + P, w:w + P], dim=2, keepdim=True)
            mean = torch.mean(mean_height, dim=1, keepdim=True)

            data['hint_B'][nn, :, h:h + P, w:w + P] = mean.view(1, C, 1, 1)  # Reshape the tensor to the correct shape
            data['mask_B'][nn, :, h:h + P, w:w + P] = 1

    return data


def add_color_patches_hybrid(H: int, W: int, P: int, n: int, data, opt) -> List[Tuple[int, int]]:
    bucket_points = math.ceil(n / 2 if n < 10 else n / 10)
    points = add_color_patches_spill_the_bucket(data, bucket_points, opt)

    blob_points = math.ceil((n - len(points)) / 10)
    points += add_color_patches_blob_detector(data, blob_points, opt)

    kmeans_spxl_points = math.ceil((n - len(points)) / 2)
    points += add_color_patches_superpixel_kmeans(H, W, P, kmeans_spxl_points, data)

    kmeans_points = n - len(points)
    points += add_color_patches_kmeans(H, W, P, kmeans_points, data, opt)

    return points


def add_color_patches_superpixel_kmeans(H: int, W: int, P: int, n: int, data, compactness=60, region_size=10):
    """
    Selects the best color patches based on superpixel segmentation and clustering superpixels using KMeans.

    Args:
        H (int): Height of the image.
        W (int): Width of the image.
        P (int): Patch size.
        n (int): Number of patches (clusters).
        data (dict): Dictionary containing the image data in Lab color space.
        compactness (float): Compactness parameter for superpixel segmentation.
        region_size (int): Region size parameter for superpixel segmentation.

    Returns:
        List of (h, w) coordinates where patches should be placed.
    """

    if n == 0:
        return []

    # Convert data['A'] to grayscale image
    gray_image = data['A'].cpu().squeeze(0).squeeze(0).numpy()  # Shape: (H, W)

    # Perform SLIC superpixel segmentation
    slic = cv2.ximgproc.createSuperpixelSLIC(
        gray_image,
        algorithm=cv2.ximgproc.SLIC,
        region_size=region_size,
        ruler=compactness
    )
    slic.iterate(10)  # Number of iterations

    # Get labels for each superpixel
    labels = slic.getLabels()
    num_superpixels = np.max(labels) + 1  # The number of distinct superpixels

    # List to store the feature vector (intensity) for each superpixel
    superpixel_centers = []
    superpixel_sizes = []

    for i in range(num_superpixels):
        # Get all coordinates of pixels in the current superpixel
        superpixel_coords = np.argwhere(labels == i)

        # Calculate the average intensity of this superpixel
        avg_intensity = np.mean(gray_image[superpixel_coords[:, 0], superpixel_coords[:, 1]])
        superpixel_centers.append(avg_intensity)
        superpixel_sizes.append(len(superpixel_coords))

    # Convert the list of superpixel centers to a NumPy array for clustering
    superpixel_centers = np.array(superpixel_centers).reshape(-1, 1)

    # Perform KMeans clustering on the superpixel centers
    kmeans = KMeans(n_clusters=int(max(1, n)), random_state=0).fit(superpixel_centers)
    cluster_labels = kmeans.labels_

    # List to store the best patch locations (center of each cluster)
    patch_centers = []

    # For each cluster, find the superpixel with the most pixels and return its center as the patch
    for cluster in range(n):
        # Get the indices of the superpixels in the current cluster
        cluster_indices = np.where(cluster_labels == cluster)[0]

        # Find the superpixel with the most pixels in the current cluster
        # Check if the list is empty
        superpixel_sizes_list = [superpixel_sizes[i] for i in cluster_indices]
        if not superpixel_sizes_list:
            print("Warning: Empty superpixel sizes list")
            continue
        else:
            largest_superpixel_idx = cluster_indices[np.argmax(superpixel_sizes_list)]

        # Get the coordinates of the largest superpixel's center
        superpixel_coords = np.argwhere(labels == largest_superpixel_idx)

        # Get the center of the selected superpixel (the mean position)
        best_h, best_w = np.mean(superpixel_coords, axis=0).astype(int)

        # Ensure the patch is within image bounds
        patch_centers.append((best_h, best_w))

    return patch_centers


def add_color_patches_rand_geometric(H: int, W: int, P: int, n: int):
    points = []

    for i in range(n):
        h = int(np.clip(np.random.normal((H - P + 1) / 2., (H - P + 1) / 4.), 0, H - P))
        w = int(np.clip(np.random.normal((W - P + 1) / 2., (W - P + 1) / 4.), 0, W - P))

        points.append((h, w))

    return points


def add_color_patches_custom(H: int, W: int, P: int, n: int):
    return [
        (310, 170),
        (350, 150),
        (280, 120),
        (300, 100),
        # (100, 50),
        # (270, 256),
        # (100, 250), # roze
        # (250, 350), # paars
        # (360, 350), # wit
        # (310, 170),
    ]


def add_color_patches_rand_uniform(H: int, W: int, P: int, n: int):
    points = []

    for i in range(n):
        h = np.random.randint(H - P + 1)
        w = np.random.randint(W - P + 1)

        points.append((h, w))

    return points


def add_color_patches_kmeans(H: int, W: int, P: int, n: int, data, opt):
    if n == 0:
        return []

    # Use only the luminance channel
    luminance = data['A'].view(-1).cpu().numpy().reshape(-1, 1)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n, random_state=0).fit(luminance)
    labels = kmeans.labels_

    # Initialize list to store the best points
    points = []
    for cluster in range(n):
        # Get the indices of the pixels in the current cluster
        cluster_indices = np.where(labels == cluster)[0]

        # Check if cluster_indices is empty
        if len(cluster_indices) == 0:
            print(f"Warning: Empty cluster indices for cluster {cluster}")
            continue

        # Randomly select a pixel from the cluster
        idx = np.random.choice(cluster_indices)
        h, w = divmod(idx, W)
        points.append((h, w))

    return points


def add_color_patches_spill_the_bucket(data, num_points: int, opt) -> List[Tuple[int, int]]:
    img = tensor2im(lab2rgb(torch.cat((data['A'], data['B']), dim=1), opt))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    kp = []

    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
    non_zero = 0
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            seedPoint = (i, j)
            if mask[seedPoint[1] + 1, seedPoint[0] + 1] == 1:
                continue
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            cv2.floodFill(img, mask, seedPoint, color, (3, 3, 3), (15, 15, 15))
            cnt_non_zero = cv2.countNonZero(mask)
            kp.append((seedPoint, cnt_non_zero - non_zero))
            non_zero = cnt_non_zero

    # if not enough points, return what we have :(
    if len(kp) < num_points:
        return list(map(lambda x: x[0], kp))
    # return the top num_points points
    return list(map(lambda x: x[0], sorted(kp, key=lambda x: x[1], reverse=True)[:num_points]))


def add_color_patches_blob_detector(data, num_points: int, opt) -> List[Tuple[int, int]]:
    img = tensor2im(lab2rgb(torch.cat((data['A'], data['B']), dim=1), opt))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    params = cv2.SimpleBlobDetector_Params()

    params.blobColor = 255
    params.minThreshold = 10
    params.maxThreshold = 200

    params.filterByArea = True
    params.minArea = 10

    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    if len(keypoints) < num_points:
        params.filterByArea = False
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(img)

    # if not enough points, return wat we have :(
    if len(keypoints) > num_points:  # if enough points, sort by size and keep the largest
        sorted_keypoints = sorted(keypoints, key=lambda kp: kp.size, reverse=True)
        keypoints = sorted_keypoints[:num_points]

    # clean up keypoints and return as List[tuple[int, int]]
    return list(map(lambda kp: (int(kp.pt[0]), int(kp.pt[1])), keypoints))


def add_color_patch(data, mask, opt, P=1, hw=[128, 128], ab=[0, 0]):
    # Add a color patch at (h,w) with color (a,b)
    data[:, 0, hw[0]:hw[0] + P, hw[1]:hw[1] + P] = 1. * ab[0] / opt.ab_norm
    data[:, 1, hw[0]:hw[0] + P, hw[1]:hw[1] + P] = 1. * ab[1] / opt.ab_norm
    mask[:, :, hw[0]:hw[0] + P, hw[1]:hw[1] + P] = 1 - opt.mask_cent

    return (data, mask)


def crop_mult(data, mult=16, HWmax=[800, 1200]):
    # crop image to a multiple
    H, W = data.shape[2:]
    Hnew = int(min(H / mult * mult, HWmax[0]))
    Wnew = int(min(W / mult * mult, HWmax[1]))
    h = int((H - Hnew) / 2)
    w = int((W - Wnew) / 2)

    return data[:, :, h:h + Hnew, w:w + Wnew]


def encode_ab_ind(data_ab, opt):
    # Encode ab value into an index
    # INPUTS
    #   data_ab   Nx2xHxW \in [-1,1]
    # OUTPUTS
    #   data_q    Nx1xHxW \in [0,Q)

    data_ab_rs = torch.round((data_ab * opt.ab_norm + opt.ab_max) / opt.ab_quant)  # normalized bin number
    data_q = data_ab_rs[:, [0], :, :] * opt.A + data_ab_rs[:, [1], :, :]
    return data_q


def decode_ind_ab(data_q, opt):
    # Decode index into ab value
    # INPUTS
    #   data_q      Nx1xHxW \in [0,Q)
    # OUTPUTS
    #   data_ab     Nx2xHxW \in [-1,1]

    data_a = data_q / opt.A
    data_b = data_q - data_a * opt.A
    data_ab = torch.cat((data_a, data_b), dim=1)

    if (data_q.is_cuda):
        type_out = torch.cuda.FloatTensor
    else:
        type_out = torch.FloatTensor
    data_ab = ((data_ab.type(type_out) * opt.ab_quant) - opt.ab_max) / opt.ab_norm

    return data_ab


def decode_max_ab(data_ab_quant, opt):
    # Decode probability distribution by using bin with highest probability
    # INPUTS
    #   data_ab_quant   NxQxHxW \in [0,1]
    # OUTPUTS
    #   data_ab         Nx2xHxW \in [-1,1]

    data_q = torch.argmax(data_ab_quant, dim=1)[:, None, :, :]
    return decode_ind_ab(data_q, opt)


def decode_mean(data_ab_quant, opt):
    # Decode probability distribution by taking mean over all bins
    # INPUTS
    #   data_ab_quant   NxQxHxW \in [0,1]
    # OUTPUTS
    #   data_ab_inf     Nx2xHxW \in [-1,1]

    (N, Q, H, W) = data_ab_quant.shape
    a_range = torch.arange(-opt.ab_max, opt.ab_max + opt.ab_quant, step=opt.ab_quant).to(data_ab_quant.device)[None, :,
              None, None]
    a_range = a_range.type(data_ab_quant.type())

    # reshape to AB space
    data_ab_quant = data_ab_quant.view((N, int(opt.A), int(opt.A), H, W))
    data_a_total = torch.sum(data_ab_quant, dim=2)
    data_b_total = torch.sum(data_ab_quant, dim=1)

    # matrix multiply
    data_a_inf = torch.sum(data_a_total * a_range, dim=1, keepdim=True)
    data_b_inf = torch.sum(data_b_total * a_range, dim=1, keepdim=True)

    data_ab_inf = torch.cat((data_a_inf, data_b_inf), dim=1) / opt.ab_norm

    return data_ab_inf


def calculate_psnr_np(img1, img2):
    import numpy as np
    SE_map = (1. * img1 - img2) ** 2
    cur_MSE = np.mean(SE_map)
    return 20 * np.log10(255. / np.sqrt(cur_MSE))


import numpy as np
from skimage import color


def calculate_ciede2000(img1, img2):
    """
    Calculate the CIEDE2000 difference between two images and return the score.

    Args:
        img1: First image in Lab color space.
        img2: Second image in Lab color space.

    Returns:
        score: The calculated score based on the CIEDE2000 difference.
    """
    # Ensure the images are in Lab color space
    if img1.shape[-1] != 3 or img2.shape[-1] != 3:
        raise ValueError("Both images must be in Lab color space with 3 channels.")

    # Calculate the CIEDE2000 difference for each pixel
    diff = color.deltaE_ciede2000(img1, img2)

    # Take the average of the differences
    avg_diff = np.mean(diff)

    # Calculate the final score
    score = 1 / (1 + avg_diff)

    return score


def calculate_psnr_torch(img1, img2):
    SE_map = (1. * img1 - img2) ** 2
    cur_MSE = torch.mean(SE_map)
    return 20 * torch.log10(1. / torch.sqrt(cur_MSE))
