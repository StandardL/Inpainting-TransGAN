# -*- coding: utf-8 -*-
# @Date    : 2019-07-26
# @Modify  : 2024-01-11
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @ModifyAuthor : nidie
# @Link    : None
# @Version : 0.0


import os
import glob
import argparse
import numpy as np
import torch
from imageio import imread
from pytorch_fid import fid_score as fid
from inception import InceptionV3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default='../val_128',
        help='set path to training set jpg images dir')
    parser.add_argument(
        '--output_file',
        type=str,
        default='../fid_stat/fid_stats_Places365_train.npz',
        help='path for where to store the statistics')

    opt = parser.parse_args()
    print(opt)
    return opt


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ########
    # PATHS
    ########
    data_path = args.data_path
    output_path = args.output_file
    print("load inception model..", end=" ", flush=True)
    inception = InceptionV3()
    inception = inception.to(device)
    print("ok")

    # loads all images into memory (this might require a lot of RAM!)
    print("load images..", end=" ", flush=True)
    image_list = glob.glob(os.path.join(data_path, '*.jpg'))
    # images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])
    print("%d images found and loaded" % len(image_list))

    print("calculte FID stats..", end=" ", flush=True)
    mu, sigma = fid.calculate_activation_statistics(image_list, inception, device=device, num_workers=0)
    np.savez_compressed(output_path, mu=mu, sigma=sigma)
    print("finished")


if __name__ == '__main__':
    main()
