# -*- coding: utf-8 -*-
import torch
import os
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from math import log10
from losses import SSIM
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calc_ssim(pred_path, gt_path, result_save_path, epoch):

    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    criterionSSIM = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3).to(device)
    ssim, total_ssim, avg_ssim = 0.0, 0.0, 0.0
    epoch_result = result_save_path + 'SSIM_epoch_' + str(epoch) + '.csv'
    epochfile = open(epoch_result, 'w')
    epochfile.write('image_name' + ','+ 'ssim' + '\n')

    total_result = result_save_path + 'SSIM_total_results_epoch_avgssim.csv'
    totalfile = open(total_result, 'a+')

    print('======================= start to calculate ssim =======================')
    test_imgs = [f for f in os.listdir(pred_path)]
    for i, img in enumerate(test_imgs):
        pred_pil = Image.open(os.path.join(pred_path, img)).convert("RGB")
        pred_tensor = transform(pred_pil)
        pred = pred_tensor.to(device).unsqueeze(0)
        imgName, _, _ = img.rsplit('_', 2)
        gt_imgName = imgName + '.jpg'
        gt = Image.open(os.path.join(gt_path, gt_imgName)).convert("RGB")
        gt_tensor = transform(gt)
        gt = gt_tensor.to(device).unsqueeze(0)

        ssim = criterionSSIM(pred, gt)

        total_ssim += ssim
        if i % 50 == 0:
            print("=== ssim is processing {:>3d}-th image ===".format(i))
    print("======================= Complete the SSIM test of {:>3d} images ======================= ".format(i+1))
    avg_ssim = total_ssim / i
    epochfile.write('Average' + ',' + str(round(avg_ssim.item(), 6)) + '\n')
    epochfile.close()
    totalfile.write(str(epoch) + ',' + str(round(avg_ssim.item(), 6)) + '\n')
    totalfile.close()
    return avg_ssim.item()

