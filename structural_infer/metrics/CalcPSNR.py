# -*- coding: utf-8 -*-
import torch
import os
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from math import log10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calc_psnr(pred_path, gt_path, result_save_path, epoch):

    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    criterionMSE = nn.MSELoss().to(device)
    psnr, total_psnr, avg_psnr = 0.0, 0.0, 0.0
    epoch_result = result_save_path + 'PSNR_epoch_' + str(epoch) + '.csv'
    epochfile = open(epoch_result, 'w')
    epochfile.write('image_name' + ','+ 'psnr' + '\n')

    total_result = result_save_path + 'PSNR_total_results_epoch_avgpsnr.csv'
    totalfile = open(total_result, 'a+')

    print('======================= start to calculate PSNR =======================')
    test_imgs = [f for f in os.listdir(pred_path)]
    valid_i = 0
    for i, img in enumerate(test_imgs):
        pred_pil = Image.open(os.path.join(pred_path, img))

        pred_tensor = transform(pred_pil)
        pred = pred_tensor.to(device)

        imgName, _, _ = img.rsplit('_', 2)
        gt_imgName = imgName + '.bmp'
        gt_pil = Image.open(os.path.join(gt_path, gt_imgName))
        gt_tensor = transform(gt_pil)
        gt = gt_tensor.to(device)
        gt = torch.cat([gt,gt,gt], dim=0)

        mse = criterionMSE(pred, gt)
        # psnr = 10 * log10(1 / mse.item())
        eps = 0.00001
        psnr = 10 * log10(1 / (mse.item() + eps))

        if mse.item() > eps:
            total_psnr += psnr
            valid_i += 1
            epochfile.write(gt_imgName + ',' + str(round(psnr, 6)) + '\n')
        if i % 200 == 0:
            print("=== PSNR is processing {:>3d}-th image ===".format(i))
    print("======================= Complete the PSNR test of {:>3d} images ======================= ".format(i+1))
    # avg_psnr = total_psnr / i
    avg_psnr = total_psnr / valid_i
    epochfile.write('Average' + ',' + str(round(avg_psnr, 6)) + '\n')
    epochfile.close()
    totalfile.write(str(epoch) + ',' + str(round(avg_psnr, 6)) + '\n')
    totalfile.close()
    print('valid_i is ',  valid_i)
    return avg_psnr
