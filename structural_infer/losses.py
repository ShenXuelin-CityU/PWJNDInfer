# -*-coding:utf-8-*-

import torch
import torch.nn.functional as F
from math import exp
import torch.nn as nn
import torchvision.models as models
import os
from math import pi


class PerceptualLoss(nn.Module):
    """
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.add_module('vgg', VGG19_conv())
        # self.add_module('vgg', VGG19_relu())
        self.add_module('vgg', VGG16_relu())
        # self.add_module('vgg', VGG16bn_relu())
        # self.criterion = torch.nn.L1Loss()
        # self.criterion = torch.nn.SmoothL1Loss()
        self.criterion = torch.nn.MSELoss()
        self.weights = torch.ones(1, 5).squeeze(0).to(self.device)
    
    def normalize(self, tensors, mean, std):
        if not torch.is_tensor(tensors):
            raise TypeError('tensor is not a torch image')
        for tensor in tensors:
            for t, m, s in zip(tensor, mean, std):
                t.sub_(m).div_(s)
        return tensors

    def __call__(self, x, y):

        x = self.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        y = self.normalize(y, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        # features before relu
        # content_loss += self.weights[0] * self.criterion(x_vgg['conv1_2'], y_vgg['conv1_2'])
        # content_loss += self.weights[1] * self.criterion(x_vgg['conv2_2'], y_vgg['conv2_2'])
        # content_loss += self.weights[2] * self.criterion(x_vgg['conv3_4'], y_vgg['conv3_4'])
        # content_loss += self.weights[3] * self.criterion(x_vgg['conv4_4'], y_vgg['conv4_4'])
        # content_loss += self.weights[4] * self.criterion(x_vgg['conv5_1'], y_vgg['conv5_1'])

        # features after relu
        # content_loss += self.weights[0] * self.criterion(x_vgg['relu1_2'], y_vgg['relu1_2'])
        # content_loss += self.weights[1] * self.criterion(x_vgg['relu2_2'], y_vgg['relu2_2'])
        # content_loss += self.weights[2] * self.criterion(x_vgg['relu3_3'], y_vgg['relu3_3'])
        # content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        return content_loss


class VGG16_relu(torch.nn.Module):
    def __init__(self):
        super(VGG16_relu, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cnn = models.vgg16(pretrained=False)
        cnn.load_state_dict(torch.load(os.path.join('./', 'vgg16-397923af.pth')))
        cnn = cnn.to(self.device)
        features = cnn.features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 19):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(19, 21):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(23, 26):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(26, 28):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(28, 30):
            self.relu5_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)

        relu4_1 = self.relu4_1(relu3_3)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)

        relu5_1 = self.relu5_1(relu4_3)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
        }
        return out


class VGG16bn_relu(torch.nn.Module):
    def __init__(self):
        super(VGG16bn_relu, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cnn = models.vgg16_bn(pretrained=False)
        cnn.load_state_dict(torch.load(os.path.join('./', 'vgg16_bn-6c64b313.pth')))
        cnn = cnn.to(self.device)
        features = cnn.features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()

        for x in range(3):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(3, 6):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(6, 10):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(10, 13):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(13, 17):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(17, 20):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(20, 23):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(23, 27):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(30, 33):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(33, 37):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(37, 40):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(40, 43):
            self.relu5_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)

        relu4_1 = self.relu4_1(relu3_3)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)

        relu5_1 = self.relu5_1(relu4_3)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
        }
        return out


class VGG19_conv(torch.nn.Module):
    def __init__(self):
        super(VGG19_conv, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cnn = models.vgg19(pretrained=False)
        cnn.load_state_dict(torch.load(os.path.join('./', 'vgg19-dcbb9e9d.pth')))
        cnn = cnn.to(self.device)
        features = cnn.features
        # features = models.vgg19(pretrained=True).to(self.device).features
        self.conv1_1 = torch.nn.Sequential()
        self.conv1_2 = torch.nn.Sequential()

        self.conv2_1 = torch.nn.Sequential()
        self.conv2_2 = torch.nn.Sequential()

        self.conv3_1 = torch.nn.Sequential()
        self.conv3_2 = torch.nn.Sequential()
        self.conv3_3 = torch.nn.Sequential()
        self.conv3_4 = torch.nn.Sequential()

        self.conv4_1 = torch.nn.Sequential()
        self.conv4_2 = torch.nn.Sequential()
        self.conv4_3 = torch.nn.Sequential()
        self.conv4_4 = torch.nn.Sequential()

        self.conv5_1 = torch.nn.Sequential()
        self.conv5_2 = torch.nn.Sequential()
        self.conv5_3 = torch.nn.Sequential()
        self.conv5_4 = torch.nn.Sequential()

        for x in range(1):
            self.conv1_1.add_module(str(x), features[x])

        for x in range(1, 3):
            self.conv1_2.add_module(str(x), features[x])

        for x in range(3, 6):
            self.conv2_1.add_module(str(x), features[x])

        for x in range(6, 8):
            self.conv2_2.add_module(str(x), features[x])

        for x in range(8, 11):
            self.conv3_1.add_module(str(x), features[x])

        for x in range(11, 13):
            self.conv3_2.add_module(str(x), features[x])

        for x in range(13, 15):
            self.conv3_3.add_module(str(x), features[x])

        for x in range(15, 17):
            self.conv3_4.add_module(str(x), features[x])

        for x in range(17, 20):
            self.conv4_1.add_module(str(x), features[x])

        for x in range(20, 22):
            self.conv4_2.add_module(str(x), features[x])

        for x in range(22, 24):
            self.conv4_3.add_module(str(x), features[x])

        for x in range(24, 26):
            self.conv4_4.add_module(str(x), features[x])

        for x in range(26, 29):
            self.conv5_1.add_module(str(x), features[x])

        for x in range(29, 31):
            self.conv5_2.add_module(str(x), features[x])

        for x in range(31, 33):
            self.conv5_3.add_module(str(x), features[x])

        for x in range(33, 35):
            self.conv5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        conv1_1 = self.conv1_1(x)
        conv1_2 = self.conv1_2(conv1_1)

        conv2_1 = self.conv2_1(conv1_2)
        conv2_2 = self.conv2_2(conv2_1)

        conv3_1 = self.conv3_1(conv2_2)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_3 = self.conv3_3(conv3_2)
        conv3_4 = self.conv3_4(conv3_3)

        conv4_1 = self.conv4_1(conv3_4)
        conv4_2 = self.conv4_2(conv4_1)
        conv4_3 = self.conv4_3(conv4_2)
        conv4_4 = self.conv4_4(conv4_3)

        conv5_1 = self.conv5_1(conv4_4)
        conv5_2 = self.conv5_2(conv5_1)
        conv5_3 = self.conv5_3(conv5_2)
        conv5_4 = self.conv5_4(conv5_3)

        out = {
            'conv1_1': conv1_1,
            'conv1_2': conv1_2,

            'conv2_1': conv2_1,
            'conv2_2': conv2_2,

            'conv3_1': conv3_1,
            'conv3_2': conv3_2,
            'conv3_3': conv3_3,
            'conv3_4': conv3_4,

            'conv4_1': conv4_1,
            'conv4_2': conv4_2,
            'conv4_3': conv4_3,
            'conv4_4': conv4_4,

            'conv5_1': conv5_1,
            'conv5_2': conv5_2,
            'conv5_3': conv5_3,
            'conv5_4': conv5_4,
        }
        return out


class VGG19_relu(torch.nn.Module):
    def __init__(self):
        super(VGG19_relu, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cnn = models.vgg19(pretrained=False)
        cnn.load_state_dict(torch.load(os.path.join('./', 'vgg19-dcbb9e9d.pth')))
        cnn = cnn.to(self.device)
        features = cnn.features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


# SSIM & MS_SSIM
def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution

    Returns:
        torch.Tensor: 1D kernel
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size//2

    g = torch.exp(-(coords**2) / (2*sigma**2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blured
        window (torch.Tensor): 1-D gauss kernel

    Returns:
        torch.Tensor: blured tensors
    """

    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out


def _ssim(X, Y, win, data_range=255, size_average=True, full=False):
    r""" Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not

    Returns:
        torch.Tensor: ssim results
    """

    K1 = 0.01
    K2 = 0.03
    # batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range)**2
    C2 = (K2 * data_range)**2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * ( gaussian_filter(X * X, win) - mu1_sq )
    sigma2_sq = compensation * ( gaussian_filter(Y * Y, win) - mu2_sq )
    sigma12   = compensation * ( gaussian_filter(X * Y, win) - mu1_mu2 )

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
        ssim_val = ssim_map.mean(-1).mean(-1).mean(-1)  # reduce along CHW
        cs = cs_map.mean(-1).mean(-1).mean(-1)

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ssim(X, Y, win_size=11, win_sigma=1.5, win=None, data_range=255, size_average=True, full=False):
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not

    Returns:
        torch.Tensor: ssim results
    """

    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    ssim_val, cs = _ssim(X, Y, win=win, data_range=data_range, size_average=False, full=True)
    if size_average:
        ssim_val = ssim_val.mean()
        cs = cs.mean()

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ms_ssim(X, Y, win_size=11, win_sigma=1.5, win=None, data_range=255, size_average=True, full=False, weights=None):
    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
        weights (list, optional): weights for different levels

    Returns:
        torch.Tensor: ms-ssim results
    """
    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if weights is None:
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(X.device, dtype=X.dtype)

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    levels = weights.shape[0]
    mcs = []
    for _ in range(levels):
        ssim_val, cs = _ssim(X, Y, win=win, data_range=data_range, size_average=False, full=True)
        mcs.append(cs)

        padding = (X.shape[2] % 2, X.shape[3] % 2)
        X = F.avg_pool2d(X, kernel_size=2, padding=padding)
        Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    mcs = torch.stack(mcs, dim=0)  # mcs, (level, batch)
    # weights, (level)
    msssim_val = torch.prod((mcs[:-1] ** weights[:-1].unsqueeze(1)) * (ssim_val ** weights[-1]), dim=0)  # (batch, )

    if size_average:
        msssim_val = msssim_val.mean()
    return msssim_val


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=None, size_average=True, channel=3):
        r""" class for ssim
        Args:
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            channel (int, optional): input channels (default: 3)
        """

        super(SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range

    def forward(self, X, Y):
        return ssim(X, Y, win=self.win, data_range=self.data_range, size_average=self.size_average)


class MS_SSIM(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=None, size_average=True, channel=3, weights=None):
        r""" class for ms-ssim
        Args:
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
        """

        super(MS_SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights

    def forward(self, X, Y):
        return ms_ssim(X, Y, win=self.win, size_average=self.size_average, data_range=self.data_range, weights=self.weights)


class SS(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SS, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.sigma = 1.5
        self.val_range = val_range
    
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - self.window_size//2)**2/float(2*sigma**2)) for x in range(self.window_size)])
        return gauss / gauss.sum()
        
    def create_window(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, self.sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, self.window_size, self.window_size).contiguous()
        return window

    def ss(self, img1, img2, window, window_size, channel, size_average=True, full=False, val_range=None):
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        if val_range is None:
            if torch.max(img1) > 128:
                max_val = 255
            else:
                max_val = 1

            if torch.min(img1) < -0.5:
                min_val = -1
            else:
                min_val = 0
            L = max_val - min_val
        else:
            L = val_range

        # padd = 0
        padd = self.window_size // 2
        mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        sigma1 = torch.sqrt(torch.abs(sigma1_sq))
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma2 = torch.sqrt(torch.abs(sigma2_sq))
        sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

        # C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2
        
        ss_map = (sigma12 + C2) / (sigma1*sigma2 + C2)

        if size_average:
            return ss_map.mean()
        else:
            return ss_map.mean(1).mean(1).mean(1)
    
    def get_ss_score(self, img1, img2):
        (_, channel, height, width) = img1.size()
        real_size = min(self.window_size, height, width)
        window = self.create_window(real_size, channel=channel).to(img1.device)
            
        return self.ss(img1, img2, window, self.window_size, channel, self.size_average, full=False)


class GMS(torch.nn.Module):
    """"gradient magnitude similarity"""
    def __init__(self):
        super(GMS, self).__init__()
    
    def gradOperator(self, channel):
        grad_kernel_size = 3
        db = torch.Tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 3.0
        _dx = db.unsqueeze(0).unsqueeze(0)
        _dy = db.t().unsqueeze(0).unsqueeze(0)
        dx = _dx.expand(channel, 1, grad_kernel_size, grad_kernel_size).contiguous()
        dy = _dy.expand(channel, 1, grad_kernel_size, grad_kernel_size).contiguous()
        return dx, dy
    
    def _gms(self, img1, img2, dx, dy, channel, mode, val_range=None):
        if val_range is None:
            if torch.max(img1) > 128:
                max_val = 255
            else:
                max_val = 1

            if torch.min(img1) < -0.5:
                min_val = -1
            else:
                min_val = 0
            L = max_val - min_val
        else:
            L = val_range
        
        # padd = 0
        padd = 1
        if mode == 'horizontal':
            GImg1 = F.conv2d(img1, dx, padding=padd, groups=channel)
            GImg2 = F.conv2d(img2, dx, padding=padd, groups=channel)
        elif mode == 'vertical':
            GImg1 = F.conv2d(img1, dy, padding=padd, groups=channel)
            GImg2 = F.conv2d(img2, dy, padding=padd, groups=channel)
        else:
            raise NotImplementedError("GMS mode [{}] is not found".format(mode))
        
        T = (0.1 * L) ** 2
        gms_map = (2 * GImg1 * GImg2 + T) / (GImg1 ** 2 + GImg2 ** 2 + T)

        return gms_map.mean()

    def get_gms_loss(self, img1, img2, mode):
        (_, channel, _, _) = img1.size()
        dx, dy = self.gradOperator(channel)
        if img1.is_cuda:
            dx = dx.cuda(img1.get_device())
            dy = dy.cuda(img1.get_device())
        dx = dx.type_as(img1)
        dy = dy.type_as(img1)

        gms = self._gms(img1, img2, dx, dy, channel, mode)
        return gms


class ES(torch.nn.Module):
    """"gradient magnitude similarity"""
    def __init__(self):
        super(ES, self).__init__()
        self.channel = 1
    
    def gradOperator(self, channel):
        grad_kernel_size = 3
        db = torch.Tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]]) / 8.0
        _db = db.unsqueeze(0).unsqueeze(0)
        db = _db.expand(channel, 1, grad_kernel_size, grad_kernel_size).contiguous()
        return db
    
    def _es(self, img1, img2, db, channel, val_range=None):
        if val_range is None:
            if torch.max(img1) > 128:
                max_val = 255
            else:
                max_val = 1

            if torch.min(img1) < -0.5:
                min_val = -1
            else:
                min_val = 0
            L = max_val - min_val
        else:
            L = val_range
        
        # padd = 0
        padd = 1
        GImg1 = F.conv2d(img1, db, padding=padd, groups=channel)
        GImg2 = F.conv2d(img2, db, padding=padd, groups=channel)
        
        T = (0.1 * L) ** 2
        gms_map = (2 * GImg1 * GImg2 + T) / (GImg1 ** 2 + GImg2 ** 2 + T)

        return gms_map.mean()

    def get_es_loss(self, img1, img2):
        (_, channel, _, _) = img1.size()
        db = self.gradOperator(channel)
        if img1.is_cuda:
            db = db.cuda(img1.get_device())
        db = db.type_as(img1)

        es = self._es(img1, img2, db, channel)
        return es


class AngularLoss(torch.nn.Module):
    def __init__(self):
        super(AngularLoss, self).__init__()

    def forward(self, feature1, feature2):
        cos_criterion = torch.nn.CosineSimilarity(dim=1)
        cos = cos_criterion(feature1, feature2)
        cos = torch.clamp(cos, -0.99999, 0.99999)
        if False:
            return 1 - torch.mean(cos)
            # return torch.mean(torch.atan2((torch.norm(torch.cross(feature1, feature2), dim=1)), (torch.sum(feature1 * feature2, dim=1))) * 180 / pi)
        else:
            return torch.mean(torch.acos(cos)) * 180 / pi
            # return 1 - torch.mean(cos)
            # return 1 - torch.mean(cos**2)





