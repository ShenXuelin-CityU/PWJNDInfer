#-*- codign:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import gray_scale, GaussianSmoothing


class Generator(nn.Module):
    """Generator network"""
    def __init__(self, conv_dim, upsample_type, use_sn):
        super(Generator, self).__init__()
        # self.enc1 = EncConvBlock(in_channels=3, out_channels=conv_dim, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, use_sn=use_sn) # 256*256*3 --> 256*256*32
        self.enc1 = EncConvBlock(in_channels=1, out_channels=conv_dim, kernel_size=7, stride=1, padding=0, dilation=1,
                                 use_bias=True, use_sn=use_sn)  # 256*256*3 --> 256*256*32
        self.enc2 = EncConvBlock(in_channels=conv_dim, out_channels=conv_dim*2, kernel_size=3, stride=2, padding=0, dilation=1, use_bias=True, use_sn=use_sn) # 256*256*32 --> 128*128*64

        self.enc3 = EncConvBlock(in_channels=conv_dim*2, out_channels=conv_dim*4, kernel_size=3, stride=2, padding=0, dilation=1, use_bias=True, use_sn=use_sn) # 128*128*64 --> 64*64*128

        self.enc4 = EncConvBlock(in_channels=conv_dim*4, out_channels=conv_dim*8, kernel_size=3, stride=2, padding=0, dilation=1, use_bias=True, use_sn=use_sn) # 64*64*128 --> 32*32*256

        self.enc5 = EncConvBlock(in_channels=conv_dim*8, out_channels=conv_dim*16, kernel_size=3, stride=2, padding=0, dilation=1, use_bias=True, use_sn=use_sn) # 32*32*256 --> 16*16*512
        
        self.upsample1 = DecUpsampleBlock(in_channels=conv_dim*16, out_channels=conv_dim*8, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, use_sn=use_sn, upsample_type=upsample_type) # 16*16*512 --> 32*32*256

        self.dec1 = DecConvBlock(in_channels=conv_dim*16, out_channels=conv_dim*8, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, use_sn=use_sn) # 32*32*512 --> 32*32*256

        self.upsample2 = DecUpsampleBlock(in_channels=conv_dim*8, out_channels=conv_dim*4, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, use_sn=use_sn, upsample_type=upsample_type) # 32*32*256 --> 64*64*128

        self.dec2 = DecConvBlock(in_channels=conv_dim*8, out_channels=conv_dim*4, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, use_sn=use_sn) # 64*64*256 --> 64*64*128

        self.upsample3 = DecUpsampleBlock(in_channels=conv_dim*4, out_channels=conv_dim*2, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, use_sn=use_sn, upsample_type=upsample_type) # 64*64*128 --> 128*128*64

        self.dec3 = DecConvBlock(in_channels=conv_dim*4, out_channels=conv_dim*2, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, use_sn=use_sn) # 128*128*128 --> 128*128*64

        self.upsample4 = DecUpsampleBlock(in_channels=conv_dim*2, out_channels=conv_dim*1, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, use_sn=use_sn, upsample_type=upsample_type) # 128*128*64 --> 256*256*32

        self.dec4 = DecConvBlock(in_channels=conv_dim*2, out_channels=conv_dim*1, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, use_sn=use_sn) # 256*256*64 --> 256*256*32

        self.dec5 = []
        self.dec5 = [nn.ReflectionPad2d(3)]
        #self.dec5 += [SpectralNorm(nn.Conv2d(in_channels=conv_dim*1, out_channels=3, kernel_size=3, stride=1, padding=0, dilation=1, bias=True), mode=False)]
        self.dec5 += [SpectralNorm(
            nn.Conv2d(in_channels=conv_dim * 1, out_channels=1, kernel_size=7, stride=1, padding=0, dilation=1,
                      bias=True), mode=False)]
        # self.dec5 += [nn.Tanh()]
        self.dec5 = nn.Sequential(*self.dec5)

        self.sa = SelfAttention(in_dim=conv_dim*16, use_sn=use_sn)

        self.skip_ga1 = GlobalAttention(in_channels=conv_dim*16, out_channels=conv_dim*16, reduction_ratio=16, use_sn=use_sn, pool_types=['avg'], global_feature=True, ca=False)
        self.skip_ga2 = GlobalAttention(in_channels=conv_dim*8, out_channels=conv_dim*8, reduction_ratio=16, use_sn=use_sn, pool_types=['avg'], global_feature=True, ca=False)
        self.skip_ga3 = GlobalAttention(in_channels=conv_dim*4, out_channels=conv_dim*4, reduction_ratio=16, use_sn=use_sn, pool_types=['avg'], global_feature=True, ca=False)
        self.skip_ga4 = GlobalAttention(in_channels=conv_dim*2, out_channels=conv_dim*2, reduction_ratio=16, use_sn=use_sn, pool_types=['avg'], global_feature=True, ca=False)
        self.skip_ga5 = GlobalAttention(in_channels=conv_dim*1, out_channels=conv_dim*1, reduction_ratio=16, use_sn=use_sn, pool_types=['avg'], global_feature=True, ca=False)
    
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        x5_1 = self.enc5(x4)
        x5_1, _ = self.sa(x5_1)
        x5 = self.skip_ga1(x5_1)

        y1_1 = self.upsample1(x5)
        y1_2 = torch.cat([y1_1, self.skip_ga2(x4)], dim=1)
        y1 = self.dec1(y1_2)

        y2_1 = self.upsample2(y1)
        y2_2 = torch.cat([y2_1, self.skip_ga3(x3)], dim=1)
        y2 = self.dec2(y2_2)

        y3_1 = self.upsample3(y2)
        y3_2 = torch.cat([y3_1, self.skip_ga4(x2)], dim=1)
        y3 = self.dec3(y3_2)

        y4_1 = self.upsample4(y3)
        y4_2 = torch.cat([y4_1, self.skip_ga5(x1)], dim=1)
        y4 = self.dec4(y4_2)

        out = self.dec5(y4)
#10.16_1907注释
        out = out + x

        return torch.clamp(out, -1.0, 1.0)


class EncConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, use_sn):
        super(EncConvBlock, self).__init__()
        self.padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.main = nn.Sequential(
            nn.ReflectionPad2d(self.padding),
            SpectralNorm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=use_bias), use_sn),
            # nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True),
            # nn.SELU(inplace=True),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.main(x)


class DecConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, use_sn):
        super(DecConvBlock, self).__init__()
        self.padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.main = nn.Sequential(
            nn.ReflectionPad2d(self.padding),
            SpectralNorm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=use_bias), use_sn),
            # nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True),
            # nn.SELU(inplace=True),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.main(x)

        
class DecUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, use_sn, upsample_type):
        super(DecUpsampleBlock, self).__init__()
        self.upsample_type = upsample_type
        self.padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.nn_main = nn.Sequential(
            Interpolate(scale_factor=2, mode='nearest', align_corners=None),
            nn.ReflectionPad2d(self.padding),
            SpectralNorm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=use_bias), use_sn),
        )
        self.bilinear_main = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReflectionPad2d(self.padding),
            SpectralNorm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=use_bias), use_sn),
        )
        self.subpixel_main = nn.Sequential(
            nn.ReflectionPad2d(self.padding),
            SpectralNorm(nn.Conv2d(in_channels=in_channels, out_channels=in_channels*2, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=use_bias), use_sn),
            nn.PixelShuffle(2),
            avg_blur(),
        )
        self.deconv_main = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=True), mode=use_sn),
        )
    def forward(self, x):
        if self.upsample_type == 'nn':
            return self.nn_main(x)
        elif self.upsample_type == 'bilinear':
            return self.bilinear_main(x)
        elif self.upsample_type == 'subpixel':
            return self.subpixel_main(x)
        elif self.upsample_type == 'deconv':
            return self.deconv_main(x)
        else:
            raise NotImplementedError('Upsample type [{}] in Decoder is not implemented'.format(self.upsample_type))


def avg_blur():
    main = nn.Sequential(
        nn.ReflectionPad2d((1, 0, 1, 0)),
        nn.AvgPool2d(kernel_size=2, stride=1, padding=0),
    )
    return main


def SpectralNorm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        
    def forward(self, x):
        out = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return out


class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim, use_sn):
        super(SelfAttention, self).__init__()
        self.query_conv = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim// 8, kernel_size=1), mode=use_sn)
        self.key_conv = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim// 8, kernel_size=1), mode=use_sn)
        self.value_conv = SpectralNorm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1), mode=use_sn)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature 
                attention: B * N * N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B * C * N --> B * N * C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B * C * (W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # B * N * N
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B * C * N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        
        out = self.gamma*out + x
        return out, attention


class GlobalAttention(nn.Module):
    """Spatial and Channel attention"""
    def __init__(self, in_channels, out_channels, reduction_ratio, use_sn, pool_types=['avg'], global_feature=False, ca=False):
        super(GlobalAttention, self).__init__()
        self.channelatten = ca
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_feature_size = 1
        self.global_pool = nn.AdaptiveAvgPool2d((self.global_feature_size, self.global_feature_size))
        self.mlp = nn.Sequential(
            SpectralNorm(nn.Linear(in_features=in_channels, out_features=in_channels//reduction_ratio), use_sn),
            nn.ReLU(inplace=True),
            SpectralNorm(nn.Linear(in_features=in_channels//reduction_ratio, out_features=out_channels), use_sn),
        )
        self.sigmoid = nn.Sigmoid()    
        self.pool_types = pool_types

        self.global_feature = global_feature
        self.fc = nn.Sequential(
            SpectralNorm(nn.Linear(in_features=in_channels * (self.global_feature_size ** 2), out_features=(self.global_feature_size ** 2) * in_channels//reduction_ratio), use_sn),
            nn.ReLU(inplace=True),
            SpectralNorm(nn.Linear(in_features=(self.global_feature_size ** 2) * in_channels//reduction_ratio, out_features=in_channels), use_sn),
        )
        self.conv = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1, stride=1, padding=0), use_sn),
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        if self.global_feature:
            gfeature = self.global_pool(x).view(b, c * (self.global_feature_size ** 2))
            gfeature = self.fc(gfeature)
            # gfeature = torch.sigmoid(gfeature)
            gfeature = gfeature.unsqueeze(2).unsqueeze(3).expand_as(x)
            x = torch.cat([x, gfeature], dim=1)
            x = self.conv(x)

        if self.channelatten:
            channel_att_sum = None        
            for pool_type in self.pool_types:
                if pool_type=='avg':
                    avg_pool = self.avg_pool(x).view(b, c)
                    channel_att_raw = self.mlp( avg_pool )
                elif pool_type=='max':
                    max_pool = self.max_pool(x).view(b, c)
                    channel_att_raw = self.mlp( max_pool )
                elif pool_type=='var':
                    var_pool = torch.var(x.view(x.size(0), x.size(1), 1, -1), dim=3, keepdim=True).view(b, c)
                    channel_att_raw = self.mlp(var_pool)

                if channel_att_sum is None:
                    channel_att_sum = channel_att_raw
                else:
                    channel_att_sum = channel_att_sum + channel_att_raw

            scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
            x = x * scale

        return x