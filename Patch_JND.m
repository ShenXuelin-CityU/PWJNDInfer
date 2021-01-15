%----------------------------------------------------------------------
% This is an implementation of the patch level structural
% visibility learning JND model.
%
% Please refer to the following paper
%
% Shen X, Ni Z, Yang W, et al. Just Noticeable Distortion Profile Inference: 
%A Patch-Level Structural Visibility Learning Approach[J]. IEEE Transactions on Image Processing, 2020, 30: 26-38.
%----------------------------------------------------------------------

addpath( [ cd '\tools' ] );
file_path =  '.\structural_infer\results\JND\test\';% Generated folder from the learning model
ori_patch_dir='.\structural_infer\data\test\';%Original patch folder
ori_img_dir='.\Test_images\';%Original image folder
img_path_list = dir(strcat(file_path,'*.png'));
files_name =sort_nat({img_path_list.name});

%Image numbers=19, image height=1080, image width=1920, Patchsize=64,Overlap=14
img_inf=[19,1080,1920,64,14];

%stas_p:PSNR for all test images;img_jnd:Generated JND images
[stas_p,img_jnd]=Patch_jnd_infer2(img_inf,files_name,ori_patch_dir,ori_img_dir,file_path)
