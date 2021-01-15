#-*- coding:utf-8 -*-


import os
import time
import torch
import datetime
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from losses import PerceptualLoss, SSIM, MS_SSIM, TVLoss
from utils import shuffle_column, Logger, denorm, ImagePool, GaussianSmoothing
from models import Generator
import torchvision.models as models
from metrics.NIMA.CalcNIMA import calc_nima
from metrics.CalcPSNR import calc_psnr
from metrics.CalcSSIM import calc_ssim


class Tester(object):
    def __init__(self, test_data_loader, config):

        # data loader

        self.test_data_loader = test_data_loader
        
        # Model configuration.
        self.g_conv_dim = config.g_conv_dim
        self.version = config.version
        self.init_type = config.init_type
        self.upsample_type = config.upsample_type
        self.g_use_sn = config.g_use_sn

        # Training configuration.
        self.pretrained_model = config.pretrained_model
        self.batch_size = config.batch_size
        self.pair_shuffle = config.pair_shuffle
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        # Directories.
        self.save_root_dir = config.save_root_dir
        self.data_root_dir = config.data_root_dir
        self.val_data_dir_raw = config.val_data_dir_raw
        self.val_data_dir_exp = config.val_data_dir_exp
        self.model_save_path = os.path.join(config.save_root_dir, config.version, config.model_save_path)
        self.sample_path = os.path.join(config.save_root_dir, config.version, config.sample_path)
        self.sample_pretrain_path = os.path.join(config.save_root_dir, config.version, config.sample_pretrain_path)
        self.sample_enhanced_path = os.path.join(config.save_root_dir, config.version, config.sample_enhanced_path)
        self.log_path = os.path.join(config.save_root_dir, config.version, config.log_path)
        self.validation_path = os.path.join(config.save_root_dir, config.version, config.validation_path)
        self.test_result_path = os.path.join(config.save_root_dir, config.version, config.test_result_path)
        self.train_csv_file = config.train_csv_file

        # step size
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_epoch = config.model_save_epoch

        # Misc
        self.parallel = config.parallel
        self.gpu_ids = config.gpu_ids
        self.use_tensorboard = config.use_tensorboard
        self.is_print_network = config.is_print_network

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()


    def test(self):
        self.load_pretrained_model(self.pretrained_model)
        start_time = time.time()
        test = {}

        print('============================== Start testing ==============================')
        with torch.no_grad():
            for test_step, test_data in enumerate(self.test_data_loader):
                self.G.eval()

                test_real_raw, test_real_raw_label,test_name = test_data[0], test_data[1], test_data[2]
                test_real_raw = test_real_raw.to(self.device)
                test_real_raw_label = test_real_raw_label.to(self.device)
                test_fake_exp = self.G(test_real_raw)


                for i in range(0, denorm(test_real_raw.data).size(0)):
                    save_imgs = denorm(test_fake_exp.data)[i:i + 1,:,:,:]

                    save_image(save_imgs, os.path.join(self.test_result_path, '{:s}_{:0>2d}_testFakeExp.png'.format(test_name[i],i)))

                  #  save_image(save_imgs, os.path.join(self.test_result_path, '{:s}_{:0>2d}_testFakeExp.png'.format(test_name, i)))


                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("=== Elapse:{}, Save test_fake_exp images into {} ===".format(elapsed, self.test_result_path))


                test['Test/testFakeExp'] = denorm(test_fake_exp.detach().cpu()).numpy()
                test['Test_compare/testRealRaw_fakeExp'] = torch.cat([denorm(test_real_raw.cpu()), denorm(test_fake_exp.detach().cpu())], 3).numpy()
                # if self.use_tensorboard:
                #     for tag, images in test.items():
                #         self.logger.image_summary(tag, images, test_step + 1)




    """define some functions"""
    def build_model(self):
        """Create a encoder, a generator, and a discriminator."""
        self.G = Generator(self.g_conv_dim, self.upsample_type, self.g_use_sn).to(self.device)
        if self.parallel:
            self.G.to(self.gpu_ids[0])
            self.G = nn.DataParallel(self.G, self.gpu_ids)
        print("Models have been created")
        
        # print network
        if self.is_print_network:
            self.print_network(self.G, 'Generator')




    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print("The number of parameters of the above model [{}] is [{}] or [{:>.4f}M]".format(name, num_params, num_params / 1e6))


    def load_pretrained_model(self, resume_steps):
        G_path = os.path.join(self.model_save_path, '{}_{}_G.pth'.format(self.version, resume_steps))
        if torch.cuda.is_available():
            # save on GPU, load on GPU
            self.G.load_state_dict(torch.load(G_path))
        else:
            # save on GPU, load on CPU
            self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        print("=========== loaded trained models (step: {})! ===========".format(resume_steps))



    def build_tensorboard(self):
        """Build a tensorboard logger."""
        self.logger = Logger(self.log_path)
