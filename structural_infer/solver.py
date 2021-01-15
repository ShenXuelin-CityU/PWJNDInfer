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


class Solver(object):
    def __init__(self, train_data_loader, val_data_loader, test_data_loader, config):

        # data loader
        self.train_data = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        
        # Model configuration.
        self.g_conv_dim = config.g_conv_dim
        self.version = config.version
        self.init_type = config.init_type
        self.upsample_type = config.upsample_type
        self.g_use_sn = config.g_use_sn

        # Training configuration.
        self.pretrained_model = config.pretrained_model
        self.total_epochs = config.total_epochs
        self.batch_size = config.batch_size
        self.g_lr = config.g_lr
        self.lr_decay = config.lr_decay
        self.lr_num_epochs_decay = config.lr_num_epochs_decay
        self.lr_decay_ratio = config.lr_decay_ratio
        self.optimizer_type = config.optimizer_type
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.alpha = config.alpha
        self.pair_shuffle = config.pair_shuffle
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # validate and test configuration.
        self.test_epochs = config.test_epochs
        self.num_epochs_start_val = config.num_epochs_start_val
        self.val_epochs = config.val_epochs

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

        
    def train(self):
        """ Train UEGAN ."""
        # Set data loader.
        self.train_data_loader = self.train_data.loader()
        train_data_iter = iter(self.train_data_loader)
        train_steps_per_epoch = len(self.train_data_loader)
        self.model_save_step = self.model_save_epoch * train_steps_per_epoch

        # set nima, psnr, ssim global parameters
        if True:
            best_psnr_epoch, best_psnr = 0, 0.0
            best_ssim_epoch, best_ssim = 0, 0.0
        
        # start from scratch or trained models
        if self.pretrained_model:
            start_step = self.pretrained_model + 1
            self.load_pretrained_model(self.pretrained_model)
        else:
            start_step = 0
    
        # start training
        print("======================================= start training =======================================")
        # start time
        start_time = time.time()
        elapsed_p = 0.0
        total_steps = self.total_epochs * train_steps_per_epoch
        val_start_steps = int(self.num_epochs_start_val * train_steps_per_epoch)
                
        """"========================== iteratively train generator and discriminator ==========================="""
        print("=========== start to train generator ===========")
        for step in range(start_step, total_steps):
            self.G.train()

            # Decay learning rates of the generator and discriminator (once every epoch)
            if (step + 1) % train_steps_per_epoch == 0:
                self.lr_scheduler_g.step()
                for param_group in self.g_optimizer.param_groups:
                    print("====== Epoch: {:>3d}/{}, Learning rate(lr) of Encoder(E) and Generator(G): [{}], ".format(((step + 1) // train_steps_per_epoch) + 1, self.total_epochs, param_group['lr']), end='')

            try:
                real_raw, real_exp, real_raw_name = next(train_data_iter)
            except:
                train_data_iter = iter((self.train_data_loader))
                real_raw, real_exp, real_raw_name = next(train_data_iter)

            real_raw = real_raw.to(self.device)
            real_exp = real_exp.to(self.device)

            """ ======================= train the generator ======================= """
            fake_exp = self.G(real_raw)
            loss = {}
            criterion = nn.L1Loss()
            # ssimLoss = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)
            g_content_loss = 1.0 * criterion(fake_exp, real_exp)
            # g_ssim_loss = 1.0 * (1.0 - ssimLoss((fake_exp+1.)/2., (real_exp+1.)/2.))
            #print('real_raw, fake_exp, real_exp: ', real_raw.size(), fake_exp.size(), real_exp.size())
            # g_loss = g_content_loss + g_ssim_loss
            g_loss = g_content_loss
            loss['G/content_loss'] = g_content_loss.item()
            # loss['G/ssim_loss'] = g_ssim_loss.item()
            loss['G/Total'] = g_loss.item()

            self.g_optimizer.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()


            """ ======== print out training info, save samples, and save model checkpoints ======= """
            # print out training info
            if (step + 1) % self.log_step == 0:
                elapsed_num = time.time() - start_time - elapsed_p
                elapsed = str(datetime.timedelta(seconds=elapsed_num))
                current_epoch = 1 + (step + 1) // train_steps_per_epoch
                print("Elapse:{:>.12s}, RD:{:.4f} h, Epoch:{:>3d}/{}, Step:{:>6d}/{}, G_loss:{:>.4f}".format(elapsed, (elapsed_num/(step+1))*(total_steps-step)/3600, \
                                                                                             current_epoch, self.total_epochs, (step + 1), total_steps, g_content_loss.item()))
                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, step + 1)
                    # for tag, images in G_Images.items():
                    #     self.logger.image_summary(tag, images, step + 1)

            # sample images
            if (step + 1) % self.sample_step == 0:
                for i in range(0, denorm(real_raw.data).size(0)):
                    current_epoch = 1 + (step + 1) // train_steps_per_epoch
                    save_imgs = torch.cat([denorm(real_raw.data)[i:i + 1,:,:,:], denorm(fake_exp.data)[i:i + 1,:,:,:], denorm(real_exp.data)[i:i + 1,:,:,:]], 3)
                    save_image(save_imgs, os.path.join(self.sample_path, '{:s}_{:0>3d}_{:0>2d}_realRaw_fakeExp_realExp.png'.format(real_raw_name[i], current_epoch, i)))

                    save_imgs_fake_exp = denorm(fake_exp.data)[i:i + 1,:,:,:]
                    save_image(save_imgs_fake_exp, os.path.join(self.sample_enhanced_path, '{:s}_{:0>3d}_{:0>2d}_fakeExp.png'.format(real_raw_name[i], current_epoch, i)))
                print("=== Save real_raw, fake_exp, and real_exp image into {} ===".format(self.sample_path))

            # save models
            if (step + 1) % self.model_save_step == 0:
                if self.parallel:
                    if torch.cuda.device_count() > 1:
                        torch.save(self.G.module.state_dict(), os.path.join(self.model_save_path, '{}_{}_G.pth'.format(self.version, (step + 1))))
                else:
                    torch.save(self.G.state_dict(), os.path.join(self.model_save_path, '{}_{}_G.pth'.format(self.version,  (step + 1))))
                print("======= Save model checkpoints {}_{}_E.pth into {} ======".format(self.version, (step + 1), self.model_save_path))


            """"======================= validation training model ======================="""
            if (step + 1) > val_start_steps:
                if (step + 1) % (self.val_epochs * train_steps_per_epoch) == 0:
                    validation = {}
                    current_epoch = 1 + (step + 1) // train_steps_per_epoch
                    validation_save_path = self.validation_path + '/' + 'validation_' + str(current_epoch)
                    validation_compare_save_path = self.validation_path + '/' + 'validatio_compare_' + str(current_epoch)

                    if not os.path.exists(validation_save_path):
                        os.makedirs(validation_save_path)
                    if not os.path.exists(validation_compare_save_path):
                        os.makedirs(validation_compare_save_path)

                    self.G.eval()

                    print("============================== Start validation ==============================")
                    with torch.no_grad():
                        for val_step, val_data in enumerate(self.val_data_loader):
                            # self.G.eval()
                            val_real_raw, val_real_raw_label, val_name = val_data[0], val_data[1], val_data[2]
                            val_real_raw = val_real_raw.to(self.device)
                            val_real_raw_label = val_real_raw_label.to(self.device)
                            val_fake_exp = self.G(val_real_raw)

                            for i in range(0, denorm(val_real_raw.data).size(0)):
                                save_imgs = denorm(val_fake_exp.data)[i:i + 1,:,:,:]
                                save_image(save_imgs, os.path.join(validation_save_path, '{:s}_{:0>3d}_valFakeExp.png'.format(val_name[i], current_epoch)))

                                save_imgs_compare = torch.cat([denorm(val_real_raw.data)[i:i + 1,:,:,:], denorm(val_fake_exp.data)[i:i + 1,:,:,:], denorm(val_real_raw_label.data)[i:i + 1,:,:,:]], 3)
                                save_image(save_imgs_compare, os.path.join(validation_compare_save_path, '{:s}_{:0>3d}_valRealRaw_valFakeExp_valLabel.png'.format(val_name[i], current_epoch)))

                            elapsed = time.time() - start_time
                            elapsed = str(datetime.timedelta(seconds=elapsed))
                            if val_step % self.log_step == 0:
                                print("=== Elapse:{}, Save {:>3d}-th val_fake_exp images into {} ===".format(elapsed, val_step, validation_save_path))

                            validation['Validation/valFakeExp'] = denorm(val_fake_exp.detach().cpu()).numpy()
                            validation['Validation_compare/valRealRaw_valFakeExp_valRealRawLabel'] = torch.cat([denorm(val_real_raw.cpu()), denorm(val_fake_exp.detach().cpu()), denorm(val_real_raw_label.cpu())], 3).numpy()
                            #if self.use_tensorboard:
                                # for tag, images in validation.items():
                                #     self.logger.image_summary(tag, images, val_step + 1)

                        if True:
                            psnr_save_path = './results/psnr_val_results/'
                            gt_path = os.path.join(self.data_root_dir, self.val_data_dir_exp)
                            curr_psnr = calc_psnr(validation_save_path, gt_path, psnr_save_path, current_epoch)
                            if best_psnr < curr_psnr:
                                best_psnr = curr_psnr
                                best_psnr_epoch = current_epoch
                            print("====== Avg. PSNR: {:>.4f} dB ======".format(curr_psnr))

        if True:
            psnr_save_path = './results/psnr_val_results/'
            psnr_result = psnr_save_path + 'PSNR_total_results_epoch_avgpsnr.csv'
            psnrfile = open(psnr_result, 'a+')
            psnrfile.write('Best epoch: ' + str(best_psnr_epoch) + ',' + str(round(best_psnr, 6)) + '\n')
            psnrfile.close()

        print('============================== Complete the training ==============================')

    def test(self):
        self.load_pretrained_model(self.pretrained_model)
        start_time = time.time()
        test = {}

        print('============================== Start testing ==============================')
        with torch.no_grad():
            for test_step, test_data in enumerate(self.test_data_loader):
                self.G.eval()

                test_real_raw, test_name = test_data[0], test_data[1][0] 

                test_real_raw = test_real_raw.to(self.device)
                test_fake_exp = self.G(test_real_raw)

                for i in range(0, denorm(test_real_raw.data).size(0)):
                    save_imgs = denorm(test_fake_exp.data)[i:i + 1,:,:,:]

                    save_image(save_imgs, os.path.join(self.test_result_path, '{:s}_{:0>2d}_testFakeExp.png'.format(test_name, i)))

                    save_imgs_compare = torch.cat([denorm(test_real_raw.data)[i:i + 1,:,:,:], denorm(test_fake_exp.data)[i:i + 1,:,:,:]], 3)

                    save_image(save_imgs_compare, os.path.join(self.test_result_path, '{:s}_{:0>2d}_testRealRaw_valFakeExp.png'.format(test_name, i)))

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

        # init network
        if self.init_type:
            self.init_weights(self.G, init_type=self.init_type, gain=0.02)

        # optimizer
        if self.optimizer_type == 'adam':
            # Adam optimizer
            self.g_optimizer = torch.optim.Adam(params=self.G.parameters(), lr=self.g_lr, betas=[self.beta1, self.beta2])
        elif self.optimizer_type == 'rmsprop':
            # RMSprop optimizer
            self.g_optimizer = torch.optim.RMSprop(params=self.G.parameters(), lr=self.g_lr, alpha=self.alpha)
        else:
            raise NotImplementedError("Optimizer [{}] is not found".format(self.optimizer_type))

        # learning rate decay
        if self.lr_decay:
            def lambda_rule(epoch):
                return 1.0 - max(0, epoch + 1 - self.lr_num_epochs_decay) / self.lr_decay_ratio
            self.lr_scheduler_g = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=lambda_rule)
            print("=== Set learning rate decay policy for Generator(G) ===")


    def init_weights(self, net, init_type='kaiming', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('Initialization method [{}] is not implemented'.format(init_type))
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                torch.nn.init.normal_(m.weight.data, 1.0, gain)
                torch.nn.init.constant_(m.bias.data, 0.0)
        print('Initialize network with [{}]'.format(init_type))
        net.apply(init_func)


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


    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
    

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        self.logger = Logger(self.log_path)
