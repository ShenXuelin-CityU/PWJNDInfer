#-*-coding:utf-8-*-

import argparse
from utils import str2bool


def get_config():
    parser = argparse.ArgumentParser()
    
    # Model configuration.
    parser.add_argument('--mode', type=str, default='train', help='train|test')
    parser.add_argument('--image_size', type=int, default=64, help='image load resolution')
    parser.add_argument('--resize_size', type=int, default=64, help='resolution after resizing')
    parser.add_argument('--g_conv_dim', type=int, default=32, help='number of conv filters in the first layer of G')
    parser.add_argument('--shuffle', type=str, default=True, help='shuffle when load dataset')
    parser.add_argument('--dropLast', type=str2bool, default=False, help=' drop the last incomplete batch')
    parser.add_argument('--version', type=str, default='JND', help='JND')
    parser.add_argument('--init_type', type=str, default='kaiming', help='normal|xavier|kaiming|orthogonal')
    parser.add_argument('--upsample_type', type=str, default='nn', help='nn|bilinear|subpixel|deconv')#nn/bilinear
    parser.add_argument('--g_use_sn', type=str2bool, default=True, help='whether use spectral normalization in G')

    # Training configuration.
    parser.add_argument('--pretrained_model', type=int, default=95676)#pretrained
    parser.add_argument('--total_epochs', type=int, default=40, help='total epochs to update the generator')#40（100）
    parser.add_argument('--batch_size', type=int, default=20, help='mini batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='subprocesses to use for data loading')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--lr_decay', type=str2bool, default=True, help='setup learning rate decay schedule')
    parser.add_argument('--lr_num_epochs_decay', type=int, default=20, help='LambdaLR: epoch at starting learning rate')#half 20（50）
    parser.add_argument('--lr_decay_ratio', type=int, default=20, help='LambdaLR: ratio of linearly decay learning rate to zero')#half 20（50）
    parser.add_argument('--optimizer_type', type=str, default='adam', help='adam|rmsprop')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--alpha', type=float, default=0.9, help='alpha for rmsprop optimizer')
    parser.add_argument('--pair_shuffle', type=str2bool, default=False, help='shuffle unpaired image pair for each epoch')

    # validation and test configuration
    parser.add_argument('--test_epochs', type=int, default=200, help='test model from this epoch')
    parser.add_argument('--num_epochs_start_val', type=int, default=0, help='start validate the model')
    parser.add_argument('--val_epochs', type=int, default=1, help='validate the model every time after training these epochs')

    # Directories.
    parser.add_argument('--data_root_dir', type=str, default='./data/')
    parser.add_argument('--save_root_dir', type=str, default='./results')
    parser.add_argument('--train_dataset', type=str, default='train')
    parser.add_argument('--train_data_dir_raw', type=str, default='OriginalPatch/')
    parser.add_argument('--train_data_dir_exp', type=str, default='GroundPatch/', help='exp|FlickrHDR')
    parser.add_argument('--val_dataset', type=str, default='val')
    parser.add_argument('--val_data_dir_raw', type=str, default='val_ori/')
    parser.add_argument('--val_data_dir_exp', type=str, default='val_gt/')
    parser.add_argument('--test_dataset', type=str, default='test')
    parser.add_argument('--test_data_dir_raw', type=str, default='test/')
    parser.add_argument('--test_data_dir_exp', type=str, default='exp_all/')
    parser.add_argument('--model_save_path', type=str, default='models')
    parser.add_argument('--sample_path', type=str, default='samples')
    parser.add_argument('--sample_pretrain_path', type=str, default='samples/pretrain')
    parser.add_argument('--sample_enhanced_path', type=str, default='samples/enhanced')
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--validation_path', type=str, default='validation')
    parser.add_argument('--test_result_path', type=str, default='test')
    parser.add_argument('--train_csv_file', type=str, default='./data/train.csv', help='csv file for training images')
    parser.add_argument('--val_csv_file', type=str, default='./data/val.csv', help='csv file for validation images')
    parser.add_argument('--test_csv_file', type=str, default='./data/test.csv', help='csv file for images')

    # step size
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_epoch', type=int, default=1) 

    # Misc    
    parser.add_argument('--parallel', type=str2bool, default=False, help='use multi-GPU for training')
    parser.add_argument('--gpu_ids', default=[0, 1, 2, 3])
    parser.add_argument('--use_tensorboard', type=str, default=True)
    parser.add_argument('--is_print_network', type=str2bool, default=False)

    return parser.parse_args()