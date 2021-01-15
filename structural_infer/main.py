#-*-coding:utf-8-*-

import os
import argparse
from solver import Solver
from dataloader import DataLoader
from utils import create_folder, setup_seed
from config import get_config
import torch
from tester import Tester


def main(config):
    # for fast training.
    torch.backends.cudnn.benchmark = True

    setup_seed(1990)
    
    # create directories if not exist.
    create_folder(config.save_root_dir, config.version, config.model_save_path)
    create_folder(config.save_root_dir, config.version, config.sample_path)
    create_folder(config.save_root_dir, config.version, config.sample_pretrain_path)
    create_folder(config.save_root_dir, config.version, config.sample_enhanced_path)
    create_folder(config.save_root_dir, config.version, config.log_path)
    create_folder(config.save_root_dir, config.version, config.validation_path)
    create_folder(config.save_root_dir, config.version, config.test_result_path)

    # data loader.
    train_loader = DataLoader(dataset=config.train_dataset,
                               data_dir_raw=config.train_data_dir_raw,
                               data_dir_exp=config.train_data_dir_exp,
                               csv_file=config.train_csv_file,
                               root_dir=config.data_root_dir,
                               image_size=config.image_size,
                               resize_size=config.resize_size,
                               batch_size=config.batch_size,
                               shuffle=config.shuffle,
                               num_workers=config.num_workers,
                               dropLast=config.dropLast
    )
    val_loader = DataLoader(dataset=config.val_dataset,
                            data_dir_raw=config.val_data_dir_raw,
                            data_dir_exp=config.val_data_dir_exp,
                            csv_file=config.val_csv_file,
                            root_dir=config.data_root_dir,
                            image_size=config.image_size,
                            resize_size=config.resize_size,
                            batch_size=config.batch_size,
                            shuffle=config.shuffle,
                            num_workers=config.num_workers,
                            dropLast=config.dropLast
    )
    test_loader = DataLoader(dataset=config.test_dataset,
                            data_dir_raw=config.test_data_dir_raw,
                            data_dir_exp=config.test_data_dir_exp,
                            csv_file=config.test_csv_file,
                            root_dir=config.data_root_dir,
                            image_size=config.image_size,
                            resize_size=config.resize_size,
                            batch_size=config.batch_size,
                            shuffle=config.shuffle,
                            num_workers=config.num_workers,
                            dropLast=config.dropLast
    )

    if config.mode == 'train':
        trainer = Solver(train_loader, val_loader.loader(), test_loader.loader(), config)
        trainer.train()
    elif config.mode == 'test':
        tester = Tester(test_loader.loader(), config)
        tester.test()
    else:
        raise NotImplementedError('Mode [{}] is not found'.format(config.mode))


if __name__ == '__main__':

    config = get_config()
    
    if config.is_print_network:
        print(config)
        
    main(config)