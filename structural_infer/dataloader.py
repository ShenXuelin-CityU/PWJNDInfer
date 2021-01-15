
# -*-coding:utf-8-*-

import os
import torch
import cv2
from torch.utils import data
from PIL import Image, ImageFile
import pandas as pd
from torchvision import transforms


class MyCustomDataset(data.Dataset):
    def __init__(self, csv_file, data_dir_raw, data_dir_exp, root_dir, transform):
        self.pairs = pd.read_csv(csv_file, sep=',', header=None)
        self.data_dir_raw = data_dir_raw
        self.data_dir_exp = data_dir_exp
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """Return the number of images."""
        return len(self.pairs)

    def __getitem__(self, index):
        """Return one image and its corresponding unpaired image"""
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        num_row, num_col = self.pairs.shape

        if num_col == 1:
            img_path1 = os.path.join(self.root_dir, self.data_dir_raw, str(self.pairs.iloc[index, 0]))
            img_path2 = os.path.join(self.root_dir, self.data_dir_exp, str(self.pairs.iloc[index, 0])) # paired high quality image
            image1 = Image.open(img_path1)
            # image1 = image1.convert("L")
            image2 = Image.open(img_path2)
            # image2 = image2.convert("L")
            name = str(self.pairs.iloc[index, 0])
            imgName, _ = name.split('.', 1)
            if self.transform:
                try:
                    image1 = self.transform(image1)
                    image2 = self.transform(image2)
                except:
                    print("Cannot transform images: {} and {}".format(img_path1, img_path2))
            return image1, image2, imgName

        elif num_col == 2:
            img_path1 = os.path.join(self.root_dir, self.data_dir_raw, str(self.pairs.iloc[index, 0])) # low-quality image
            img_path2 = os.path.join(self.root_dir, self.data_dir_exp, str(self.pairs.iloc[index, 1])) # unpaired high quality image
            #img_path3 = os.path.join(self.root_dir, self.data_dir_exp, str(self.pairs.iloc[index, 1])) # paired high quality image
            image1 = Image.open(img_path1)
            image2 = Image.open(img_path2)
            # print(len(image2.split()))
            # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            #image2 = cv2.imread(img_path2,1)
            #image3 = Image.open(img_path3)


            #image3 = cv2.imread(img_path3,1)
            name = str(self.pairs.iloc[index, 0])
            imgName, _ = name.split('.', 1)
            if self.transform:
                try:
                    image1 = self.transform(image1)
                    image2 = self.transform(image2)
                    #image3 = self.transform(image3)
                except:
                    print("Cannot transform images: {}, {} and {}".format(img_path1, img_path2))
            return image1, image2, imgName

class DataLoader():
    def __init__(self, dataset, data_dir_raw, data_dir_exp, csv_file, root_dir, image_size, resize_size, batch_size, shuffle, num_workers, dropLast):
        self.dataset = dataset
        self.data_dir_raw = data_dir_raw
        self.data_dir_exp = data_dir_exp
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.resize_size = resize_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.dropLast = dropLast

    def __make_power_32(self, img, base, method=Image.BICUBIC):
        ow, oh = img.size
        h = int(round(oh / base) * base)
        w = int(round(ow / base) * base)
        if (h == oh) and (w == ow):
            return img

        print('image resized from {:} x {:} to {:} x {:}'.format(ow, oh, w, h))
        return img.resize((w, h), method)


    def transform(self, MakePower32, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomCrop, CenterCrop, Resize, ToTensor, Normalize):
        transform_options = []
        if MakePower32:
            transform_options.append(transforms.Lambda(lambda img: self.__make_power_32(img, base=32, method=Image.BICUBIC)))
        if RandomHorizontalFlip:
            transform_options.append(transforms.RandomHorizontalFlip(p=0.5))
        if RandomVerticalFlip:
            transform_options.append(transforms.RandomVerticalFlip(p=0.5))
        if ColorJitter:
            transform_options.append(transforms.ColorJitter(brightness=0, contrast=0.15, saturation=0))
        if RandomCrop:
            transform_options.append(transforms.RandomCrop(self.image_size, padding=0, pad_if_needed=False))
        if CenterCrop:
            transform_options.append(transforms.CenterCrop(self.image_size))
        if Resize:
            transform_options.append(transforms.Resize(self.resize_size))
        if ToTensor:
            transform_options.append(transforms.ToTensor())
        if Normalize:
            transform_options.append(transforms.Normalize([0.5], [0.5]))
            # transform_options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(transform_options)
        return transform

    def load_trainSet(self):
        """Build and return the training data loader"""
        train_transform = self.transform(False, False, False, False, False, False, False, True, True)
        trainSet = MyCustomDataset(
            csv_file=self.csv_file,
            data_dir_raw=self.data_dir_raw,
            data_dir_exp=self.data_dir_exp,
            root_dir=self.root_dir,
            transform=train_transform
        )
        return trainSet

    def load_valSet(self):
        """Build and return the validation data loader"""
        val_transform = self.transform(False, False, False, False, False, False, False, True, True)
        valSet = MyCustomDataset(
            csv_file=self.csv_file,
            data_dir_raw=self.data_dir_raw,
            data_dir_exp=self.data_dir_exp,
            root_dir=self.root_dir,
            transform=val_transform
        )
        return valSet

    def load_testSet(self):
        """Build and return the validation data loader"""
        test_transform = self.transform(False, False, False, False, False, False, False, True, True)
        testSet = MyCustomDataset(
            csv_file=self.csv_file,
            data_dir_raw=self.data_dir_raw,
            data_dir_exp=self.data_dir_exp,
            root_dir=self.root_dir,
            transform=test_transform
        )
        return testSet

    def loader(self):
        """Build and return a data loader"""
        if self.dataset == 'train':
            self.dataset_in = self.load_trainSet()
            dataLoader = torch.utils.data.DataLoader(
                dataset=self.dataset_in,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                drop_last=self.dropLast,
                # pin_memory=True
            )
            return dataLoader
        elif self.dataset == 'val':
            self.dataset_in = self.load_valSet()
            dataLoader = torch.utils.data.DataLoader(
                dataset=self.dataset_in,
                batch_size=1,
                # shuffle=self.shuffle,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=self.dropLast,
                # pin_memory=True
            )
            return dataLoader
        elif self.dataset == 'test':
            self.dataset_in = self.load_testSet()
            dataLoader = torch.utils.data.DataLoader(
                dataset=self.dataset_in,
                batch_size=1,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                drop_last=self.dropLast,
                # pin_memory=True
            )
            return dataLoader
