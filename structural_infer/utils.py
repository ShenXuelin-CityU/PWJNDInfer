#-*-coding:utf-8-*-

import os
import torch
import torch.nn as nn
import math
import numbers
import torch.nn.functional as F
import numpy as np
import csv
import random
import tensorflow as tf
import scipy.misc 
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images


class Logger(object):
    """Create a tensorboard logger to log_dir."""
    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = tf.compat.v1.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
    
    def image_summary(self, tag, images, step):
        """Log a list of images."""
        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.compat.v1.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.compat.v1.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.compat.v1.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)


def shuffle_column(csv_file):
    first_column = []
    second_column = []

    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        first_column = [row[0] for row in reader]

    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        second_column = [row[1] for row in reader]

    # if pair_shuffle:
    random.shuffle(first_column)
    random.shuffle(second_column)

    # final_file_dir = './data/train.csv'
    final_file = open(csv_file, 'w')
    for i, j in zip(first_column, second_column):
        final_file.write(i)
        final_file.write(',')
        final_file.write(j)
        final_file.write('\n')


def create_folder(root_dir, path, version):
        if not os.path.exists(os.path.join(root_dir, path, version)):
            os.makedirs(os.path.join(root_dir, path, version))


def var2tensor(x):
    return x.data.cpu()


def var2numpy(x):
    return x.data.cpu().numpy()


def denorm(x):
    out = (x + 1) / 2.0
    return out.clamp_(0, 1)
    

def denorm_nzk(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x.cpu()[0] + 1) / 2
    return out
    

def str2bool(v):
    return v.lower() in ('true')


def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed seperately for each channel in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data. Default value is 2 (spatial).
    """
    def __init__(self, channels=3, kernel_size=21, sigma=3, dim=2):
        super(GaussianSmoothing, self).__init__()
        self.padding = nn.ReflectionPad2d(kernel_size//2)
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid( [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ] )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError( 'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim) )

    def forward(self, input):
        input = self.padding(input)
        out = self.conv(input, weight=self.weight, groups=self.groups)

        return out


def gray_scale(image):
    """
    image  : (batch_size, image_size), image_size = image_width * image_height * channels
    return : (batch_size, image_size with one channel)
    """
    # rgb_image = np.reshape(image, newshape=(-1, channels, height, width)) # 3 channel which is rgb
    # gray_image = np.reshape(gray_image, newshape=(-1, 1, height, width))
    # gray_image = torch.from_numpy(gray_image)

    gray_image = torch.unsqueeze(image[:, 0] * 0.299 + image[:, 1] * 0.587 + image[:, 2] * 0.114, 1)

    return gray_image
    
