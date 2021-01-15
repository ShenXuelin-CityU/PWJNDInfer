import os

import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


from nima.train.utils import SCORE_NAMES


class AVADataset(Dataset):
    def __init__(self, path_to_csv: str, images_path: str, transform):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        y = np.array([row[k] for k in SCORE_NAMES])
        p = y / y.sum()

        image_id = row['image_id']
        image_path = os.path.join(self.images_path, f'{image_id}.jpg')
        image = default_loader(image_path)
        x = self.transform(image)
        return x, p.astype('float32')
