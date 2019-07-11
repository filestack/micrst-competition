import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from sklearn.model_selection import train_test_split


class MICRSTDataset(Dataset):
    def __init__(self, df, config):
        self.config = config
        self.df = df

        # TODO Add your own transformations
        self.transform = Compose([Normalization()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df.iloc[idx, 0]

        img_path = self.df.iloc[idx, 1]
        img = cv2.imread(img_path)
        sample = {
            'img': img,
            'label': label
        }

        sample = self.transform(sample)

        return sample


class Normalization(object):
    def __call__(self, sample, stats=None):
        # Make it so the dataset has mean = 0, std = 1
        img = sample['img']

        # This may not be the best way to normalize
        # this type of data - black will change into
        # a various shades of gray - you could prevent
        # that by using a global value of mean and std
        img = img - img.mean()
        img = img / img.std()
        sample["img"] = img

        return sample


def make_dataset_loaders(config, batch_size=17, test_size=0.1):
    data_root_dir = config['root_directory']

    labels_path = os.path.join(data_root_dir, 'labels.csv')
    df = pd.read_csv(labels_path)

    trainf, testf = train_test_split(
        df,
        test_size=test_size,
        random_state=137
    )

    train_set = MICRSTDataset(trainf, config)
    test_set = MICRSTDataset(testf, config)

    BATCH_SIZE = batch_size
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    return train_loader, test_loader
