from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray


class FacialKeypointsDataset(Dataset):

    def __init__(self, filename, root_dir, transform=None):

        self.fk_frame = pd.read_csv(filename)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.fk_frame)

    @staticmethod
    def downscale(frame, sh):
        points = []
        for i in range(len(frame)):
            if i % 2 == 0:
                points.append(frame[i] * 100 / sh[0])
            else:
                points.append(frame[i] * 100 / sh[1])
        return np.asarray(points)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.fk_frame.iloc[idx, 0])
        image = imread(img_name)

        keypoints = self.fk_frame.iloc[idx, 1:].values
        keypoints = keypoints.astype('float')
        keypoints = self.downscale(keypoints, image.shape)

        #image = rgb2gray(image)
        image = resize(image, (100, 100, 3))

        if image is not None:
            sample = {'image': image, 'keypoints': keypoints}

        if self.transform:
            sample = self.transform(sample)

        return sample

