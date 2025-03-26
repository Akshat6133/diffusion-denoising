from skimage import io, transform
from configs import *
# import matplotlib
# matplotlib.use("TkAgg")  # Or "Qt5Agg"
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import warnings
warnings.filterwarnings("ignore")


class MyDataset(Dataset):
    def __init__(self, original_dir, mf_file_names, target_dir, transform=None):
        self.original_dir = original_dir
        self.mf_file_names = mf_file_names
        self.target_dir = target_dir
        self.transform = transform

    def __len__(self):
        return len(self.mf_file_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.original_dir, self.mf_file_names[idx])
        target_name = os.path.join(self.target_dir, self.mf_file_names[idx])
        print(f"Reading image: {img_name}")
        print(f"Reading target: {target_name}")
        # image = Image.open(img_name)

        image = io.imread(img_name)
        target = io.imread(target_name)
        sample = {"image": image, "target": target}

        return sample



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        # swap color axis because
        image = image.transpose((2, 0, 1))
        target = target.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'target': torch.from_numpy(target)}
        


