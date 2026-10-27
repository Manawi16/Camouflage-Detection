import os
import cv2
import torch
from torch.utils.data import Dataset

class CamouflageDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.images = os.listdir(image_dir)
        self.image_dir = image_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png"))

        image = cv2.imread(img_path)
        image = cv2.resize(image, (352, 352)) / 255.0
        image = torch.tensor(image).permute(2,0,1).float()

        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (352, 352)) / 255.0
        mask = torch.tensor(mask).unsqueeze(0).float()

        return image, mask