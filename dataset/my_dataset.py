import numpy as np
import cv2
import os
import torch
import json
from torch.utils.data.dataset import Dataset
from pathlib import Path

def get_random_crop(image, crop_h, crop_w):
    h, w = image.shape[:2]
    max_x = w - crop_w
    max_y = h - crop_h
    
    x = np.random.randint(0, max_x+1)
    y = np.random.randint(0, max_y+1)
    crop = image[y: y + crop_h, x: x + crop_w, :]
    return crop


def get_center_crop(image):
    h, w = image.shape[:2]
    if h > w:
        start_w = 0
        start_h = (h - w) // 2
        end_w = w
        end_h = start_h + w
    else:
        start_h = 0
        start_w = (w - h) // 2
        end_h = h
        end_w = start_w + h
        
    return image[start_h:end_h, start_w:end_w, :]


class MyDataset(Dataset):
    r"""
    Minimal image dataset where we take mnist images
    add a texture background
    change the color of the digit.
    Model trained on this dataset is then required to predict the below 3 values
    1. Class of texture
    2. Class of number
    3. R, G, B values (0-1) of the digit color
    """
    def __init__(self, split, config, im_h=224, im_w=224):
        self.split = split
        self.db_root = config['dataset_params']['root_dir']
        self.im_h = im_h
        self.im_w = im_w
        self.class_map = config['class_labels']

        im_dir = self.db_root + "/" + self.split 
        self.im_list = [str(p) for p in Path(im_dir).glob("*/*.jpg")]
        self.im_posix_list = list(Path(im_dir).glob("*/*.jpg"))
        print(self.im_list[0:5])
        
        
    def __len__(self):
        return len(self.im_list)
    
    def __getitem__(self, index):

        image_path = self.im_list[index]
        im_posix_path = self.im_posix_list[index]
        classname = im_posix_path.parent.stem
        
        im = cv2.imread(image_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (self.im_h, self.im_w))

        if self.split == 'train':
            im = get_random_crop(im, self.im_h, self.im_w)
        else:
            im = get_center_crop(im)
            im = cv2.resize(im, (self.im_h, self.im_w))
        
     
        im_tensor = torch.from_numpy(im).permute((2, 0, 1))
        im_tensor = 2 * (im_tensor / 255) - 1
        return {
            "image" : im_tensor,
            "number_cls" : self.class_map[classname]
        }