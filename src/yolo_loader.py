from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, RandomSampler, WeightedRandomSampler, SequentialSampler
import os
import torchvision.transforms as transforms
import torch
import numpy as np
from data_loader import get_labels, get_label_to_idx

def file_to_index():
    file_name = 'data/TrainImagesYes.txt'
    with open(file_name, 'r') as f:
        lines = f.readlines()

    return {line.strip().split('Images/')[1] : idx for idx, line in enumerate(lines)}


class CustomDataset(Dataset):
    def __init__(self, image_paths, txt_path, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.label_to_idx = file_to_index()
        self.image_labels = [self.label_to_idx[label] for label in get_labels(txt_path)]
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):        
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.image_labels[index]
        return image, label

print(file_to_index())