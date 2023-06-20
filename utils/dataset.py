# Imports
import os
import re
import glob
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Definim la classe IDC_Dataset, que conté les imatges, les etiquetes i les metadades de les imatges
class IDC_Dataset(Dataset):
    def __init__(self, images, labels, metadata, transform=None):
        self.images = images
        self.labels = labels
        self.metadata = metadata
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        meta = self.metadata[idx]

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(float(label)), torch.tensor(meta)


# Funció que carrega el dataset a partir del directori especificat
def load_dataset(base_path):
    images = []
    labels = []
    metadata = []

    for class_label in [0, 1]:
        class_path = os.path.join(base_path, '*', str(class_label), '*.png')
        image_files = glob.glob(class_path)

        for image_file in tqdm(image_files, desc=f"Carregant imatges {'IDC-positives' if class_label == 1 else 'IDC-negatives'}..."):
            pattern = r'(\d+)_idx5_x(\d+)_y(\d+)_class(\d+)\.png'
            match = re.search(pattern, image_file)
            id_biopsy, x_coord, y_coord, _ = map(int, match.groups())

            image = Image.open(image_file).convert('RGB')
            image = np.array(image)

            # Descartem les imatges que no tenen les dimensions esperades
            if image.shape != (50, 50, 3):
                continue

            images.append(image)
            labels.append(class_label)
            metadata.append((id_biopsy, x_coord, y_coord, class_label))

    images = np.array(images)
    labels = np.array(labels)
    metadata = np.array(metadata)

    return images, labels, metadata


# Funció que crea les transformacions a aplicar a les imatges, a partir de la mitjana i la desviació estàndard
def create_transforms(mean, std):
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return data_transforms
