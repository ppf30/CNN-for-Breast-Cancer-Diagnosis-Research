import os
import torch
import tifffile as tif
import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.transforms import v2

class TIFFSegmentationDataset(torch.utils.data.Dataset):
    
    def __init__(self, img_dir, mask_dir, augmentate:bool = False):

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        # Define if augmentation is seeked
        self.augmentate = augmentate
        # Define resize
        self.resize = v2.Resize((256, 256))

        all_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".tif", ".tiff"))
        ])

        # filtrar imágenes sin tumor
        self.files = []

        for f in all_files:
            mask_path = os.path.join(mask_dir, f)

            # si no existe la máscara → se descarta
            if not os.path.exists(mask_path):
                continue

            self.files.append(f)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, name)

        # Cargar y copiar para evitar problemas de memoria
        image = tif.imread(img_path).copy()
        mask = tif.imread(mask_path).copy()

        if image.ndim == 3: image = image[..., 0]
        if mask.ndim == 3: mask = mask[..., 0]

        # Convertir a Tensores [1, H, W]
        image_t = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        mask_t = (mask_t > 0).float()

        # Scaling the pictures
        image_res = self.resize(image_t)
        mask_res = self.resize(mask_t)
        
        # Turn masks into binary files
        mask_res = (mask_res > 0.5).float()


        if self.augmentate:
            # Horizontal flip: 
            if random.random() < 0.5:
                image_res = TF.hflip(image_res)
                mask_res = TF.hflip(mask_res)

            # Vertical flip
            if random.random() < 0.5:
                image_res = TF.vflip(image_res)
                mask_res = TF.vflip(mask_res)

            # Random Rotation
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image_res = TF.rotate(image_res, angle)
                mask_res = TF.rotate(mask_res, angle)

        return image_res, mask_res