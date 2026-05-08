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
    
    def __init__(self, img_dir, mask_dir, patch_size=1024):

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        # Patch size is not really used 
        self.patch_size = patch_size
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

        return image_res, mask_res