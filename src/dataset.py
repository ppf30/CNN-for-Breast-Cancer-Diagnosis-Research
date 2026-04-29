import os
import torch
import tifffile as tif
import random

class TIFFSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, patch_size=1024):

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size

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

        image = tif.imread(img_path)
        mask = tif.imread(mask_path)

        # convertir imagen a 1 canal quedándonos con R
        if image.ndim == 3:
            image = image[..., 0]

        # asegurar máscara 2D
        if mask.ndim == 3:
            mask = mask[..., 0]

        # extraer patch aleatorio para optimizar con imágenes grandes
        H, W = image.shape
        ps = self.patch_size

        i = random.randint(0, H - ps)
        j = random.randint(0, W - ps)

        image = image[i:i+ps, j:j+ps]
        mask = mask[i:i+ps, j:j+ps]

        image = torch.tensor(image).unsqueeze(0).float() / 255.0

        mask = torch.tensor(mask)
        mask = (mask > 0).float().unsqueeze(0)

        return image, mask
    

