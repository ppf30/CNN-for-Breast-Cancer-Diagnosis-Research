import os
import torch
import tifffile as tif
import random
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

random.seed(42)

class TIFFSegmentationDataset(torch.utils.data.Dataset):
    """
    Dataset personalizado para tareas de segmentación de imágenes médicas en formato TIFF.

    Este dataset carga pares imagen-máscara desde dos directorios y genera muestras
    en forma de parches aleatorios de tamaño fijo. Está diseñado para trabajar con
    modelos de segmentación tipo U-Net.

    Funcionamiento general:
    -----------------------
    - Busca imágenes TIFF en `img_dir`.
    - Filtra aquellas que no tienen máscara correspondiente en `mask_dir`.
    - Para cada muestra:
        - Carga imagen y máscara desde disco.
        - Convierte ambas a formato 2D si tienen múltiples canales.
        - Extrae un parche aleatorio de tamaño `patch_size x patch_size`.
        - Normaliza la imagen a rango [0, 1].

    Parámetros:
    -----------
    img_dir : str
        Ruta al directorio que contiene las imágenes TIFF.
    mask_dir : str
        Ruta al directorio que contiene las máscaras TIFF.
        Se asume que cada máscara tiene el mismo nombre que su imagen.
    patch_size : int, opcional (default=1024)
        Tamaño del parche cuadrado que se extraerá de cada imagen.

    Returns:
    -----------
    Tensores con forma [Batch, 1, H, W]
    """

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

            # si no existe la máscara (no hay tumor) → se descarta
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

        # # extraer patch aleatorio para optimizar con imágenes grandes
        # H, W = image.shape
        # ps = self.patch_size

        # # nos aseguramos de que el patch es menor que la imagen
        # if H < ps or W < ps:
        #     raise ValueError("Patch size mayor que la imagen")
        
        # i = random.randint(0, H - ps)
        # j = random.randint(0, W - ps)

        # image = image[i:i+ps, j:j+ps]
        # mask = mask[i:i+ps, j:j+ps]

        image = torch.tensor(image).unsqueeze(0).float() / 255.0

        mask = torch.tensor(mask)
        mask = (mask > 0).float().unsqueeze(0)

        return image, mask  #X,y
    
    def _augment(self, image, mask):
        # flips
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # rotación (con cuidado en máscara)
        if random.random() > 0.5:
            angle = random.uniform(-90, 90)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

        return image, mask

 
        
    
    

