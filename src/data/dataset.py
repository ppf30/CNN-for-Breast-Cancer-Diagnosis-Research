# data/dataset.py
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path


class MammographyROIDataset(Dataset):
    def __init__(self, files: list, augment: bool = False):
        self.files = files
        self.augment = augment

        # transformaciones base
        base_transforms = [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

        # Si es modo entrenamiento, le inyectamos el Data Augmentation antes del ToTensor
        if self.augment:
            aug_transforms = [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=45), # Los tumores no tienen "arriba o abajo"
                transforms.ColorJitter(brightness=0.2, contrast=0.2), # Inmunidad a luz
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
            self.transform = transforms.Compose(aug_transforms)
        else:
            self.transform = transforms.Compose(base_transforms)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path  = self.files[idx]
        label = int(Path(path).stem[-1])
        img   = Image.open(path).convert("L") # fuerza escala de grises antes del pipeline
        img   = self.transform(img)
        return img, label, path