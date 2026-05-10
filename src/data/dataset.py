# data/dataset.py
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path


class MammographyROIDataset(Dataset):
    def __init__(self, files: list):
        self.files = files
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),           # garantiza tamaño mínimo ResNet
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path  = self.files[idx]
        label = int(Path(path).stem[-1])
        img   = Image.open(path).convert("L")   # fuerza escala de grises antes del pipeline
        img   = self.transform(img)
        return img, label, path