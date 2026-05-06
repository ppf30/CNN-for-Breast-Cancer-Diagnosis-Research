from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path


class MammographyROIDataset(Dataset):
    def __init__(self, files: list):
        self.files = files
        #resnet espera 3 canales en lugar de 1
        self.transform_base = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = int(Path(path).stem[-1])  #obtenemos la etiqueta a partir del nombre del archivo
        img = Image.open(path)  #abrimos con PIL porque transforms espera una Image, no un numpy array
        img = self.transform_base(img)

        return img, label, path
