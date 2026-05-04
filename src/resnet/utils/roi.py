import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

def extract_roi(image: np.ndarray,
                x: int, y: int, radius: int,
                padding_factor: float = 2.0,
                output_size: int = 224) -> np.ndarray | None:
    """
    Extrae un ROI cuadrado centrado en (x, y) con margen = radius * padding_factor.
    Devuelve un recorte redimensionado a output_size x output_size.
    Maneja bordes con padding reflectivo para no perder información.
    """
    H, W = image.shape[:2]
    half = int(radius * padding_factor)

    # Coordenadas del recorte (pueden salirse de la imagen)
    x1, y1 = x - half, y - half
    x2, y2 = x + half, y + half

    # Padding si el recorte se sale de los límites
    pad_top    = max(0, -y1)
    pad_bottom = max(0, y2 - H)
    pad_left   = max(0, -x1)
    pad_right  = max(0, x2 - W)

    if any([pad_top, pad_bottom, pad_left, pad_right]):
        image = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REFLECT_101
        )
        # Actualizar coordenadas tras el padding
        x1 += pad_left;  x2 += pad_left
        y1 += pad_top;   y2 += pad_top

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    # Resize consistente para la ResNet
    crop_resized = cv2.resize(crop, (output_size, output_size),
                              interpolation=cv2.INTER_LINEAR)
    return crop_resized


def build_roi_dataset(annotations_csv: str,
                      images_dir: str,
                      output_dir: str,
                      padding_factor: float = 2.0,
                      output_size: int = 224):
    """
    Lee el CSV de anotaciones, extrae cada ROI y lo guarda
    en output_dir/benigno/ o output_dir/maligno/
    """
    df = pd.read_csv(annotations_csv)
    label_map = {0: "benigno", 1: "maligno"}

    for label_name in label_map.values():
        Path(f"{output_dir}/{label_name}").mkdir(parents=True, exist_ok=True)

    for idx, row in df.iterrows():
        img_path = os.path.join(images_dir, row["image_id"])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"No se pudo cargar: {img_path}")
            continue

        # CLAHE para mejorar contraste en mamografías
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

        roi = extract_roi(image, int(row["x"]), int(row["y"]),
                          int(row["radius"]), padding_factor, output_size)
        if roi is None:
            continue

        label_name = label_map[int(row["label"])]
        save_path = f"{output_dir}/{label_name}/{row['image_id']}_tumor{idx}.png"
        cv2.imwrite(save_path, roi)

    print("Dataset de ROIs generado.")