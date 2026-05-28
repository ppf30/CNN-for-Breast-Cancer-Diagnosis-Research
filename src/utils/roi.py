import cv2
import numpy as np


def extract_roi(image: np.ndarray, x: int, y: int, radius: int, scale: float = 1.5, output_size: int = 224, apply_mask: bool = True) -> np.ndarray | None:

    H, W = image.shape[:2]

    # Window size 
    half = int(radius * scale)

    x1, y1 = x - half, y - half
    x2, y2 = x + half, y + half

    # Padding necesario si el crop se sale de la imagen
    pad_top    = max(0, -y1)
    pad_bottom = max(0, y2 - H)
    pad_left   = max(0, -x1)
    pad_right  = max(0, x2 - W)

    image_padded = cv2.copyMakeBorder(
        image,
        pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )

    # Ajustar coordenadas tras padding
    x1 += pad_left; x2 += pad_left
    y1 += pad_top;  y2 += pad_top

    crop = image_padded[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    # Máscara cuadrada: se elimina el círculo, se mantiene el crop tal cual
    # (apply_mask se conserva por si se quiere enmascarar con otra lógica)

    # Redimensionado a 224x224 cuadrado (el crop ya es cuadrado por construcción)
    crop_resized = cv2.resize(
        crop,
        (output_size, output_size),
        interpolation=cv2.INTER_LINEAR
    )

    # Si el crop redimensionado es completamente negro, descartar
    if cv2.countNonZero(crop_resized.reshape(-1) if len(crop_resized.shape) == 2
                        else crop_resized.reshape(-1, crop_resized.shape[2]).max(axis=1)) == 0:
        return None

    return crop_resized