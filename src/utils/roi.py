import cv2
import numpy as np

def extract_roi(image: np.ndarray,
                x: int, y: int, radius: int,
                scale: float = 1.3,
                output_size: int = 224,
                apply_mask: bool = True) -> np.ndarray | None:

    H, W = image.shape[:2]

    # Tamaño de la ventana
    half = int(radius * scale)

    x1, y1 = x - half, y - half
    x2, y2 = x + half, y + half

    # Padding necesario (siempre se calcula)
    pad_top    = max(0, -y1)
    pad_bottom = max(0, y2 - H)
    pad_left   = max(0, -x1)
    pad_right  = max(0, x2 - W)

    # Aplicar padding (aunque sea 0)
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

    # Máscara circular para aislar el tumor central
    if apply_mask:
        h, w = crop.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        center = (w // 2, h // 2)
        masked_radius = int(radius * scale * 0.9)  # ligeramente más pequeño

        cv2.circle(mask, center, masked_radius, 1, -1)

        if len(crop.shape) == 3:
            crop = crop * mask[:, :, None]
        else:
            crop = crop * mask

    # Redimensionado final
    crop_resized = cv2.resize(
        crop,
        (output_size, output_size),
        interpolation=cv2.INTER_LINEAR
    )

    return crop_resized