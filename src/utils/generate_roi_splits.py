from roi import extract_roi
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import os
import tifffile as tif

# 1. Definir la raíz del proyecto dinámicamente para evitar problemas de rutas
# Asumimos que el script está en: .../CNN-for-Breast-Cancer-Diagnosis-Research/src/utils/generate_roi_splits.py
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent # Sube de utils -> src -> raíz del proyecto

# 2. Definir todas las rutas basadas en la raíz del proyecto
dataset_dir = project_root / "dataset"
xlsx_path = dataset_dir / "Metadata.xlsx"
split_dir = dataset_dir / "split_masks"
output_base_dir = dataset_dir / "isolated_tumors"

df = pd.read_excel(xlsx_path)

df.columns=['IMG_REF','delete1', 'delete2', 'delete3', 'CLASS', 'X_COORD', 'Y_COORD', 'RADIUS']
label_map = {"B": 0, "M":1}
df=df.drop(columns=['delete1', 'delete2', 'delete3'])
df['CLASS']=df['CLASS'].map(label_map)
df = df.dropna()

for idx, row in df.iterrows():
    image_id = str(row["IMG_REF"]).strip()
    x = int(row["X_COORD"])
    y = int(row["Y_COORD"])
    label = int(row["CLASS"])
    try:    
        radio = int(row["RADIUS"])
    except:
        continue

    image_path = None

    # Buscar en train, test y val
    for subfolder in ["train", "test", "val"]:
        folder = split_dir / subfolder # Usando Path directamente

        matches = list(folder.rglob(f"{image_id}.tif"))

        if matches:
            image_path = matches[0]
            save_sub = subfolder
            break
    
    if image_path is None:
        print(f"No encontrada: {image_id}")
        continue

    image = tif.imread(image_path)
    crop_roi = extract_roi(image, x, y, radio)
    
    if crop_roi is None:
        print(f"Crop vacío para {image_id} en {subfolder}")
        continue
    
    # 3. CREAR LA CARPETA SI NO EXISTE ANTES DE GUARDAR
    target_dir = output_base_dir / save_sub
    target_dir.mkdir(parents=True, exist_ok=True) # Crea "isolated_tumors/train" (o test/val) si no existe
    
    # 4. Guardar el archivo usando la ruta construida
    target_file = target_dir / f"{image_id}_{idx}_{label}.tif"
    tif.imwrite(str(target_file), crop_roi)