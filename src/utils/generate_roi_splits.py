from roi import extract_roi
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import os
import tifffile as tif

'''Script para generar los crops de las regiones de interés (tumores) a partir de las coordenadas y radios del excel.
El script busca cada imagen en las carpetas train, test y val, extrae el crop con la función extract_roi y lo guarda en una nueva carpeta "isolated_tumors" 
manteniendo la estructura de subcarpetas (train, test, val) y añadiendo el label al nombre del archivo.'''

xlsx_path = "../../dataset/Metadata.xlsx"

split_dir = Path("../../dataset/copia")

df = pd.read_excel(xlsx_path)

# Quedarse con columnas concretas
df = df.iloc[:, [0, 4, 5, 6, 7]]

# Eliminar NaN
df = df.dropna()

# Primera fila como nombres de columnas
df.columns = df.iloc[0]

# Eliminar primera fila
df = df[1:]

# Resetear índices
df = df.reset_index(drop=True)

df = df[df["CLASS"] != 'N']

# Cambiar B -> 0 y M -> 1
df["CLASS"] = df["CLASS"].replace({"B": 0,"M": 1})

df = df.drop(389)  #radio invalido

for idx, row in df.iterrows():

    image_id = str(row["IMG_REF"]).strip()
    x = row["X_COORD"]
    y = row["Y_COORD"]
    label = row["CLASS"]    
    radio = row["RADIUS"]
    image_path = None

    # Buscar en train, test y val
    for subfolder in ["train", "test", "val"]:

        folder = os.path.join(split_dir, subfolder)
        folder = Path(folder)

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
    
    tif.imwrite(f"../../dataset/isolated_tumors/{save_sub}/{image_id}_{idx}_{label}.tif", crop_roi)
    
