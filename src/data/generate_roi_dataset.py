import pandas as pd
import os
import tifffile as tif
from utils.roi import extract_roi
import numpy as np



def generate_roi_dataset():

    '''
    This function is intended to be run only once.
    It generates pairs of a single tumor ROI mask and its label (1 = malign | 0 = benign) from the ground truth masks.
    '''

    #paths
    path_metadata= '../../dataset/Metadata.xlsx'
    masks_dir='../../dataset/ROI Masks/ROI Masks'

    #preprocess data
    df = pd.read_excel(path_metadata)
    label_map = {"B": 0, "M":1}
    df.columns=['image_id','delete1', 'delete2', 'delete3', 'class', 'x', 'y', 'radius']
    df=df.drop(columns=['delete1', 'delete2', 'delete3'])
    df['class']=df['class'].map(label_map)
    df = df.dropna()

    for idx, row in df.iterrows():
        if str(row["radius"]).strip() in ("-", "", "nan") or \
        str(row["x"]).strip()      in ("-", "", "nan") or \
        str(row["y"]).strip()      in ("-", "", "nan"):
            print(f"Fila {idx} saltada (valores inválidos): {row['image_id']}")
            continue

        img_path = os.path.join(masks_dir, row["image_id"].strip() + '.tif')
        try:
            mask = tif.imread(img_path)

        except FileNotFoundError: #algunos tumores del excel no aparecen en las mascaras del ground truth
            continue

        if mask is None:
            print(f"No se pudo cargar: {img_path}")
            continue

        if mask.ndim == 3:
            mask = mask[..., 0]

        roi = extract_roi(mask, int(row["x"]), int(row["y"]),
                            int(row["radius"]))
        
        if not np.any(roi): #evita imagenes completamente negras
            continue

        save_path_roi = f"../../dataset/classification/{row['image_id'].strip()}_tumor{idx}_label{int(row['class'])}.tif"
        if os.path.exists(save_path_roi):
            os.remove(save_path_roi)

        tif.imwrite(save_path_roi, roi)

    print("Dataset de ROIs generado.")