# utils/extract_embeddings.py
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path


def extract_embeddings(model, dataloader, device="cuda", mode="mean"):
    """
    Extrae embeddings por ROI y los agrega a nivel de imagen completa.

    Args:
        model      : nn.Module con método get_embeddings(imgs) → [B, D].
        dataloader : devuelve (imgs, labels, paths).
        device     : "cuda" | "cpu".
        mode       : "mean" | "max" | "sum".

    Returns:
        embeddings : np.ndarray [N_imagenes, D]
        ids        : np.ndarray [N_imagenes]
    """
    model = model.to(device)
    model.eval()

    groups       = defaultdict(list)
    label_groups = defaultdict(list)

    with torch.no_grad():                          # un solo contexto, fuera del método
        for imgs, labels, paths in dataloader:
            imgs      = imgs.to(device)
            emb       = model.get_embeddings(imgs) # [B, D]
            emb_np    = emb.cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i, path in enumerate(paths):
                img_id = Path(path).stem.split("_")[0][3:]   # "IMG001_..." → "001"
                groups[img_id].append(emb_np[i])
                label_groups[img_id].append(int(labels_np[i]))

    agg_embeddings, agg_ids = [], []

    for img_id in sorted(groups.keys()):
        embs = np.stack(groups[img_id])   # [k, D]

        if mode == "mean":
            agg = embs.mean(axis=0)
        elif mode == "max":
            agg = embs.max(axis=0)
        elif mode == "sum":
            agg = embs.sum(axis=0)
        else:
            raise ValueError(f"Modo no soportado: '{mode}'. Usa: mean | max | sum")

        agg_embeddings.append(agg)
        agg_ids.append(img_id)

    embeddings = np.array(agg_embeddings)  
    ids        = np.array(agg_ids)

    print(f"Embeddings por imagen: {embeddings.shape}")
    return embeddings, ids