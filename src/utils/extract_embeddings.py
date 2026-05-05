import numpy as np
import torch
from collections import defaultdict
from pathlib import Path

def extract_embeddings(model, dataloader, device="cuda", mode='mean'):
    """
    Extrae embeddings de un modelo a partir de ROIs y los agrega a nivel de imagen.

    Para cada batch, obtiene embeddings por ROI mediante `model.get_embeddings`
    y los agrupa usando el identificador de imagen (IMGxxx) extraído del path.
    Después, combina los embeddings de cada imagen (por defecto con la media).

    Args:
        model (nn.Module): Modelo con método `get_embeddings`.
        dataloader (DataLoader): Debe devolver (img, label, path).
        device (str): Dispositivo de cómputo ("cuda" o "cpu").
        weights_path (str): Ruta a los pesos entrenados.
        mode (str): Método de agregación ("mean" o "max").

    Returns:
        embeddings (np.ndarray): Array [N_imagenes, D].
        ids (np.ndarray): Array [N_imagenes].
    """
    
    model = model.to(device)
    model.eval()

    groups = defaultdict(list)
    label_groups = defaultdict(list)

    with torch.no_grad():
        for imgs, labels, paths in dataloader:
            imgs = imgs.to(device)
            emb  = model.get_embeddings(imgs)  # [batch, D]

            emb = emb.cpu().numpy()
            labels = labels.cpu().numpy()

            for i, path in enumerate(paths):
                img_id = Path(path).stem.split('_')[0]

                groups[img_id].append(emb[i])
                label_groups[img_id].append(labels[i])

    agg_embeddings = []
    agg_ids = [] #para trazabilidad

    for img_id in sorted(groups.keys()):

        embs = np.stack(groups[img_id])  # [k, D]

        if mode == "mean":
            agg = embs.mean(axis=0)
        elif mode == "max":
            agg = embs.max(axis=0)
        elif mode == 'sum':
            agg = embs.sum(axis=0)
        else:
            raise ValueError("Modo no soportado")

        agg_embeddings.append(agg)
        agg_ids.append(img_id)

    embeddings = np.array(agg_embeddings)
    ids  = np.array(agg_ids)

    print(f"Embeddings por imagen: {embeddings.shape}")
    
    return embeddings, ids