# main.py
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
from train import train_model

from data.dataset import MammographyROIDataset
from models.resnet import ResNet18Embedding
from utils.extract_embeddings import extract_embeddings


if __name__ == "__main__":

    BATCH_SIZE    = 32
    EPOCHS        = 50
    LR            = 1e-4
    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
    EMBEDDING_DIM = 64

    print(f"Usando: {DEVICE}")

    train_dir  = "dataset/isolated_tumors/train"
    val_dir    = "dataset/isolated_tumors/val"
    test_dir   = "dataset/isolated_tumors/test"   

    def load_files(directory):
        return [os.path.join(directory, f) for f in os.listdir(directory)]

    train_dataset = MammographyROIDataset(load_files(train_dir))
    val_dataset   = MammographyROIDataset(load_files(val_dir))
    test_dataset  = MammographyROIDataset(load_files(test_dir))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    # Entrenar
    resnet = ResNet18Embedding(embedding_dim=EMBEDDING_DIM, unfreeze_since="layer4")
    model_trained, history = train_model(
        resnet, train_loader, val_loader,
        epochs=EPOCHS, lr=LR, device=DEVICE,
    )

    # Guardamos historial para la vis
    with open("training_history.pkl", "wb") as f:
        pickle.dump(history, f)

    # Extraer y guardar los embedding
    embeddings, ids = extract_embeddings(model_trained, train_loader, device=DEVICE)

    with open("train_embeddings.pkl", "wb") as f:
        pickle.dump((ids, embeddings), f)

    print(f"Guardado: {len(ids)} imágenes | embeddings shape: {embeddings.shape}")