from utils.make_splits import make_splits
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import torch
from sklearn.metrics import roc_auc_score
from data.dataset import MammographyROIDataset
from models.resnet import ResNet18Embedding
from utils.extract_embeddings import extract_embeddings
import numpy as np
from data.generate_roi_dataset import generate_roi_dataset
import os


def train_model(model, train_loader, val_loader,
                epochs=50, lr=1e-4, device="cuda",
                save_path="best_resnet18_embedding.pth"):
    
    "Entrena ResNet y lo devuelve para utilizarlo en inferencia y extraer embeddings"
    model = model.to(device)

    all_train_labels = [int(Path(f).stem[-1]) for f in train_loader.dataset.files]
    n_ben = all_train_labels.count(0)
    n_mal = all_train_labels.count(1)
    class_weights = torch.tensor([1.0, n_ben / n_mal]).to(device)
    print(f"Class weights → benigno: 1.0 | maligno: {n_ben/n_mal:.2f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)  #paliar el desbalanceo
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_auc = 0.0

    for epoch in range(epochs):

        # Train
        model.train()
        train_loss = 0.0
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validación
        model.eval()
        all_probs, all_labels_val = [], []
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs = imgs.to(device)
                probs = F.softmax(model(imgs), dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels_val.extend(labels.numpy())

        auc = roc_auc_score(all_labels_val, all_probs)
        scheduler.step()

        print(f"Epoch {epoch+1:03d}/{epochs} | "
              f"Loss: {train_loss/len(train_loader):.4f} | "
              f"AUC val: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), save_path)
            
    print(f" Modelo guardado (mejor AUC: {best_auc:.4f})")

    return model


if __name__ == "__main__":

    BATCH_SIZE = 32
    EPOCHS     = 50
    LR         = 1e-4
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
    EMBEDDING_DIM = 64
    
    print(f"Usando: {DEVICE}")

    train_dir = 'dataset/isolated_tumors/train'
    train_files = [
        os.path.join(train_dir, file)
        for file in os.listdir(train_dir)
    ]
    val_dir = 'dataset/isolated_tumors/val'
    val_files = [
        os.path.join(val_dir, file)
        for file in os.listdir(val_dir)
    ]
    test_dir = 'dataset/isolated_tumors/val'
    test_files = [
        os.path.join(test_dir, file)
        for file in os.listdir(test_dir)
    ]
    
    train_dataset = MammographyROIDataset(train_files)
    val_dataset   = MammographyROIDataset(val_files)
    test_dataset  = MammographyROIDataset(test_files)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    # Entrenar
    resnet = ResNet18Embedding(embedding_dim=EMBEDDING_DIM, unfreeze_since="layer4")
    model_trained = train_model(resnet, train_loader, val_loader,
                        epochs=EPOCHS, lr=LR, device=DEVICE)
    
    # Extraer embeddings del test set
    embeddings, ids = extract_embeddings(model_trained, test_loader, device=DEVICE)

    # Guardar para uso posterior
    np.save("embeddings.npy", embeddings)
    np.save("embeddings_ids.npy", ids)
    print("Embeddings guardados en embeddings.npy y embedding_ids.npy")