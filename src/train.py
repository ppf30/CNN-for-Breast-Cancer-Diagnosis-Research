# train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
from torch.amp import autocast, GradScaler


def train_model(model, train_loader, val_loader, epochs=50, lr=1e-4, device="cuda", save_path="best_resnet18_embedding.pth", patience=10):
    
    """
    Entrena ResNet18Embedding y devuelve el modelo con el mejor AUC de validación.
    También devuelve el historial de métricas para visualización.
    """
    model = model.to(device)

    all_train_labels = [int(Path(f).stem[-1]) for f in train_loader.dataset.files]
    n_ben = all_train_labels.count(0)
    n_mal = all_train_labels.count(1)
    class_weights = torch.tensor([1.0, n_ben / n_mal]).to(device)
    print(f"Class weights -> benigno: 1.0 | maligno: {n_ben/n_mal:.2f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    scaler = GradScaler()

    best_auc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_auc": []}   

    for epoch in range(epochs):

        # Train
        model.train()
        train_loss = 0.0
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast(device_type=device):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)

        # Val
        model.eval()
        all_probs, all_labels_val = [], []
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs  = imgs.to(device)

                with autocast(device_type=device):
                    outputs = model(imgs)

                probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels_val.extend(labels.numpy())

        auc = roc_auc_score(all_labels_val, all_probs)
        scheduler.step()

        history["train_loss"].append(avg_loss)
        history["val_auc"].append(auc)

        print(f"Epoch {epoch+1:03d}/{epochs} | "
              f"Loss: {avg_loss:.4f} | AUC val: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
            print(f"Nuevo mejor modelo guardado (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            print(f"Sin mejora. Paciencia: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"Early stopping activado en el epoch {epoch+1}")
                print(f"El modelo dejo de mejorar hace {patience} epochs")
                break

    # Cargar los mejores pesos
    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"Modelo cargado con mejor AUC: {best_auc:.4f}")

    return model, history   
