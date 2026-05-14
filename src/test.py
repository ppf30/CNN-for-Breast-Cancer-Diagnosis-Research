import torch
import torch.nn.functional as F
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from models.resnet import ResNet18Embedding
from sklearn.metrics import roc_curve
import os
import numpy as np
import matplotlib.pyplot as plt
from data.dataset import MammographyROIDataset
from torch.utils.data import DataLoader
from utils.extract_embeddings import extract_embeddings
from utils.tsne import plot_tsne

def evaluate(trained_model, test_loader: list, device: str = "cuda"):

    
    trained_model.eval()

    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for imgs, labels, _ in test_loader:
            imgs   = imgs.to(device)
            logits = trained_model(imgs)
            model.get_embeddings(imgs)
            probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    # ── Métricas ──────────────────────────────
    auc = roc_auc_score(all_labels, all_probs)

    print("\n" + "═"*50)
    print(f"  AUC-ROC:  {auc:.4f}")
    print("═"*50)
    print(classification_report(all_labels, all_preds,
                                target_names=["Benigno", "Maligno"],
                                digits=4))

    # ── Plots ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Benigno", "Maligno"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Matriz de Confusión")

    # 2. Curva ROC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    axes[1].plot(fpr, tpr, color="steelblue", lw=2,
                 label=f"AUC = {auc:.4f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("Curva ROC")
    axes[1].legend(loc="lower right")

    # 3. Distribución de probabilidades predichas
    axes[2].hist(all_probs[all_labels == 0], bins=20, alpha=0.6,
                 color="steelblue", label="Benigno")
    axes[2].hist(all_probs[all_labels == 1], bins=20, alpha=0.6,
                 color="tomato", label="Maligno")
    axes[2].axvline(0.5, color="black", linestyle="--", lw=1, label="Umbral 0.5")
    axes[2].set_xlabel("Probabilidad predicha (maligno)")
    axes[2].set_ylabel("Frecuencia")
    axes[2].set_title("Distribución de predicciones")
    axes[2].legend()

    plt.suptitle("Evaluación ResNet18 — Clasificación Tumor Mamario", fontsize=13)
    plt.tight_layout()
    plt.savefig("test_results.png", dpi=150)
    print("\nGráficas guardadas en test_results.png")

    return all_preds, all_probs, all_labels



if __name__ == "__main__":

    WEIGHTS_PATH = "best_resnet18_embedding.pth"
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
    test_dir   = "dataset/isolated_tumors/test"   
    BATCH_SIZE = 32

    model = ResNet18Embedding(embedding_dim=64)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model = model.to(DEVICE)

    print(f"Usando: {DEVICE}")
    def load_files(directory):
        return [os.path.join(directory, f) for f in os.listdir(directory)]

    test_dataset  = MammographyROIDataset(load_files(test_dir))
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)
    
    preds, probs, labels = evaluate(
        trained_model=model,
        test_loader=test_loader,
        device=DEVICE
    )

    tuples = extract_embeddings(model=model,dataloader=test_loader,device=DEVICE,
                                    test=True)

    plot_tsne(tuples)