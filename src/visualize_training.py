# visualize_training.py
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_training(history_path="training_history.pkl", save_path="training_dashboard.png"):
    """
    Genera un dashboard con:
      - Loss de entrenamiento por epoch
      - AUC de validación por epoch
      - Mejor AUC marcado
      - Ventana móvil (suavizado) sobre la loss
    """
    with open(history_path, "rb") as f:
        history = pickle.load(f)

    losses  = np.array(history["train_loss"])
    aucs    = np.array(history["val_auc"])
    epochs  = np.arange(1, len(losses) + 1)

    best_epoch = int(np.argmax(aucs)) + 1
    best_auc   = aucs.max()

    # Suavizado de loss con ventana móvil de 5 epochs
    window  = 5
    smooth  = np.convolve(losses, np.ones(window)/window, mode="valid")
    smooth_x = np.arange(window, len(losses) + 1)

    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # Plot 1: Train Loss 
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#0f1117")
    ax1.plot(epochs, losses, color="#3a3f55", linewidth=1.2, alpha=0.6, label="Loss por epoch")
    ax1.plot(smooth_x, smooth, color="#7c83f5", linewidth=2.2, label=f"Media móvil ({window} ep.)")
    ax1.set_xlabel("Epoch", color="#aab0c6", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss", color="#aab0c6", fontsize=11)
    ax1.set_title("Train Loss", color="white", fontsize=13, fontweight="bold", pad=12)
    ax1.tick_params(colors="#aab0c6")
    ax1.spines[:].set_color("#2a2f45")
    ax1.legend(framealpha=0, labelcolor="#aab0c6", fontsize=9)
    ax1.grid(True, color="#2a2f45", linestyle="--", linewidth=0.6)

    #  Plot 2: Val AUC
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#0f1117")
    ax2.plot(epochs, aucs, color="#f57c83", linewidth=2.0, label="AUC validación")
    ax2.axhline(best_auc, color="#ffc107", linewidth=1.0, linestyle="--", alpha=0.7)
    ax2.scatter([best_epoch], [best_auc], color="#ffc107", s=80, zorder=5,
                label=f"Mejor AUC: {best_auc:.4f} (ep. {best_epoch})")
    ax2.set_xlabel("Epoch", color="#aab0c6", fontsize=11)
    ax2.set_ylabel("ROC-AUC", color="#aab0c6", fontsize=11)
    ax2.set_title("Validation AUC", color="white", fontsize=13, fontweight="bold", pad=12)
    ax2.set_ylim(max(0, aucs.min() - 0.05), min(1.02, aucs.max() + 0.05))
    ax2.tick_params(colors="#aab0c6")
    ax2.spines[:].set_color("#2a2f45")
    ax2.legend(framealpha=0, labelcolor="#aab0c6", fontsize=9)
    ax2.grid(True, color="#2a2f45", linestyle="--", linewidth=0.6)

    plt.suptitle("ResNet18 · Entrenamiento mamografías",
                 color="white", fontsize=15, fontweight="bold", y=1.02)

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Dashboard guardado en: {save_path}")
    plt.show()


if __name__ == "__main__":
    plot_training()