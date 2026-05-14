from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def plot_tsne(embedding_tuples: list):
    """
    Recibe la lista de tuplas (embedding_64d, label) que devuelve
    extract_embeddings con test=True y pinta el t-SNE.
    """
    embeddings = np.array([t[0] for t in embedding_tuples])  # [N, 64]
    labels     = np.array([t[1] for t in embedding_tuples])  # [N]

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(embeddings)  # [N, 2]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {0: "steelblue", 1: "tomato"}
    nombres = {0: "Benigno", 1: "Maligno"}

    for label in [0, 1]:
        mask = labels == label
        ax.scatter(reduced[mask, 0], reduced[mask, 1],
                   c=colors[label], label=nombres[label],
                   alpha=0.7, edgecolors="white", linewidths=0.4, s=60)

    ax.set_title("t-SNE de embeddings 64d — ResNet18")
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.legend()
    plt.tight_layout()
    plt.savefig("tsne_embeddings.png", dpi=150)
    plt.show()
    print("t-SNE guardado en tsne_embeddings.png")
