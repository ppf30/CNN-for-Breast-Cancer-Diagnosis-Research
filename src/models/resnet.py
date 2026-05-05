import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Embedding(nn.Module):
    def __init__(self, embedding_dim: int = 64, unfreeze_since: str = "layer4"):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Congelar todo
        for param in backbone.parameters():
            param.requires_grad = False

        # Descongelar desde unfreeze_since
        unfreeze = False
        for name, param in backbone.named_parameters():
            if unfreeze_since in name:
                unfreeze = True
            if unfreeze:
                param.requires_grad = True

        # Quedarse con todo menos el fc original
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        # → salida: [batch, 512, 1, 1]

        # Cabeza: 512 → 64 (embeddings) → 2 (clasificador)
        self.embedding = nn.Sequential(
            nn.Flatten(),               # [batch, 512]
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, embedding_dim),  # [batch, 64] ← los embeddings
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(embedding_dim, 2)     # [batch, 2]  ← solo para entrenar
        )

    def forward(self, x):
        features = self.backbone(x)       # [batch, 512, 1, 1]
        emb      = self.embedding(features)  # [batch, 64]
        logits   = self.classifier(emb)   # [batch, 2]
        return logits

    def get_embeddings(self, x):
        """Extraer los embeddings de 64d (sin clasificador)"""
        with torch.no_grad():
            features = self.backbone(x)
            emb      = self.embedding(features)
        return emb  # [batch, 64]