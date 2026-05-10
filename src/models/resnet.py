import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Embedding(nn.Module):
    def __init__(self, embedding_dim:int = 64, unfreeze_since:str = "layer4"):
        """
        ResNet-18 based model for binary classification with an embedding head.
        This module dapts a pre-trained ResNEt18 nackbone to produce low-dimensional
        embeddings. It supports partial final-tuning by unfreezing layers from a 
        specified point onwards and includes a custom MLP head to project features into
        a embedding space, followed bya linear classifier.
        Args:
        - embedding_dim (int): dimensionality of the embedding space (default: 64)
        - unfreeze_since (str): layer name from which to start unfreezing parameters for 
        fine-tuning (default: "layer4")
        """
        super().__init__()

        # Load pre-trained ResNet-18
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze all parameters initially
        for param in backbone.parameters():
            param.requires_grad = False

        # Unfreeze parameters starting from the specified layer
        unfreeze = False
        for name, param in backbone.named_parameters():
            if unfreeze_since in name:
                unfreeze = True
            if unfreeze:
                param.requires_grad = True

        # Remove the original classification head (fully connected layer)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        # Output: [batch, 512, 1, 1]

        # Head: 512 -> 64 (embeddings) -> 2 (clasificador)
        self.embedding = nn.Sequential(
            nn.Flatten(),               # Results: [batch, 512]
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),                 # estabiliza entrenamiento con LR alto
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, embedding_dim),  # Results: [batch, 64]
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(embedding_dim, 2)     # Results: [batch, 2]
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Foward pass for training
        Args:
        - x (Tensor): input image batch of shape [batch, 3, H, W]
        Returns:
        - logits (Tensor): logits for binary classification [batch, 2]
        """
        features = self.backbone(x)       # [batch, 512, 1, 1]
        emb      = self.embedding(features)  # [batch, 64]
        logits   = self.classifier(emb)   # [batch, 2]
        return logits

    def get_embeddings(self, x:torch.Tensor) -> torch.Tensor:
        """
        Extracts the embeddings without the classifier

        Args:
        - x (Tensor): input image batch of shape [batch, 3, H, W]
        Returns:
        - emb (Tensor): projected embedding vectors [batch, 64]
        """
        with torch.no_grad():
            features = self.backbone(x)
            emb      = self.embedding(features)
        return emb  # [batch, 64]