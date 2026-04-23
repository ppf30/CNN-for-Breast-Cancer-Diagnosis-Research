import sys
import argparse
from typing import Union
from .main import UNet, AttentionUNet
import torch.nn as nn
import torch.optim as optim


# Define the parser
parser = argparse.ArgumentParser(description="Entrenamiento de UNet para segmentación")

# Add arguments to the parser
parser.add_argument("--batch_size", type=int, default=64, help = "Batch Size")
parser.add_argument("--lr", type=float, default=0.001, help = "Learning Rate")
parser.add_argument("--in_ch",type=int, default=64, help='Input Channels')
parser.add_argument("--out_cl", type=int, default=6, help = "Output Classes")
parser.add_argument("--model", type=int, default=6, help = "Model(attention_unet:'att'--unet:'base')")
parser.add_argument("--epochs", type=int, default=6, help = "Training Epochs")

# Parse args from Command Line Input
args = parser.parse_args()


# Define constant params
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
IN_CHANNELS = args.in_ch
CLASSES = args.out_cl
EPOCHS = args.epochs