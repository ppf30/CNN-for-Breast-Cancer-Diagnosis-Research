# Libraries import
import sys
import argparse
from typing import Union
import torch
import torch.nn as nn
import torch.optim as optim
from general_utils import General

# Files import
from train import train
from features import generate_outputs
from unet.models import UNet, AttentionUNet



# Define the parser
parser = argparse.ArgumentParser(description="Entrenamiento de UNet para segmentación")

# Add arguments to the parser
parser.add_argument("--batch_size", type=int, default=64, help = "Batch Size")
parser.add_argument("--lr", type=float, default=0.001, help = "Learning Rate")
parser.add_argument("--in_ch",type=int, default=64, help='Input Channels')
parser.add_argument("--out_cl", type=int, default=6, help = "Output Classes")
parser.add_argument("--model", type=str, default="unet", choices = ["unet", "aunet"], help = "Model Selection")
parser.add_argument("--epochs", type=int, default=100, help = "Training Epochs")
parser.add_argument("--mode", type=str, choices = ["train", "features", "infer"], help = "Mode Selection")

# Parse args from Command Line Input
args = parser.parse_args()

# Define constant params
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
IN_CHANNELS = args.in_ch
CLASSES = args.out_cl
EPOCHS = args.epochs
device = 'cuda' if torch.cuda.is_available() else "cpu"
path = './models'




#TRAINING PROCESS --> Train unet-> store the model -> get the data
if args.mode == 'train':

    # Check model's storing path
    General.ensure_model_dir(path)
    # Create the model
    model = UNet(in_channels = args.in_ch, out_channels = args.out_cl ) if args.model =="unet" else AttentionUNet(in_channels = args.in_ch, out_channels = args.out_cl )
    # Call training
    train(args.model, train_dataloader, args.batch_size, args.epochs, device, path)


# Train the CNN using the other dataset
if args.mode=="features":
    model = UNet() if args.model =="unet" else AttentionUNet()
    generate_outputs(model, train_dataloader, device, path)


# TESTING PROCESS
if args.mode == 'inference':
    pass
