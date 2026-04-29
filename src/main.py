# Libraries import
import sys
import argparse
from typing import Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Files import
from train import train
from test import test
from general_utils import General
from unet.models import UNet, AttentionUNet
from dataset import TIFFSegmentationDataset

# Define the parser
parser = argparse.ArgumentParser(description="Entrenamiento de UNet para segmentación")

# Add arguments to the parser
parser.add_argument("--batch_size", type=int, default=64, help = "Batch Size")
parser.add_argument("--lr", type=float, default=0.001, help = "Learning Rate")
parser.add_argument("--in_ch",type=int, default=64, help='Input Channels')
parser.add_argument("--out_cl", type=int, default=6, help = "Output Classes")
parser.add_argument("--model", type=str, default="unet", choices = ["unet", "aunet"], help = "Model Selection")
parser.add_argument("--epochs", type=int, default=100, help = "Training Epochs")
parser.add_argument("--mode", type=str, choices = ["train", "features", "test"], help = "Mode Selection")

# Parse args from Command Line Input
args = parser.parse_args()

# Define constant params
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
IN_CHANNELS = args.in_ch
CLASSES = args.out_cl
EPOCHS = args.epochs
device = 'cuda' if torch.cuda.is_available() else "cpu"

# Paths to store models and generated data
path_model = './models'
path_data = './data_cnn/'

# Define the model 
model = UNet(in_channels = args.in_ch, num_classes = args.out_cl ) if args.model =="unet" else AttentionUNet(in_channels = args.in_ch, num_classes = args.out_cl )





#TRAINING PROCESS --> Train unet-> store the model
if args.mode == 'train':

    # Retrieve data 
    dataset = TIFFSegmentationDataset(
        img_dir=r"dataset\TIFF Images\TIFF Images",
        mask_dir=r"dataset\ROI Masks\ROI Masks",
        patch_size=1024)
    
    # Generate a DataLoader object for train
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Check model's storing path
    General.ensure_dir(path_model)


    # Call training with the model
    loss_values = train(model, train_dataloader, args.epochs, device, path_model)


#TESTING PROCESS --> Evaluate a given model 
if args.mode=="test":

    # Retrieve data 
    dataset = TIFFSegmentationDataset(
        img_dir=r"dataset\TIFF Images\TIFF Images",
        mask_dir=r"dataset\ROI Masks\ROI Masks",
        patch_size=1024)
    
    # Generate a DataLoader object for test
    test_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Call simple inference
    prediction = test(model, test_dataloader, device, path_model, needed = False)

    # Call metrics generation


# FEATURE GENERATION --> Generate data for the application of the later CNN
if args.mode=="features": 

    # Retrieve data 
    dataset = TIFFSegmentationDataset(
        img_dir=r"dataset\TIFF Images\TIFF Images",
        mask_dir=r"dataset\ROI Masks\ROI Masks",
        patch_size=1024)
    
    # Generate a DataLoader object for test
    test_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    new_dataset = test(model, test_dataloader, device, path_data, needed = True)

    # Store data -> if needed is True: that is, we need data to feed the later CNN
    # Check data storing path
    General.ensure_dir(path_data)
    # Serialize and store data
    General.serialize_data(new_dataset, path_data)


