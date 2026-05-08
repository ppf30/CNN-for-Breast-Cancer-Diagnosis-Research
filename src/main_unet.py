# Libraries import
import sys
import argparse
from typing import Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

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
parser.add_argument("--in_ch",type=int, default=1, help='Input Channels')
parser.add_argument("--out_ch", type=int, default=1, help = "Output Classes")
parser.add_argument("--model", type=str, default="unet", choices = ["unet", "aunet"], help = "Model Selection")
parser.add_argument("--epochs", type=int, default=100, help = "Training Epochs")
parser.add_argument("--mode", type=str, choices = ["train", "test"], help = "Mode Selection")
parser.add_argument("--modelname", type=str, help = "The name to store the model")

# Parse args from Command Line Input
args = parser.parse_args()

# Define constant params
BATCH_SIZE = args.batch_size
IN_CHANNELS = args.in_ch
CLASSES = args.out_ch
EPOCHS = args.epochs
MODEL = args.model
MODE = args.mode
device = 'cuda' if torch.cuda.is_available() else "cpu"

# Paths to store models 
path_model = './models'


# Define the model 
model = UNet(in_channels = IN_CHANNELS, num_classes = CLASSES ) if MODEL =="unet" else AttentionUNet(in_channels = IN_CHANNELS, num_classes = CLASSES )
model = model.to(device)

# Retrieve data 
dataset = TIFFSegmentationDataset(
    img_dir=r"./dataset/TIFF Images",
    mask_dir=r"./dataset/ROI Masks",
    )

# Define train and test split sizes--> parte de ph
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Define the splits and store them only once
path_data = './data'
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
General.serialize_data(train_dataset, path_data,  name = 'train.pkl')
General.serialize_data(test_dataset, path_data, name = 'test.pkl')



# Retrieve train object
path_data = './data'
train_dataset = General.recover_data(path_data, name = 'train')







#TRAINING PROCESS --> Train unet-> store the model
if MODE == 'train':

    # Retrieve train object
    train_dataset = General.recover_data(path_data, name = 'train')

   # Generate a DataLoader object for train
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    # Retrieve val object
    val_dataset = General.recover_data(path_data, name = 'val')

    # Generate a DataLoader object for train
    val_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True
)

    # Check model's storing path
    General.ensure_dir(path_model)

    # Compute metametrics for weighting
    weight = 9

    # Call training with the model
    plot = True
    modelname = f'{MODEL}_{EPOCHS}'
    loss_values = train(modelname, val_dataloader, model, train_dataloader, EPOCHS, device, path_model, weight, plot)






#TESTING PROCESS --> Evaluate a given model 
if MODE=="test":

    # Retrieve test object
    test_dataset = General.recover_data(path_data, name = 'test')

    # Generate a DataLoader object for test
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=4,
        num_workers=4,
        shuffle=True,
        pin_memory=True
    )

    # Call simple inference
    metrics_dict, embedding = test(args.modelname, model, test_dataloader, device, path_model, sample = False)
    print(embedding)
