# Libraries import
import sys
import argparse
from typing import Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os

# Files import
from model_related.train import train
from model_related.test import test
from model_related.embeddings import embeddings

from general_utils import General, splits_masks, splits_tiff
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
parser.add_argument("--mode", type=str, choices = ["train", "test", "emb"], help = "Mode Selection")
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
BASE_DATA = './dataset'
IMAGES_DIR_MASKS = os.path.join(BASE_DATA,"ROI Masks" )
IMAGES_DIR_TIFF = os.path.join(BASE_DATA, 'TIFF Images')

# Get splits by name
path_masks = './data/masks'
train_ids,val_ids,test_ids, already_there = splits_masks(IMAGES_DIR_MASKS)
print(min(train_ids), max(train_ids))
print(min(val_ids), max(val_ids))
print(min(test_ids), max(test_ids))

General.serialize_data(General.create_id(train_ids), path_masks,  name = 'train_ids.pkl')
General.serialize_data(General.create_id(val_ids), path_masks, name = 'val_ids.pkl')
General.serialize_data(General.create_id(test_ids), path_masks, name = 'test_ids.pkl')


# Align names and  folders
if not already_there:
    splits_tiff(BASE_DATA, IMAGES_DIR_TIFF, train_ids, val_ids, test_ids)


# Create the different datasets
train_dataset = TIFFSegmentationDataset(
    img_dir=r"./dataset/split_tiff/train",
    mask_dir=r"./dataset/split_masks/train",
    )

val_dataset = TIFFSegmentationDataset(
    img_dir=r"./dataset/split_tiff/val",
    mask_dir=r"./dataset/split_masks/val",
    )


test_dataset = TIFFSegmentationDataset(
    img_dir=r"./dataset/split_tiff/test",
    mask_dir=r"./dataset/split_masks/test",
    )

# Define the splits and store them only once
path_data = './data/splits'
General.serialize_data(train_dataset, path_data,  name = 'train.pkl')
General.serialize_data(test_dataset, path_data, name = 'test.pkl')
General.serialize_data(val_dataset, path_data, name = 'val.pkl')



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
    metrics_dict = test(args.modelname, model, test_dataloader, device, path_model, sample = False)


#EMBEDDING PROCESS --> Get the embeddings
if MODE=="emb":

    # Retrieve the test dataset
    test_dataset = General.recover_data(path_data, name = 'test')

    # Retrieve the indices
    indices = General.recover_data(path_masks, name = 'test_ids')
    # Generate a DataLoader object for test
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        num_workers=4,
        shuffle=False,
        pin_memory=True
    )

 
    # Call simple inference
    embeddings_val = embeddings(args.modelname, model, test_dataloader, device, path_model)
    # Serialize the embeddings together with the indices

    path_emb = './data/emb'
    General.serialize_data((indices, embeddings_val), path_emb, name = 'emb.pkl')

    # Print info -> using test split
    print(f'Shape of the embeddings({embeddings_val.shape})')

