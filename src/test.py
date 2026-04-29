# Libraries import
from typing import Union
import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Withing project imports
from unet.models import UNet, AttentionUNet
def test(model:Union[UNet], dataloader:DataLoader, device:str, path:str, needed:bool = False)->TensorDataset:
    """ 
        Function aimed to provide testing over a given model
        given model 

        Params:
            model(Union[UNet, AttentionUnet]): The given model 
            dataloader(DataLoader): The given dataset
            device(str): Define the device to use
            path(str): The path to load the model from
            needed(bool): Flag to switch between inference and data returning


        Returns:
            dataset(TensorDataset): The dataset of (data, label) pairs
    """


    # Load the model  -> build path adding name
    model_path  = path + '/model.pth'
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)

    # Set the model to evaluation
    model.eval()

    # Define the storage structures
    features = []
    labels_list = []

    # Generate the outputs
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            # Get the output from the model 
            output = model(data)
            # Update the structures 
            features.append(output.to(device = 'cpu'))
            labels_list.append(labels)

    # Concat the structures before returning
    features = torch.cat(features)
    labels_list = torch.cat(labels_list)

    # Create the TrainDataset: indexable pair of (data, label)
    return TensorDataset(features, labels_list) if needed else features