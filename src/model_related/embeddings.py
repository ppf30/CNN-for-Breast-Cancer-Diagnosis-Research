# Libraries import
from typing import Union
import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from general_utils import Metrics, General
import random

# Withing project imports
from unet.models import UNet, AttentionUNet

def embeddings(modelname:str, model:Union[UNet, AttentionUNet], dataloader:DataLoader, device:str, path:str)->torch.Tensor:
    """ 
        Function aimed get the embeddings from a given model

        Params:
            modelname(str): The name of the model to load
            model(Union[UNet, AttentionUnet]): The given model 
            dataloader(DataLoader): The given dataset
            device(str): Define the device to use
            path(str): The path to load the model from


        Returns:
            embeddings(torch.Tensor): The embeddings tensor
           
    """


    # Load the model  -> build path adding name
    model_path  = path + f'/{modelname}.pth'
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()


    # Generate the embeddings from a trained model
    with torch.no_grad():
        # Input a single batch
        for idx, (data, labels) in enumerate(dataloader):
            data = data.to(device)
            embeddings = model.embeddings(data)

    return embeddings

