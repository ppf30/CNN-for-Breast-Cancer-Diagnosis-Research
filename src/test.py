# Libraries import
from typing import Union
import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from general_utils import Metrics

# Withing project imports
from unet.models import UNet, AttentionUNet

def test(modelname:str, model:Union[UNet], dataloader:DataLoader, device:str, path:str, needed:bool = False)->dict[str, float]:
    """ 
        Function aimed to provide testing over a given model
        given model 

        Params:
            modelname(str): The name of the model to load
            model(Union[UNet, AttentionUnet]): The given model 
            dataloader(DataLoader): The given dataset
            device(str): Define the device to use
            path(str): The path to load the model from
            needed(bool): Flag to switch between inference and data returning


        Returns:
            metrics(dict[str, float]): The dictionary of metrics
    """


    # Load the model  -> build path adding name
    model_path  = path + f'/{modelname}.pth'
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)

    # Set the model to evaluation
    model.eval()

    # Define the storage structures
    prediction_list = []
    labels_list = []

    # Generate the outputs
    with torch.no_grad():
        for data, labels in dataloader:

            data = data.to(device)
            # Get the output from the model 
            logits = model(data).to(device = 'cpu')
            # Get the prediction [0,1]-->sigmoid(logit)
            probability = torch.sigmoid(logits)
            classification = (probability>0.5).float()

            # Store the pair (output, labels)
            prediction_list.append(classification)
            labels_list.append(labels)


    
    # Create structure to feed the metrics class (total, ch, h, w)
    predictions_tensor = torch.cat(prediction_list)
    print(predictions_tensor.shape)
    labels_tensor  = torch.cat(labels_list)

    # Metrics management
    metrics = Metrics(predictions_tensor, labels_tensor)

    # Create the metrics
    metrics.feed_metrics()

    # Show info
    print(metrics)

    return metrics.metrics