# Libraries import
from typing import Union
import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from general_utils import Metrics, General
import random

# Withing project imports
from unet.models import UNet, AttentionUNet

def test(modelname:str, model:Union[UNet, AttentionUNet], dataloader:DataLoader, device:str, path:str, sample:bool)->tuple[dict, torch.Tensor]:
    """ 
        Function aimed to provide testing over a given model
        given model 

        Params:
            modelname(str): The name of the model to load
            model(Union[UNet, AttentionUnet]): The given model 
            dataloader(DataLoader): The given dataset
            device(str): Define the device to use
            path(str): The path to load the model from
            sample(bool): If showing an image is seeeked


        Returns:
            A tuple of:
                - metrics(dict[str, float]): The dictionary of metrics
                - embedding_tensor(torch.Tensor): The tensor containing the embeddings
    """


    # Load the model  -> build path adding name
    model_path  = path + f'/{modelname}.pth'
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)

    # Define random image plot
    batch_idx = [random.randint(0, len(dataloader)-1) for i in range(3)] if sample else [i for i in range(len(dataloader))]
    batch_idx = list(set(batch_idx))
    print('The random batches to overlap:', batch_idx if sample is True else 'all')

    # Set the model to evaluation
    model.eval()

    # Define the storage structures
    prediction_list = []
    labels_list = []
    embedding_list = []
    # Generate the outputs
    with torch.no_grad():
        for idx, (data, labels) in enumerate(dataloader):

            data = data.to(device)
            # Get the output from the model 
            logits = model(data)
            # Get the prediction [0,1]-->sigmoid(logit)
            probability = torch.sigmoid(logits)
            classification = (probability>0.5).float()

            # Store the pair (output, labels)
            prediction_list.append(classification.cpu())
            labels_list.append(labels.cpu())

            # Store the embedding
            embedding_list.append(model.embedding_storage.detach().cpu())

            # Generate random image from selected batch
            if idx in batch_idx:
                # Select random img from the batch
                random_image = random.randint(0, data.shape[0]-1)
                # Call the overlapping plot
                General.overlapping_plot(
                    data[random_image, 0].cpu(),
                    f"test_batch_{idx}_image_{random_image}",
                    labels[random_image, 0].cpu(),
                    classification[random_image, 0].cpu(),
                )


    # Create structure to feed the metrics class (total, ch, h, w)
    predictions_tensor = torch.cat(prediction_list)
    labels_tensor  = torch.cat(labels_list)
    embeddings_tensor = torch.cat(embedding_list)

    # Metrics management
    metrics = Metrics(predictions_tensor, labels_tensor)

    # Create the metrics
    metrics.feed_metrics()

    # Show info
    print(metrics)

    return metrics.metrics, embeddings_tensor