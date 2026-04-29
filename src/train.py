from typing import Union
import torch
from unet.models import UNet, AttentionUNet
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train(model:Union[UNet], dataloader:DataLoader, epochs:int, device:str, path:str)->list[float]:
    """ 
        Function aimed to train the given model 
        over a given dataset

        Params:
            model(Union[UNet, AttentionUnet]): The selected model
            dataloader(DataLoader): The object to sample batches from
            epochs(int): The number of epochs
            device(str): Define the device to use
            PATH(str): Model's storing path

        Returns:
            loss(list[float]): The loss evolution
    """

    # Define mode and loss
    model.to(device)
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    loss_storage = {}

    # Iterate over the different epochs
    for epoch in range(epochs):
        epoch_loss = 0
        batch_loss  = 0
        # Information print about the epoch
        print(f'EPOCH({epoch})')

        # Run each batch
        for i, batch in enumerate(dataloader):
    
            
            # Divide between input and data
            data, labels = batch
            # Matching the device
            data = data.to(device)
            labels = labels.to(device).float()
    
            # Set gradients to zero per batch
            optimizer.zero_grad()
            # Make predictions for the batch
            output = model(data)
    
            # Compute loss and gradients
            loss = criterion(output, labels)
            loss.backward()

            # Perform an optimizer's step
            optimizer.step()

            # Update data and report
            batch_loss +=loss.item()
            epoch_loss +=loss.item()

            # XBatch information
            if (i+1)%10 == 0:
                avg_loss = batch_loss/10
                print(f"·Batch({i+1}):loss({avg_loss})")
                # batch_loss = 0

        # Store average loss
        avg_loss = epoch_loss/len(dataloader)
        loss_storage[epoch] = avg_loss
        print()

    
    # Save the trained model
    torch.save(model.state_dict(),path+"/model.pth")  

    # Return
    return loss_storage
