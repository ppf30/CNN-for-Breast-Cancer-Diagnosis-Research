import sys
import argparse
from typing import Union
from unet.models import UNet, AttentionUNet
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader



def train(model:Union[UNet, AttentionUNet], dataloader:DataLoader, batch_size:int, epochs:int, device:str, path:str)->Union[UNet, AttentionUNet]:
    """ 
        Function aimed to train the given model 
        over a given dataset

        Params:
            model(Union[UNet, AttentionUnet]): The selected model
            dataloader(DataLoader): The object to sample batches from
            batch_size(int): The input size per steps
            epochs(int): The number of epochs
            device(str): Define the device to use
            PATH(str): Model's storing path

        Returns:
            model(Union[UNet, AttentionUnet]): The trained model 
    """


    # Define mode and loss
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())


    # Iterate over the different epochs
    for epoch in range(epochs):
        # Information print about the epoch
        print(f'EPOCH({epoch}):')

        # Run each batch
        for i, batch in enumerate(dataloader):
            
            # Divide between input and data
            data, labels = batch
            # Matching the device
            data = data.to(device)
    
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
            current_loss +=loss.item()
            if (i+1)%1000== 0:
                print(f"batch({i+1}):loss{loss}")
                current_loss = 0

    # Save the trained model
    torch.save(model.state_dict(),path)  
