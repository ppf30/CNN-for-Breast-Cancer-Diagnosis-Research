from typing import Union
import torch
from unet.models import UNet, AttentionUNet
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train(modelname:str, model:Union[UNet], dataloader:DataLoader, epochs:int, device:str, path:str, weight:float)->list[float]:
    """ 
        Function aimed to train the given model 
        over a given dataset

        Params:
            modelname(str): The name of the model
            model(Union[UNet, AttentionUnet]): The selected model
            dataloader(DataLoader): The object to sample batches from
            epochs(int): The number of epochs
            device(str): Define the device to use
            PATH(str): Model's storing path
            weight(float): The positive class weight

        Returns:
            loss(list[float]): The loss evolution
    """

    # Define mode and loss
    model.to(device)
    model.train()
    weigth_tensor = torch.tensor(weight).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight = weigth_tensor)  
    optimizer = optim.Adam(model.parameters())
    loss_storage = {}

    # early stopping
    patience = 20
    trigger_times = 0
    best_loss = float("inf")
    best_model_path = f'{path}/{modelname}_best.pth'
    last_model_path = f'{path}/{modelname}_last.pth'

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
                batch_loss = 0

        # Store average loss
        avg_loss = epoch_loss/len(dataloader)
        loss_storage[epoch] = avg_loss
        print()

        # Early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            trigger_times = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            trigger_times += 1
            if trigger_times >= patience:
                break

    
    # Save the trained model
    torch.save(model.state_dict(), last_model_path)  

    # Return
    return loss_storage
