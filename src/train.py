from typing import Union
import torch
from unet.models import UNet, AttentionUNet
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from general_utils import General


def combined_loss(output:torch.Tensor, labels:torch.Tensor, bce_criterion, alpha:float)->float:
    """
        Function aimed to provide a weighted combined loss schedule.It 
        includes BCE and DICE. The weight they're provided depends on 
        which phase of the process we are currently at; preferring
        greater important of BCE for earlier stages. 
    
        
        Params:
            output(torch.Tensor): Net's output
            labels(torch.Tensor): Label's tensor
            bce_criterion(BCEWithLogitsLoss): The bce loss criterion
            alpha(float): The weighting value

        Returns:
            final_loss(float): The final loss value

   """
    # Compute bce
    bce = bce_criterion(output, labels)
    
    # Compute dice
    probs = torch.sigmoid(output)
    smooth = 1.
    intersection = (probs * labels).sum(dim=(1,2,3))
    dice_loss = 1 - (2. * intersection + smooth) / (
        probs.sum(dim=(1,2,3)) + labels.sum(dim=(1,2,3)) + smooth)
    return alpha * bce + (1-alpha) * dice_loss.mean()
def train(modelname:str, val_loader, model:Union[UNet, AttentionUNet], dataloader:DataLoader, epochs:int, device:str, path:str, weight:float, plot:bool)->list[float]:
    """ 
        Function aimed to train the given model 
        over a given dataset

        Params:
            modelname(str): The name of the model
            model(Union[UNet, AttentionUnet]): The selected model
            dataloader(DataLoader): The object to sample batches from
            epochs(int): The number of epochs
            device(str): Define the device to use
            path(str): Model's storing path
            weight(float): The positive class weight
            plot(bool): If a plot is seeked or not

        Returns:
            loss(list[float]): The loss evolution (train, val)
    """

    # Define base loss and schedule
    weigth_tensor = torch.tensor(weight).to(device)
    base_criterion = nn.BCEWithLogitsLoss(pos_weight = weigth_tensor)

    
    # Define mode and device
    model.to(device)
    model.train()

    # Define scaler, scheduler and optimizer
    scaler = torch.amp.GradScaler(device)
    optimizer = optim.Adam(model.parameters(), lr  = 1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max = epochs*2, eta_min=1e-05)


    # Define reporting structures
    loss_storage = {}
    best_val = float('inf')

    # Iterate over the different epochs
    for epoch in range(epochs):
        epoch_loss = 0
        batch_loss = 0
        # Information print about the epoch
        print(f'\nEPOCH({epoch})')

        # Run each batch
        for i, batch in enumerate(dataloader):
            # Divide between input and data
            data, labels = batch
            # Matching the device
            data = data.to(device)
            labels = labels.to(device).float()
            
            # Set gradients to zero per batch
            optimizer.zero_grad()

            # Switching to automatic mixed precission
            with torch.amp.autocast(device):
                # Make predictions for the batch
                output = model(data)
                # Compute loss
                loss = combined_loss(output, labels, base_criterion,  0.5)

            # Scales loss. Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()

            # Perform an optimizer's step
            scaler.step(optimizer)
            scaler.update()

            # Update data and report
            batch_loss +=loss.item()
            epoch_loss +=loss.item()

            # XBatch information
            if (i+1)%10 == 0:
                avg_loss = batch_loss/10
                print(f"·Batch({i+1}):loss({avg_loss})")
                batch_loss = 0

        # Perform scheduler step -> case Cossine LR scheduler
        scheduler.step()

        # Validation -> the end of each epoch
        # Set the model to eval
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                # Load data and labels
                data, labels = batch
                # Set the device
                data, labels = data.to(device), labels.to(device).float()
                # Apply automatic quantization
                with torch.amp.autocast(device_type = device):
                    # Get output and loss
                    output = model(data)
                    val_loss += combined_loss(output, labels, base_criterion, alpha = 0.5).item()

        # Handle validation loss
        avg_val_loss = val_loss / len(val_loader)
        # Scheduler step based on validation
        #scheduler.step(avg_val_loss)
        #print("LR:", optimizer.param_groups[0]['lr'])
        # Model storage based on validation loss
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            torch.save(model.state_dict(), f'{path}/{modelname}.pth')
        
        # Handle train loss behaviour
        avg_loss = epoch_loss/len(dataloader)
        loss_storage[epoch] = (avg_loss, avg_val_loss)
       
    
        # Info printing about loss
        print(f'Val loss: {avg_val_loss:.4f}')
        print(f'Cumulated loss epoch:{avg_loss}')

        # Set model again to train
        model.train() 

    
    # Save the trained model -> now is validation-based approach
    #torch.save(model.state_dict(), f'{path}/{modelname}.pth')  

    # Save train-val loss curves
    try:
        if plot:
            title = f"Epochs({epochs})|No loss schedule|"
            General.plotting_module(loss_storage, title, modelname)
    except:
        pass

    return loss_storage