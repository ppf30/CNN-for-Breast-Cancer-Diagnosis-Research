import sys
import argparse
from typing import Union
from .unet.models import UNet, AttentionUNet
import torch.nn as nn
import torch.optim as optim


# Define the parser
parser = argparse.ArgumentParser(description="Entrenamiento de UNet para segmentación")

# Add arguments to the parser
parser.add_argument("--batch_size", type=int, default=64, help = "Batch Size")
parser.add_argument("--lr", type=float, default=0.001, help = "Learning Rate")
parser.add_argument("--in_ch",type=int, default=64, help='Input Channels')
parser.add_argument("--out_cl", type=int, default=6, help = "Output Classes")
parser.add_argument("--model", type=int, default=6, help = "Model(attention_unet:'att'--unet:'base')")
parser.add_argument("--epochs", type=int, default=6, help = "Training Epochs")

# Parse args from Command Line Input
args = parser.parse_args()


# Define constant params
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
IN_CHANNELS = args.in_ch
CLASSES = args.out_cl
EPOCHS = args.epochs


# CREATE THE MODEL  
params = {"in_channels": IN_CHANNELS, "num_classes":CLASSES}
clean_req = args.model.strip().lower()
device = 'gpu' if torch.cuda.is_available() else "cpu"
model =  UNet(**params) if clean_req == "base" else AttentionUNet()
model = model.to(device)


def train(model:Union[UNet, AttentionUnet], batch_size:int, epochs:int)->Union[UNet, AttentionUnet]:
    """ 
        Function aimed to train the given model 
        over a given dataset

        Params:
            model(Union[UNet, AttentionUnet]): The selected model
            BATCH_SIZE(int): The input size per steps
            EPOCHS(int): The number of epochs

        Returns:
            model(Union[UNet, AttentionUnet]): The trained model 
    """


    # Define mode and loss
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    # Assume we already have a dataloader (--)
    ## to_do LOOK AT ME!!!!


    # Iterate over the different epochs
    for epoch in range(EPOCHS):
        # Information print about the epoch
        print(f'EPOCH({epoch}):')

        # Run each batch
        for i, batch in enumerate(data_loader):
            
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


def generate_outputs(dataloader:DataLoader, model:Union[UNet, AttentionUnet] = model)->TensorDataset:
    """ 
        Function aimed to generate data using a 
        given model 

        Params:
            dataloader(DataLoader): The given dataset
            model(Union[UNet, AttentionUnet]): The given model 

        Returns:
            dataset(TensorDataset): The dataset of (data, label) pairs
    """


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
    return TorchDataset(features, labels_list)
