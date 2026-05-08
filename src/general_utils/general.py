import os
import pickle
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

class General():

    def __init__():
        pass

    @staticmethod
    def ensure_dir(path: str)->None:
        """
            Function aimed to check path's 
            existance
            
            Params:
                path(str): The expected path

            Returns:
                bool: True if exists and None if it is created

        """

        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print(f"The dir ({path}) exists")
            return True
        
    
    @staticmethod
    def path_builder(base:str, name:str)->str:
        """     
            Function aimed to build a full path

            Params:
                base(str): The base path
                name(str): The filename

            Returns:
                final_str(str): Composed address
        """

        return os.path.join(base, name)


    @staticmethod
    def serialize_data(data:object, path:str, name:str)->None:
        """ 
            Function aimed to serialize data 

            Params:
                data(object): The data object to store
                path(str): The base path to store data
                name(str): The specific filename
            Returns:
                None

        """
        # Check extension
        if len(name.split('.'))==1:
            name+='.pkl'
        
        # Create full path
        full_path = General.path_builder(path, name )

        # Ensure dir existance
        General.ensure_dir(path)

        #Make sure the file is not already there
        if not os.path.exists(full_path):
            # Serialize using pickle
            with open(full_path, mode = 'wb') as file:
                pickle.dump(data, file)
        


    @staticmethod
    def recover_data(path:str, name:str)->DataLoader:
        """ 
            Function aimed to recover serialized
            data

            Params:
                path(str): The path where data is stored
                name(str): The name of the specific file where data is

            Returns:
                data(DataLoader): The data object 
        """
        # Check extension
        if len(name.split('.'))==1:
            name+='.pkl'

        # Create the full path
        full_path = General.path_builder(path, name)

        # Get data from path
        try:
            with open(full_path, mode = "rb") as file:
                data = pickle.load(file)
            return data
        
        except FileNotFoundError as e:
            print(f'Exception({e})')

        

    @staticmethod
    def metametrics(dataloader:DataLoader)->float:
        """     
            Function aimed to compute class 
            one's weight
        """
        # Define structures
        zeros = 0
        ones = 0
        total = 0

        # Iterate dl
        for _, masks in dataloader:
            total+=masks.numel()
            ones+=torch.sum(masks).item()

        # Obtain weight ones/zeros *x =  1 --> x = 1/(ones/zeros)
        zeros = total - ones
        return 1/(ones/zeros)
    

        
    @staticmethod
    def plotting_module(loss_values:dict[str, tuple[float]],title:str, name:str)->None:
        """     
            Function aimed to visualize train and validation
            loss per epoch. 

            Params:
                loss_values(dict[str, tuple[float]]): The loss values
                title(str): The main title for the plot
                name(str): The name of the figure
            Returns:
                None
        """

        # Define the figure
        fig, axes = plt.subplots(1,2, figsize = (10,5))

        # Get the values
        train_loss, val_loss = list(zip(*loss_values.values()))
        
        # Set main tile
        plt.suptitle(title, fontweight = 'bold')
        # Define the first plot
        axes[0].set_title('Train loss evolution', fontweight = 'bold')
        axes[0].plot(train_loss)
        axes[0].set_xlabel('Epoch', color = 'red')
        axes[0].set_ylabel("Cumulative loss", color = 'red')

        # Define the second plot
        axes[1].set_title('Validation loss evolution', fontweight = 'bold')
        axes[1].plot(val_loss)
        axes[1].set_xticks(ticks = loss_values.keys())

        # Save the image 
        base_path = './loss'
        os.makedirs(base_path, exist_ok = True)
        plt.savefig(f'{base_path}/{name}.png', dpi = 600)



    @staticmethod
    def overlapping_plot(img:torch.Tensor, name:str,  mask_gt:torch.Tensor, mask_pre:torch.Tensor)->None:
        """
            Function aimed to visualize img tensor and its 
            overlap with the masks

            Params:
                patch(torch.Tensor): The image 
                name(str): The name of the image
                mask_gt(torch.Tensor): The ground truth mask
                mask_pre(torch.Tensor): The predicted mask
        """

        # Show the main image 
        plt.figure(figsize = (8,8))
        plt.imshow(img, cmap = 'gray')

        # Define types of predictions
        tp = (mask_gt==1)&(mask_pre==1)
        fp = (mask_pre==1)&(mask_gt==0)
        fn = (mask_pre==0)&(mask_gt==1)

        # Print tp, fp, fn
        plt.imshow(np.ma.masked_where(~tp, tp), vmin = 0, vmax = 1, cmap = 'Greens', alpha = 0.6)
        plt.imshow(np.ma.masked_where(~fp, fp), vmin = 0, vmax = 1, cmap = 'Reds')
        plt.imshow(np.ma.masked_where(~fn, fn),vmin = 0, vmax = 1,  cmap = 'summer')
        # Enhance the plot
        plt.title('Overlap between gt and predicted mask', fontweight = 'bold')
        plt.axis('off')

        # Extract metrics
        overlap_degree = round(((mask_gt*mask_pre).sum()/(mask_gt.sum())).item(), 3)

        # Define handles for the legend
        legend_handles = [
            Patch(facecolor = 'green', label = f'True Positives'),
            Patch(facecolor = 'red', label = f'False Positives'),
            Patch(facecolor = 'yellow', label = f'False Negatives'),
            Patch(facecolor = 'None', label = f'Overlap degree({overlap_degree})')
        ]
        plt.legend(handles = legend_handles, loc='upper left')
        
        # Save the figure
        os.makedirs('./test_visz', exist_ok = True)
        plt.savefig(f'./test_visz/{name}.png')
        plt.close()