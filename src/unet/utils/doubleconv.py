import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
        Implementation of the double convolution
        module
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Defining the sequential module
        self.convolutions = nn.Sequential(
            # Two convolutions
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)     
        )

    def forward(self, x:torch.Tensor):
        return self.convolutions(x)  