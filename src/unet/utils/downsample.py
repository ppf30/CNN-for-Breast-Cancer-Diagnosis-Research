
from .doubleconv import DoubleConv
import torch
import torch.nn as nn

class DownSample(nn.Module):
    """ 
        Implementing downsampling blocks
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Getting the double convolution
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x:torch.Tensor):
        conv_out = self.conv(x)
        down = self.pool(conv_out)
        return conv_out, down