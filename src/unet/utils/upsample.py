from .doubleconv import DoubleConv
import torch
import torch.nn as nn

class UpSample(nn.Module):
    """  
        Implementing upsampling block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Define the upsample
        self.ups = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size = 2, stride = 2)

        # Define the double conv
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1:torch.Tensor, x2:torch.Tensor):

        # Upsampling
        x1 = self.ups(x1)

        # Concat process --> concat channels 256 + 256
        x = torch.cat([x1,x2],1)

        return self.conv(x)