from .attention_gate import AttentionGate
from .doubleconv import DoubleConv
import torch
import torch.nn as nn

class UpSampleAttention(nn.Module):
    """  
        Implementing upsampling block 
        including attention gate
    
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        # Define the upsample
        self.ups = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size = 2, stride = 2)

        # Define the attention gate
        self.attention = AttentionGate(in_channels//2, skip_channels, out_channels)

        # Define the double conv
        self.conv = DoubleConv(in_channels//2 + skip_channels, out_channels)

       

    def forward(self, x:torch.Tensor, skip:torch.Tensor):

        # Upsampling
        x = self.ups(x)

        # Pass skip connection through attention gate
        skip = self.attention(x, skip)

        # Concat process --> different channels (in//2  + )
        x = torch.cat([x,skip],1)

        return self.conv(x)