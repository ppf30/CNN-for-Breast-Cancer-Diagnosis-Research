import torch.nn as nn
import torch

class AttentionGate(nn.Module):
    def __init__(self, gate_channels, skip_channels, out_channels):
        super().__init__()
        self.conv_gate = nn.Sequential(
            nn.Conv2d(gate_channels, out_channels, kernel_size = 1),
            nn.BatchNorm2d(out_channels)
        )

        self.conv_skip=  nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, kernel_size = 1),
            nn.BatchNorm2d(out_channels)
        )

        self.reduction = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size = 1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace = True)

    def forward(self, g:torch.Tensor, skip:torch.Tensor):

        # Transform to a common space
        # Gate signal --> decoder
        g1 = self.conv_gate(g)
        #Skip connection --> encoder
        skip1 =  self.conv_skip(skip)
        
        # Merge the two signals
        inter = self.relu(g1 + skip1)

        # Generate the attention map 
        att_map = self.reduction(inter)

        # Filter the skip connection
        filtered = att_map * skip

        return filtered