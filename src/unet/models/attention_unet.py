from unet.utils import DownSample, UpSampleAttention
import torch
import torch.nn as nn

class AttentionUNet(nn.Module):
    """ 
        Implementation of AttentionUNet general architecture
    """
    def __init__(self, in_channels:int, num_classes:int):
        super().__init__()

        # Define the downsampling part
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)
        self.embedding_storage = None


        # Define the bottleneck
        self.bottleneck =  DownSample(512, 1024)

        # Define the upsampling part
        self.up_convolution_1 = UpSampleAttention(1024, 512, 512)
        self.up_convolution_2 = UpSampleAttention(512, 256, 256)
        self.up_convolution_3 = UpSampleAttention(256, 128,  128)
        self.up_convolution_4 = UpSampleAttention(128, 64,  64)

        # Defining the output conv
        self.out_conv = nn.Conv2d(in_channels = 64, out_channels = num_classes, kernel_size = 1)

    def forward(self, x:torch.Tensor):

        # Downsampling
        conv_out_1, down_1 = self.down_convolution_1(x)
        conv_out_2, down_2 = self.down_convolution_2(down_1)
        conv_out_3, down_3 = self.down_convolution_3(down_2)
        conv_out_4, down_4 = self.down_convolution_4(down_3)

        # The bottleneck
        b, _ = self.bottleneck(down_4)

        #Upsampling
        up_1  = self.up_convolution_1(b, conv_out_4)
        up_2  = self.up_convolution_2(up_1, conv_out_3)
        up_3  = self.up_convolution_3(up_2, conv_out_2)
        up_4  = self.up_convolution_4(up_3, conv_out_1)

        # Defining the outout
        out = self.out_conv(up_4)

        # Mean per map and keep the channels -->[batch, 64]
        embedding = torch.mean(up_4, dim = [2,3])
        self.embedding_storage = embedding.detach()

        return out



    def embeddings(self, x:torch.Tensor)->torch.Tensor:
        """  
            Function aimed to extract the embeddings
            from a given net model

            Params:
                x(torch.Tensor): The input Tensor size (batch, ch, h, w)

            Returns:
                embeddings(torch.Tensor): The embeddings Tensor
        
        """

        # Downsampling
        conv_out_1, down_1 = self.down_convolution_1(x)
        conv_out_2, down_2 = self.down_convolution_2(down_1)
        conv_out_3, down_3 = self.down_convolution_3(down_2)
        conv_out_4, down_4 = self.down_convolution_4(down_3)

        # The bottleneck
        b, _ = self.bottleneck(down_4)

        #Upsampling
        up_1  = self.up_convolution_1(b, conv_out_4)
        up_2  = self.up_convolution_2(up_1, conv_out_3)
        up_3  = self.up_convolution_3(up_2, conv_out_2)
        up_4  = self.up_convolution_4(up_3, conv_out_1)

        #Get the embedding:mean per map and keep the channels -->[batch, 64]
        embeddings = torch.mean(up_4, dim = [2,3])
        return embeddings
        
    