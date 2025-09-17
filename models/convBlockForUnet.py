import torch
import torch.nn as nn 

class ConvBlockForUnet(nn.Module):
    
    def __init__(self, inChannels, outChannels, timeEmbeddingDimension=None):
        super().__init__()
        
        # Projection Layer if TimeEmbeddingDimension is available to match output shape
        self.TimeProjection = None
        if timeEmbeddingDimension is not None:
            self.TimeProjection = nn.Linear(timeEmbeddingDimension, outChannels)
            
        # First Conv Block
        self.convLayer1 = nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=3, padding=1, stride=1)
        
        # Should Be divisible by numGroups
        numGroups = min(8,outChannels)
        if(outChannels % 8 != 0):
            numGroups = 1
        self.groupNorm1 = nn.GroupNorm(num_groups=numGroups, num_channels=outChannels)
        
        self.activation1 = nn.SiLU()
        
        # Second Conv Block
        self.convLayer2 = nn.Conv2d(in_channels=outChannels, out_channels=outChannels, kernel_size=3, padding=1, stride=1)
        self.groupNorm2 = nn.GroupNorm(num_groups=numGroups, num_channels=outChannels)
        self.activation2 = nn.SiLU()
        
    def forward(self, x, time_embedding=None):
        
        # Forward Pass through the convolutional block defined above
        
        # First Block Pass
        h = self.convLayer1(x)
        h = self.groupNorm1(h)
        h = self.activation1(h)
        
        # Time information
        if self.TimeProjection is not None and time_embedding is not None:
            
            TimeEmbedding = self.activation1(self.TimeProjection(time_embedding))
            TimeEmbedding = TimeEmbedding[:, :, None, None]
            h = h + TimeEmbedding
            
        # Second Block Pass
        h = self.convLayer2(h)
        h = self.groupNorm2(h)
        h = self.activation2(h)
        
        return h