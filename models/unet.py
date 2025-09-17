import torch 
import torch.nn as nn 
import sys

sys.path.append("/home/jovyan/beomi/ayushraina/ddpm_pipeline")
from models.time_embedding import SinusoidalTimeEmbedding
from models.convBlockForUnet import ConvBlockForUnet
from models.selfAttention import SelfAttentionModule

class SimpleUNetModel(nn.Module):
    # Basic U-Net structure for diffusion model
    
    def __init__(self, inChannels=3, outChannels=3, baseChannels=64, timeEmbeddingDimension=128, depth=3, debug=False):
        super().__init__()
        
        # Time Embedding Layer
        self.timeEmbeddingGenerator = SinusoidalTimeEmbedding(timeEmbeddingDimension) # Class Instance
        self.depth = depth 
        self.debug = debug
        
        # Another MLP for projection
        self.TimeProjectionII = nn.Sequential(
            nn.Linear(timeEmbeddingDimension, timeEmbeddingDimension),
            nn.SiLU(),
            nn.Linear(timeEmbeddingDimension, timeEmbeddingDimension)
        )
        
        self.initial_ConvLayer = nn.Conv2d(in_channels=inChannels, out_channels=baseChannels, kernel_size=3, padding=1, stride=1)
        
        # Downsampling Path (Encoder)
        self.downsampleBlocks = nn.ModuleList()
        self.downSamplers = nn.ModuleList()
        
        # Tracking Channels at each level for adding skip connections later on
        channels = [baseChannels]
        currentChannels = baseChannels
        
        # Creating Downsampling Layers
        for i in range(depth):
            
            # Doubling the number of feature maps at each level
            outChannelsCopy = 2*currentChannels
            
            # Add two conv blocks at each resolution
            self.downsampleBlocks.append(
                ConvBlockForUnet(inChannels=currentChannels, outChannels=outChannelsCopy, timeEmbeddingDimension=timeEmbeddingDimension)
            )
            self.downsampleBlocks.append(
                ConvBlockForUnet(inChannels=outChannelsCopy, outChannels=outChannelsCopy, timeEmbeddingDimension=timeEmbeddingDimension)
            )
            
            # Add downsampler (max pooling) except for the last layer
            if i < depth:
                self.downSamplers.append(nn.MaxPool2d(kernel_size=2))
                
            channels.append(outChannelsCopy)
            currentChannels = outChannelsCopy
            
        # Store Channel Information
        self.middleChannels = currentChannels
        self.channels = channels
        
        # Middle BloPathck with Self Attention Applied to it
        self.middleBlock1 = ConvBlockForUnet(inChannels=currentChannels, outChannels=currentChannels, timeEmbeddingDimension=timeEmbeddingDimension)
        self.middleBlockAttention = SelfAttentionModule(currentChannels)
        self.middleBlock2 = ConvBlockForUnet(inChannels=currentChannels, outChannels=currentChannels, timeEmbeddingDimension=timeEmbeddingDimension)
            
        # Upsampling Path (Decoder)
        self.upsampleBlocks = nn.ModuleList()
        self.upSamplers = nn.ModuleList()
        
        # Reverse the Channel list for upsampling
        reversedChannelList = list(reversed(channels))
        print(reversedChannelList)
        
        # Creating the UpSampling Layers
        for i in range(depth):
            # Upsampler
            self.upSamplers.append(nn.ConvTranspose2d(in_channels=currentChannels, out_channels=currentChannels, kernel_size=2, stride=2))
            
            #Concatenate with skip connection
            skipChannels = reversedChannelList[i+1]  # i+1 because we are skipping bottleneck layer     
            self.upsampleBlocks.append(ConvBlockForUnet(inChannels=skipChannels+currentChannels, outChannels=skipChannels, timeEmbeddingDimension=timeEmbeddingDimension))
            self.upsampleBlocks.append(ConvBlockForUnet(inChannels=skipChannels, outChannels=skipChannels, timeEmbeddingDimension=timeEmbeddingDimension))
            
            # update current channels
            currentChannels = skipChannels
            
        # Final Output Convolution
        self.finalConv = nn.Conv2d(in_channels=baseChannels, out_channels=outChannels, kernel_size=1) # 1*1 convolution at end
            
                    
        print("Simple U net initialized with: ")
        print(f"Input Channels: {inChannels}")
        print(f"Output Channels: {outChannels}")
        print(f"Base Channels: {baseChannels}")
        print(f"Time Embedding Dimension: {timeEmbeddingDimension}")
        print(f"Depth: {depth}")
        print(f"Channel Dimensions: {channels}")
        
    def forward(self, x, timestep):
        
        time_embedding = self.timeEmbeddingGenerator(timestep)
        time_embedding = self.TimeProjectionII(time_embedding)
        
        # Downsampling Path
        # Initial Convolution
        h = self.initial_ConvLayer(x)
        
        # Skip Connection
        skipConnections  = [h]
        
        # Process each level of encoder
        blockIndex = 0
        downsampleIndex = 0
        
        for level in range(self.depth):
            if self.debug:
                print(f"Level {level} - Before downsampling: {h.shape}")
            
            # Conv Block 1
            for i in range(2):
                h = self.downsampleBlocks[blockIndex](h, time_embedding)
                blockIndex += 1
            
            if self.debug:
                print(f"Level {level} - After conv blocks: {h.shape}")
            skipConnections.append(h)
            
            if level < self.depth-1:
                h = self.downSamplers[downsampleIndex](h)
                downsampleIndex += 1
                
                if self.debug:
                    print(f"Level {level} - After downsampling: {h.shape}")
                
        # Middle Block with attention
        h = self.middleBlock1(h, time_embedding)
        h = self.middleBlockAttention(h)
        h = self.middleBlock2(h, time_embedding)
        
        # Upsampling path
        blockIndex = 0
        for level in range(self.depth):
            
            # Upsample
            if self.debug:
                print(f"Level {level} - Before upsampling: {h.shape}")
            if level < self.depth-1:
                h = self.upSamplers[level](h)
            
            if self.debug:
                print(f"Level {level} - After upsampling: {h.shape}")
            
            # Get Skip connection
            skipConnection = skipConnections[-(level+2)]
            
            if self.debug:
                print(f"Level {level} - Skip connection shape: {skipConnection.shape}")
            h = torch.cat([h, skipConnection], dim=1)
            
            # Apply Conv Transpose
            for i in range(2):
                h = self.upsampleBlocks[blockIndex](h, time_embedding)
                blockIndex += 1
        
        if self.debug:
            print(f"Shape before final conv: {h.shape}")
        
        output = self.finalConv(h)
        
        if self.debug:
            print(f"Shape after final conv: {output.shape}")
        
        return output
    