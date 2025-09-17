import torch 
import torch.nn as nn
import torch.nn.functional as F 

class SelfAttentionModule(nn.Module):
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels 
        
        # Group Norm
        numGroups = min(8,channels)
        if(numGroups % 8 != 0):
            numGroups = 1    
        self.norm = nn.GroupNorm(num_groups=numGroups, num_channels=channels)
        
        # Projections for K,Q,V
        self.Query = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.Key = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.Value = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)
        
        # Output Projection (Feed Forward Network)
        self.FFN = nn.Conv1d(channels, channels, kernel_size=1)
        
        # Scaling factor for dot product attention
        self.ScalingFactor = channels ** -0.5
        
    def forward(self, x):
        
        BatchSize, Channels, Height, Width = x.shape
        residual = x
        
        # Normalization
        x = self.norm(x)
        
        # Reshaping to [B,C,HW]
        flattened_X = x.reshape(BatchSize, Channels, -1)
        
        Q = self.Query(flattened_X)
        K = self.Key(flattened_X)
        V = self.Value(flattened_X)
        
        # Computing Attention Scores
        attentionScores = torch.bmm(Q.permute(0,2,1),K) * self.ScalingFactor
        attentionScores = F.softmax(attentionScores, dim=-1)
    
        # BMM with V    
        Output = torch.bmm(V, attentionScores.permute(0,2,1))
    
        # FFN Layer
        Output = self.FFN(Output)
        
        # Reshaping
        Output = Output.reshape(BatchSize, Channels, Height, Width)
        
        # Residual connection
        return Output+residual
        