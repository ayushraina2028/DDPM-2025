import torch
import torch.nn as nn 

class SinusoidalTimeEmbedding(nn.Module):
    
    def __init__(self, embedding_dimension):
        super().__init__()
        self.embedding_size = embedding_dimension
        
    def forward(self, timesteps):
        # Convertes timestep to embeddings
        assert len(timesteps.shape) == 1, "Timesteps should be a 1D Tensor only!"
        
        half_dimension = self.embedding_size // 2
        embedding = torch.log(torch.tensor(10000.0)) / (half_dimension-1)
        embedding = torch.exp(torch.arange(half_dimension, device=timesteps.device) * -embedding)
        
        # Outer Product
        embedding = timesteps[:,None] * embedding[None,:]
        
        # Sin and Cos
        embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=1)
        
        # Pad with 0 if embedding size is odd
        if self.embedding_size % 2 != 0:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:,:1])], dim=1)
            
        return embedding

