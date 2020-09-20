# MLP
import torch 
import torch.nn as nn
import torch.nn.functional as F 

class SCMLP(nn.Module):
    def __init__(self, genome):
        super().__init__()
        self.genome = genome
        
    def forward(self):
        pass
    
    