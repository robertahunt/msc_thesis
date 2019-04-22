import os
import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
    
    # adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L139
    def save(self, state, filepath):
        torch.save(state, filepath)
        
    def load(self, optimizer, filepath):
        if os.path.isfile(filepath):
            print(f"=> loading checkpoint '{filepath}'")
            checkpoint = torch.load(filepath)
            start_epoch = checkpoint['epoch']
            val_loss = checkpoint['val_loss']
            self.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_dict'])
            print(f"=> loaded checkpoint '{filepath}' (epoch {epoch}) (val_loss {val_loss})")
            return optimizer
        else:
            print(f"=> no checkpoint found at '{filepath}'")