import torch.nn as nn
import copy

class Encoder(nn.Module):
    def __init__(self, encoder_block, n_layer):
        super(Encoder, self).__init__()
        self.layers = []
        for i in range(n_layer):
            self.layers.append(copy.deepcopy(encoder_block))
    
    def forward(self, src, src_mask):
        # 첫 block의 input은 encoder 전체의 input인 x
        out = src
        # 이전 block의 output을 이후 block의 input으로 넣음.
        for layer in self.layers:
            out = layer(out, src_mask)
        
        return out
    
    
    