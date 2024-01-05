import torch.nn as nn
import copy
from models.layer.residual_connection_layer import ResidualConnectionLayer

class EncoderBlock(nn.Module):
    
    # self_attention : Attention Layer
    # position_ff : Feed-Forward Layer
    def __init__(self, self_attention, position_ff, norm, dr_rate=0):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff
        self.residual1 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.residual2 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        
    def forward(self, src, src_mask):
        out = src
        out = self.residual1(out, lambda out: self.self_attention(query=out, key=out, value=out, mask=src_mask))
        out = self.residual2(out, self.position_ff)
        
        return out