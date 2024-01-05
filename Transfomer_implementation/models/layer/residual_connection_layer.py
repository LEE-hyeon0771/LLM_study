import torch.nn as nn

class ResidualConnectionLayer(nn.Module):
    # 잔차연결 : Backpropagation 시 발생할 수 있는 Gradient Vanishing 방지
    
    def __init__(self, norm, dr_rate=0):
        super(ResidualConnectionLayer, self).__init__()
        self.norm = norm
        self.dropout = nn.Dropout(p=dr_rate)
        
    def forward(self, x, sub_layer):
        out = x
        out = self.norm(out)
        out = sub_layer(out)
        out = self.dropout(out)
        out = out + x
        return out
        