import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

class MultiHeadAttentionLayer(nn.Module):
    
    def __init__(self, d_model, h, qkv_fc, out_fc):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.q_fc = copy.deepcopy(qkv_fc)
        self.k_fc = copy.deepcopy(qkv_fc)
        self.v_fc = copy.deepcopy(qkv_fc)
        self.out_fc = out_fc
        
    def calculate_attention(query, key, value, mask):
        d_k = key.shape[-1]
        attention_score = torch.matmul(query, key.transpose(-2, -1)) # Q * K^T
        attention_score /= math.sqrt(d_k)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, -1e9)
        attention_prob = F.softmax(attention_score, dim=-1)
        out = torch.matmul(attention_prob, value)
        return out            
    
    
    def forward(self, *args, query, key, value, mask=None):
    # query, key, value: (n_batch, seq_len, d_embed)
    # mask: (n_batch, seq_len, seq_len)
    # value: (n_batch, seq_len, d_k)
        n_batch = query.size(0)
    
        def transform(x, fc):
            out = fc(x)
            out = out.view(n_batch, -1, self.h, self.d_model // self.h)
            out = out.transpose(1,2)
            return out
        
        # (n_batch, seq_len, d_embed)
        # (n_batch, seq_len, d_model)
        # (n_batch, seq_len, h, d_k)
        # (n_batch, h, seq_len, d_k)
        
        query = transform(query, self.q_fc)
        key = transform(key, self.k_fc)
        value = transform(value, self.v_fc)
        
        out = self.calculate_attention(query, key, value, mask) # (n_batch, h, seq_len, d_k)
        out = out.transpose(1,2) # (n_batch, seq_len, h, d_k)
        out = out.contiguous().view(n_batch, -1, self.d_model) # (n_batch, seq_len, d_model)
        out = self.out_fc(out) # (n_batch, seq_len, d_embed)
        
        return out
    