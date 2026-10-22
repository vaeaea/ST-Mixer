# 出处：论文 Table 4/5 自跑 baseline，迁移自 part1 Model/patchtst.py
import math
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import torch
import argparse
import numpy as np 
from math import sqrt

class trans_encoder(nn.Module):
    def __init__(self,  patch_len, stride, hidden_size, forward_size, heads, dropout):
        super(trans_encoder, self).__init__()  
        self.atten = FullAttention(hidden_size, heads)
        
        self.conv1 = nn.Conv1d(in_channels=hidden_size, out_channels=forward_size, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=forward_size, out_channels=hidden_size, kernel_size=1, bias=False)

        self.dropout = nn.Dropout(dropout) 
        self.activation = F.relu 
        #self.norm1 = nn.LayerNorm(embed_size)  
        self.norm1 = nn.LayerNorm(hidden_size) 
        self.norm = nn.LayerNorm(hidden_size)
             
    def forward(self, x): #(B*C,N,d)
        x = self.norm(self.atten(x)[0] + x) #(B,T,C)  
        y = x
        #前馈层
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))#(B,e,L)  
        y = self.conv2(y).transpose(-1, 1)#(B,L,e)  
        #分解
        res = self.norm1(x + y)
        return  res#(B,T,C)

class Flatten_Head(nn.Module):
    def __init__(self, input_len, out_len):
        super().__init__()
        
        self.linear = nn.Linear(input_len, out_len)
            
    def forward(self, x):   # x: [bs x nvars x d_model x patch_num] 
        x = self.linear(x)
        return x
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]    
class patchtst(nn.Module):
    def __init__(self, configs):
        super(patchtst, self).__init__()
        self.configs = configs
        self.patch_num = (self.configs.seq_len-self.configs.patch_len_list) // self.configs.stride_list + 1
        self.encoder = nn.ModuleList([trans_encoder(self.configs.patch_len_list, self.configs.stride_list, self.configs.d_model, self.configs.d_ff, self.configs.n_heads, self.configs.dropout) for l in range(self.configs.elayer_num)] )
        self.W_P = nn.Linear(self.configs.patch_len_list, self.configs.d_model)
        self.dropout = nn.Dropout(configs.dropout) 
        self.head = Flatten_Head(self.configs.d_model*self.patch_num, self.configs.pred_len)
        self.W_pos = PositionalEmbedding(self.configs.d_model)
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):  
        # 标准化
        if self.configs.norm==True:
            means = x_enc.mean(1, keepdim=True).detach() #B, T, N
            x_enc = x_enc - means #
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev #标准化  
            
        x_enc, x_enc_clip = patch_clip(x_enc, x_enc.size(1), self.configs.patch_len_list, self.configs.stride_list)  #(B,T,C)
        B,T,C = x_enc_clip.shape
        # print(x_enc_clip.shape)
        x_enc_clip = x_enc_clip.transpose(1,2).reshape(B*C,T,1)
        
        x_enc_clip = x_enc_clip.transpose(1,2).unfold(dimension=-1, size=self.configs.patch_len_list, step=self.configs.stride_list)  #(B,C,P_N,P_L)
        _,_,N,L = x_enc_clip.shape    
        # print(x_enc_clip.shape)
        x_enc_clip = self.dropout(self.W_P(x_enc_clip.reshape(B*C,N,L))+self.W_pos(self.W_P(x_enc_clip.reshape(B*C,N,L))))  #(B*C,N,d)

        # print(x_enc_clip.shape)
        for i in range(self.configs.elayer_num):
            x_enc_clip = self.encoder[i](x_enc_clip)  #(B*C,N,D)
            

        x_enc_clip = x_enc_clip.reshape(B,C,self.patch_num*self.configs.d_model)
        x_enc_clip = self.head(x_enc_clip).permute(0,2,1)  #直接由隐藏状态进行预测 
         
        if self.configs.norm==True:
            x_enc_clip = x_enc_clip * (stdev[:, 0, :].unsqueeze(1).repeat(1,x_enc_clip.shape[1], 1))
            x_enc_clip = x_enc_clip + (means[:, 0, :].unsqueeze(1).repeat(1, x_enc_clip.shape[1], 1))
        return x_enc_clip
    
    
def patch_clip(x, seq_len, patch_len, stride):
    patch_num = (seq_len-patch_len) // stride + 1
    tgt_len = patch_len + stride*(patch_num - 1)
    s_begin = seq_len - tgt_len
    x_clip = x[:, s_begin:, :]  # xb: [bs x tgt_len x nvars]    #对序列进行截断，根据实际长度进行修正
    return x, x_clip    
    
    
class return_qkv(nn.Module):
    def __init__(self, input_size, heads, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.heads = heads
        self.hidden_size = hidden_size
        self.query = nn.Linear(input_size, heads * hidden_size)
        self.key = nn.Linear(input_size, heads * hidden_size)
        self.value = nn.Linear(input_size, heads * hidden_size)
        
    def transpose_for_multihead(self, x): #(B,L,heads *hidden_size)
        x = torch.reshape(x,[x.shape[0],x.shape[1],self.heads,self.hidden_size])   #(B,L,heads ,hidden_size)
        return x
    
    def forward(self, x):  #(B,L,INPUT_sIZE)    
        mix_q = self.query(x)
        mix_k = self.key(x)
        mix_v = self.value(x)
        
        q = self.transpose_for_multihead(mix_q)#(B,L,heads ,hidden_size)
        k = self.transpose_for_multihead(mix_k)#(B,L,heads ,hidden_size)
        v = self.transpose_for_multihead(mix_v)#(B,L,heads ,hidden_size)
        return q,k,v

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask
    
class FullAttention(nn.Module):
    def __init__(self, hidden_size,heads):
        super(FullAttention, self).__init__()
        self.dropout = nn.Dropout(0.05)
        d_head = hidden_size // heads 
        self.return_qkv = return_qkv(hidden_size,heads,d_head)
        self.project = nn.Linear(heads*d_head, hidden_size)
        
    def forward(self, x):
        queries,keys,values = self.return_qkv(x)
        
        B, L, H, E = queries.shape 
        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        
        attn_mask = TriangularCausalMask(B, L, device=queries.device)

        scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)  #(B,L,head,dim)
        V = self.project(V.reshape(V.size(0),V.size(1),-1)) #(B,L,head*dim)==>(B,L,dim)
       
        return (V.contiguous(), A) 
    