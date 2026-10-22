# 出处：论文 Table 4/5 自跑 baseline，迁移自 part1 Model/Linear.py
import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
 

class Linear(nn.Module): 

    def __init__(self, args ):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Linear, self).__init__() 
        self.args = args
        self.seq_len = args.seq_len  
        self.pred_len = args.pred_len   
 
        self.Linear = MLPBlock(args.seq_len  ,args.seq_len  ,args.pred_len,0,1  )

        self.Linear.fc1.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
 
    def encoder(self, x):
        x = x.permute( 0, 2, 1)#（B,V,T）
 
        x_output = self.Linear(x) 
        x = x_output #（B,V,P）
        return x.permute(0, 2, 1) #（B,P,V）

 
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec ): 
        dec_out = self.encoder(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
 
#对季节项下采样
class MLPBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim, output_dim, dropout, mlp_mode=1) :
        super().__init__()
        self.mlp_mode = mlp_mode
        if mlp_mode==2:
            self.fc1 = nn.Linear(input_dim, mlp_dim)
            self.gelu = nn.GELU()
            self.dropout = nn.Dropout(dropout) 
            self.fc2 = nn.Linear(mlp_dim, output_dim)
        elif mlp_mode==1:
            self.fc1 = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        # [B, L, D] or [B, D, L]
        if self.mlp_mode==2:
            return self.fc2(self.dropout(self.gelu(self.fc1(x))))
        elif self.mlp_mode==1:
            return self.fc1(x)