# 出处：论文 Table 4/5 自跑 baseline，迁移自 part1 Model/Nlinear.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Nlinear(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, args):
        super(Nlinear, self).__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec ):
        # x: [Batch, Input length, Channel]
        x = x_enc
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        x = x + seq_last
        return x # [Batch, Output length, Channel]