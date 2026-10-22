# 出处：论文 Table 4/5 自跑 baseline，迁移自 part1 Model/MixLinear.py
import torch
import torch.nn as nn
import torch.fft
import math
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
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

class MixLinear(nn.Module):
    def __init__(self, configs):
        super(MixLinear, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.period_len = configs.moving_avg[0]

        self.kernel = self.period_len
        self.lpf = configs.lpf
        self.alpha = configs.alpha

        self.seg_num_x = math.ceil(self.seq_len / self.period_len)
        self.seg_num_y = math.ceil(self.pred_len / self.period_len)

        self.sqrt_seg_num_x = math.ceil(math.sqrt(self.seq_len / self.period_len))
        self.sqrt_seg_num_y = math.ceil(math.sqrt(self.pred_len / self.period_len))

        self.fft_dim = min(self.lpf, self.seg_num_x)

        self.TLinear1 = nn.Linear(self.sqrt_seg_num_x, self.sqrt_seg_num_y, bias=False)
        self.TLinear2 = nn.Linear(self.sqrt_seg_num_x, self.sqrt_seg_num_y, bias=False)

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=self.kernel + 1,
                                stride=1, padding=int(self.kernel / 2), padding_mode="zeros", bias=False)

        self.FLinear1 = nn.Linear(self.fft_dim, 2, bias=False)
        self.FLinear2 = nn.Linear(2, self.seg_num_y, bias=False)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
        batch_size = x.shape[0]
        enc_in = x.shape[2]

        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, enc_in, self.seq_len) + x

        x = x.reshape(batch_size, enc_in, -1, self.period_len).permute(0, 1, 3, 2)

        x_o = F.pad(x, (0, self.sqrt_seg_num_x ** 2 - x.shape[-1], 0, 0, 0, 0))
        x_o = x_o.reshape(batch_size, enc_in, self.period_len, self.sqrt_seg_num_x, self.sqrt_seg_num_x)

        x_o = self.TLinear1(x_o).permute(0, 1, 2, 4, 3)
        x_t = self.TLinear2(x_o).permute(0, 1, 2, 4, 3)

        x_t = x_t.reshape(batch_size, enc_in, self.period_len, -1).permute(0, 1, 3, 2).reshape(
            batch_size, enc_in, -1).permute(0, 2, 1)

        x_fft = torch.fft.fft(x, dim=3)[:, :, :, :self.fft_dim]
        x_fft_real = x_fft.real
        x_fft_imag = x_fft.imag

        x_fft_real = self.FLinear1(x_fft_real)
        x_fft_imag = self.FLinear1(x_fft_imag)
        x_fft_real = self.FLinear2(x_fft_real)
        x_fft_imag = self.FLinear2(x_fft_imag)

        x_fft = torch.complex(x_fft_real, x_fft_imag).reshape(batch_size, enc_in, self.period_len, -1)

        x_rfft = torch.fft.ifft(x_fft, dim=3).float()
        x_f = x_rfft.permute(0, 1, 3, 2).reshape(batch_size, enc_in, -1).permute(0, 2, 1)

        x = x_t[:, :self.pred_len, :] * self.alpha + seq_mean + x_f[:, :self.pred_len, :] * (1 - self.alpha)

        return x[:, :self.pred_len, :]
