# 冒烟测试：从 MODEL_REGISTRY 取模型类，用 ETTh1 配置做随机张量前向，校验输出形状。
import argparse
import os
import sys
from argparse import Namespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.Main import MODEL_REGISTRY


def build_args() -> Namespace:
    # 论文 Table 3 的 ETTh1 超参（对应 scripts/STMixer/ETTh1/our_modelv5.sh）：
    # seq_len=336, pred_len=96, d_model=8 (embed_size/sample_embed_size/time_embed_size),
    # M_s=2 (elayer_num), M_t=1 (dlayer_num), dropout=0.3, lr=1e-4
    return Namespace(
        seq_len=336,
        decomp_len=432,
        pred_len=96,
        label_len=48,
        d_model=8,
        embed_size=8,
        sample_embed_size=8,
        time_embed_size=8,
        t_size=4,
        sample_fusion=4,
        singe_sample_dim=4,
        sampling_list=[4, 8],
        moving_avg=[12, 24, 168],
        patch_len_list=7,
        stride_list=4,
        layer_num_cnn=1,
        elayer_num=2,
        dlayer_num=1,
        coeff=1,
        dropout=0.3,
        norm=True,
        mark_size=4,
        output_size=7,
    )


def _baseline_inputs(seq_len, pred_len, batch_size, channels, mark_size):
    x_enc = torch.randn(batch_size, seq_len, channels)
    x_mark_enc = torch.randn(batch_size, seq_len, mark_size)
    x_dec = torch.randn(batch_size, pred_len, channels)
    x_mark_dec = torch.randn(batch_size, pred_len, mark_size)
    return (x_enc, x_mark_enc, x_dec, x_mark_dec)


def _smoke_itransformer(batch_size, channels):
    # 超参来源：part1 scripts/iTransformer/etth1.sh（ETTh1-96, seq_len=336）
    args = Namespace(seq_len=336, pred_len=96, d_model=512, d_ff=512, n_heads=8,
                     e_layers=3, dropout=0.1, embed='timeF', freq='h', factor=3,
                     activation='gelu')
    inputs = _baseline_inputs(args.seq_len, args.pred_len, batch_size, channels, 4)
    return MODEL_REGISTRY['iTransformer'](args), inputs, args.pred_len


def _smoke_patchtst(batch_size, channels):
    # 超参来源：part1 scripts/PatchTST/all/etth1.sh（ETTh1-96, seq_len=336）
    args = Namespace(seq_len=336, pred_len=96, d_model=128, d_ff=256, n_heads=8,
                     elayer_num=2, patch_len_list=16, stride_list=8, dropout=0.1,
                     norm=True)
    inputs = _baseline_inputs(args.seq_len, args.pred_len, batch_size, channels, 4)
    return MODEL_REGISTRY['PatchTST'](args), inputs, args.pred_len


def _smoke_timemixer(batch_size, channels):
    # 超参来源：part1 scripts/TimeMixer/etth1.sh（ETTh1-96, seq_len=336）
    args = Namespace(seq_len=336, label_len=48, pred_len=96, d_model=16, d_ff=32,
                     e_layers=3, moving_avg=[12], dropout=0.1,
                     down_sampling_layers=3, down_sampling_window=2,
                     down_sampling_method='avg')
    inputs = _baseline_inputs(args.seq_len, args.pred_len, batch_size, channels, 4)
    return MODEL_REGISTRY['TimeMixer'](args), inputs, args.pred_len


def _smoke_semixer(batch_size, channels):
    # 超参来源：part1 scripts/SEMixer/etth1.sh（ETTh1-96, seq_len=336）
    args = Namespace(seq_len=336, pred_len=96, d_model=128, d_ff=128, n_heads=4,
                     dropout=0.3, input_size=channels, is_training=0)
    inputs = _baseline_inputs(args.seq_len, args.pred_len, batch_size, channels, 4)
    return MODEL_REGISTRY['SEMixer'](args), inputs, args.pred_len


def _smoke_mixlinear(batch_size, channels):
    # 超参来源：part1 scripts/MixLinear/etth1.sh（ETTh1-96, seq_len=336, moving_avg=24）
    args = Namespace(seq_len=336, pred_len=96, moving_avg=[24], alpha=0.95, lpf=15)
    inputs = _baseline_inputs(args.seq_len, args.pred_len, batch_size, channels, 4)
    return MODEL_REGISTRY['MixLinear'](args), inputs, args.pred_len


def _smoke_dlinear(batch_size, channels):
    # 超参来源：part1 scripts/Dlinear/all/etth1.sh（ETTh1-96, seq_len=336, moving_avg=24）
    args = Namespace(seq_len=336, pred_len=96, moving_avg=[24])
    inputs = _baseline_inputs(args.seq_len, args.pred_len, batch_size, channels, 4)
    return MODEL_REGISTRY['DLinear'](args), inputs, args.pred_len


def _smoke_linear(batch_size, channels):
    # 超参来源：part1 scripts/Linear/all/Linear.sh（ETTh1-96, seq_len=336）
    args = Namespace(seq_len=336, pred_len=96)
    inputs = _baseline_inputs(args.seq_len, args.pred_len, batch_size, channels, 4)
    return MODEL_REGISTRY['Linear'](args), inputs, args.pred_len


def _smoke_nlinear(batch_size, channels):
    # 超参来源：part1 scripts/Nlinear/all/Nlinear.sh（ETTh1-96, seq_len=336）
    args = Namespace(seq_len=336, pred_len=96)
    inputs = _baseline_inputs(args.seq_len, args.pred_len, batch_size, channels, 4)
    return MODEL_REGISTRY['NLinear'](args), inputs, args.pred_len


# baseline 前向分派：注册名 -> (model, inputs, pred_len) 构造器
BASELINE_FORWARD = {
    'iTransformer': _smoke_itransformer,
    'PatchTST': _smoke_patchtst,
    'TimeMixer': _smoke_timemixer,
    'SEMixer': _smoke_semixer,
    'MixLinear': _smoke_mixlinear,
    'DLinear': _smoke_dlinear,
    'Linear': _smoke_linear,
    'NLinear': _smoke_nlinear,
}


def main():
    parser = argparse.ArgumentParser(description='ST-Mixer forward smoke test')
    parser.add_argument('--model', default='STMixerv5', help='model name in MODEL_REGISTRY')
    parser.add_argument('--batch_size', type=int, default=4)
    args_cli = parser.parse_args()

    if args_cli.model not in MODEL_REGISTRY:
        raise SystemExit(f"model '{args_cli.model}' not in MODEL_REGISTRY: {sorted(MODEL_REGISTRY)}")

    torch.manual_seed(2023)

    if args_cli.model in BASELINE_FORWARD:
        model, inputs, pred_len = BASELINE_FORWARD[args_cli.model](args_cli.batch_size, 7)
        model = model.eval()
        with torch.no_grad():
            outputs = model(*inputs)
        expected = (args_cli.batch_size, pred_len, 7)
        assert tuple(outputs.shape) == expected, f"expected {expected}, got {tuple(outputs.shape)}"
        print(f"model={args_cli.model} output shape: {tuple(outputs.shape)}")
        assert not torch.isnan(outputs).any(), "output contains NaN"
        return

    args = build_args()
    model = MODEL_REGISTRY[args_cli.model](args).eval()

    B, C = args_cli.batch_size, 7  # ETTh1: 7 个变量
    x_enc = torch.randn(B, args.decomp_len, C)
    x_mark_enc = torch.randn(B, args.decomp_len, args.mark_size)
    x_dec = torch.randn(B, args.pred_len, C)
    x_mark_dec = torch.randn(B, args.pred_len, args.mark_size)

    with torch.no_grad():
        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    # STMixerv5 系列返回 (outputs, end_paddings_list)
    outputs = out[0] if isinstance(out, tuple) else out
    expected = (B, args.pred_len, C)
    assert tuple(outputs.shape) == expected, f"expected {expected}, got {tuple(outputs.shape)}"
    print(f"model={args_cli.model} output shape: {tuple(outputs.shape)}")
    assert not torch.isnan(outputs).any(), "output contains NaN"


if __name__ == '__main__':
    main()
