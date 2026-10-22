#!/usr/bin/env bash
# 论文 Table 9 / Figure 12 趋势分支替换：Top-Down (T)
# 超参来源: scripts/STMixer/ETTm1/our_modelv5.sh（全模型同配置，仅换 --model）
# 下采样参数: --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg（part1 实测值，见任务台账）
# cd 到仓库根目录（脚本内全部使用相对路径）
cd "$(dirname "$0")"/../../../../

mkdir -p logs/STMixer_Trend_TD/ETTm1_slim

model_name=STMixer_Trend_TD
Dataset=ETTm1
seq_len=336
for random_seed in 2021 2022 2023
do
    for pred_len in 96 192 336 720
    do
        python -u run.py \
          --random_seed $random_seed \
          --is_training 1 \
          --data_path 'ETT-small/ETTm1.csv' \
          --data_set 'ETTm1' \
          --model STMixer_Trend_TD \
          --seq_len 336 \
          --label_len 48 \
          --pred_len $pred_len \
          --decomp_len 432 \
          --dropout 0.1 \
          --embed_size 8 \
          --sample_embed_size 8 \
          --time_embed_size 8 \
          --t_size 4 \
          --sample_fusion 4 \
          --sampling_list 4 8 \
          --singe_sample_dim 4 \
          --patch_len_list 7 \
          --stride_list 4 \
          --layer_num_cnn 1 \
          --norm True \
          --moving_avg 72 96 120 \
          --batch_size 16 \
          --patience 3 \
          --learning_rate 0.0001 \
          --elayer_num 1 \
          --dlayer_num 2 \
          --down_sampling_layers 2 \
          --down_sampling_window 2 \
          --down_sampling_method avg \
          --train_epochs 100 \
          --gpu 2 \
          --itr 1 >logs/STMixer_Trend_TD/ETTm1_slim/STMixer_Trend_TD_ETTm1_336_${pred_len}_${random_seed}.log
    done
done
