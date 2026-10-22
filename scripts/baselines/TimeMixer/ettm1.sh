#!/usr/bin/env bash
# 论文 Table 4/5 baseline 自跑脚本: TimeMixer on ETTm1
# 超参来源: part1 scripts/TimeMixer/etth1.sh
# cd 到仓库根目录（脚本内全部使用相对路径）
cd "$(dirname "$0")"/../../..

mkdir -p logs/TimeMixer/ETTm1

model_name=TimeMixer
Dataset=ETTm1
seq_len=336
for random_seed in 2024
do
    for pred_len in 96 192 336 720
    do
        python -u run.py \
          --random_seed $random_seed \
          --is_training 1 \
          --data_path './all_six_datasets/ETT-small/ETTm1.csv' \
          --data_set 'ETTm1' \
          --model TimeMixer \
          --seq_len 336 \
          --decomp_len 336 \
          --label_len 48 \
          --pred_len $pred_len \
          --moving_avg 12 \
          --dropout 0.1 \
          --patch_len_list 16 \
          --stride_list 8 \
          --d_model 16 \
          --d_ff 32 \
          --n_heads 8 \
          --batch_size 16 \
          --patience 5 \
          --learning_rate 0.001 \
          --elayer_num 3 \
          --e_layers 3 \
          --down_sampling_layers 3 \
          --down_sampling_method avg \
          --down_sampling_window 2 \
          --dlayer_num 1 \
          --train_epochs 100 \
          --gpu 0 \
          --itr 1 >logs/TimeMixer/ETTm1/TimeMixer_ETTm1_336_${pred_len}_${random_seed}.log
    done
done
