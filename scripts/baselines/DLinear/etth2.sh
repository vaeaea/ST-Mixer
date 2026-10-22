#!/usr/bin/env bash
# 论文 Table 4/5 baseline 自跑脚本: DLinear on ETTh2
# 超参来源: part1 scripts/Dlinear/all/etth1.sh
# cd 到仓库根目录（脚本内全部使用相对路径）
cd "$(dirname "$0")"/../../..

mkdir -p logs/DLinear/ETTh2

model_name=DLinear
Dataset=ETTh2
seq_len=336
for random_seed in 2021
do
    for pred_len in 96 192 336 720
    do
        python -u run.py \
          --random_seed $random_seed \
          --is_training 1 \
          --data_path './all_six_datasets/ETT-small/ETTh2.csv' \
          --data_set 'ETTh2' \
          --model DLinear \
          --seq_len 336 \
          --label_len 48 \
          --pred_len $pred_len \
          --decomp_len 336 \
          --moving_avg 24 \
          --batch_size 512 \
          --patience 10 \
          --learning_rate 0.005 \
          --itr 1 >logs/DLinear/ETTh2/DLinear_ETTh2_336_${pred_len}_${random_seed}.log
    done
done
