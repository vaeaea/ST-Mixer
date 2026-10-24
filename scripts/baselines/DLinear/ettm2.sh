#!/usr/bin/env bash
# 论文 Table 4/5 baseline 自跑脚本: DLinear on ETTm2
# 超参来源: part1 scripts/Dlinear/all/etth1.sh
# cd 到仓库根目录（脚本内全部使用相对路径）
cd "$(dirname "$0")"/../../..

mkdir -p logs/DLinear/ETTm2

model_name=DLinear
Dataset=ETTm2
seq_len=336
for random_seed in 2021 2022 2023
do
    for pred_len in 96 192 336 720
    do
        python -u run.py \
          --random_seed $random_seed \
          --is_training 1 \
          --data_path './all_six_datasets/ETT-small/ETTm2.csv' \
          --data_set 'ETTm2' \
          --model DLinear \
          --seq_len 336 \
          --label_len 48 \
          --pred_len $pred_len \
          --decomp_len 336 \
          --moving_avg 24 \
          --batch_size 512 \
          --patience 10 \
          --learning_rate 0.005 \
          --itr 1 >logs/DLinear/ETTm2/DLinear_ETTm2_336_${pred_len}_${random_seed}.log
    done
done
