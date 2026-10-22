#!/usr/bin/env bash
# 论文 Table 4/5 baseline 自跑脚本: NLinear on ETTm1
# 超参来源: part1 scripts/Nlinear/all/Nlinear.sh
# cd 到仓库根目录（脚本内全部使用相对路径）
cd "$(dirname "$0")"/../../..

mkdir -p logs/NLinear/ETTm1

model_name=NLinear
Dataset=ETTm1
seq_len=336
for random_seed in 2021 2022 2023
do
    for pred_len in 96 192 336 720
    do
        python -u run.py \
          --random_seed $random_seed \
          --is_training 1 \
          --data_path './all_six_datasets/ETT-small/ETTm1.csv' \
          --data_set 'ETTm1' \
          --model NLinear \
          --seq_len 336 \
          --label_len 48 \
          --pred_len $pred_len \
          --decomp_len 336 \
          --batch_size 512 \
          --patience 10 \
          --learning_rate 0.005 \
          --itr 1 >logs/NLinear/ETTm1/NLinear_ETTm1_336_${pred_len}_${random_seed}.log
    done
done
