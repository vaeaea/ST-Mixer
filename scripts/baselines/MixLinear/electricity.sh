#!/usr/bin/env bash
# 论文 Table 4/5 baseline 自跑脚本: MixLinear on electricity
# 超参来源: part1 scripts/MixLinear/electricity.sh
# cd 到仓库根目录（脚本内全部使用相对路径）
cd "$(dirname "$0")"/../../..

mkdir -p logs/MixLinear/electricity

model_name=MixLinear
Dataset=electricity
seq_len=336
for random_seed in 2021
do
    for pred_len in 96 192 336 720
    do
        python -u run.py \
          --random_seed $random_seed \
          --is_training 1 \
          --data_path './all_six_datasets/electricity/electricity.csv' \
          --data_set 'electricity' \
          --model MixLinear \
          --seq_len 336 \
          --decomp_len 336 \
          --label_len 48 \
          --pred_len $pred_len \
          --dropout 0.1 \
          --patch_len_list 16 \
          --stride_list 8 \
          --d_model 512 \
          --d_ff 512 \
          --n_heads 8 \
          --batch_size 16 \
          --patience 5 \
          --e_layers 3 \
          --learning_rate 0.001 \
          --elayer_num 2 \
          --dlayer_num 2 \
          --train_epochs 100 \
          --moving_avg 24 \
          --alpha 0.5 \
          --lpf 19 \
          --gpu 0 \
          --itr 1 >logs/MixLinear/electricity/MixLinear_electricity_336_${pred_len}_${random_seed}.log
    done
done
