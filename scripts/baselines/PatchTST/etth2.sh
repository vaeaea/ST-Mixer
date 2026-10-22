#!/usr/bin/env bash
# 论文 Table 4/5 baseline 自跑脚本: PatchTST on ETTh2
# 超参来源: part1 scripts/PatchTST/all/etth1.sh
# cd 到仓库根目录（脚本内全部使用相对路径）
cd "$(dirname "$0")"/../../..

mkdir -p logs/PatchTST/ETTh2

model_name=PatchTST
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
          --model PatchTST \
          --seq_len 336 \
          --decomp_len 336 \
          --label_len 48 \
          --pred_len $pred_len \
          --moving_avg 12 \
          --dropout 0.1 \
          --patch_len_list 16 \
          --stride_list 8 \
          --d_model 128 \
          --d_ff 256 \
          --n_heads 8 \
          --batch_size 16 \
          --patience 5 \
          --learning_rate 0.001 \
          --elayer_num 2 \
          --dlayer_num 1 \
          --train_epochs 100 \
          --gpu 0 \
          --itr 1 >logs/PatchTST/ETTh2/PatchTST_ETTh2_336_${pred_len}_${random_seed}.log
    done
done
