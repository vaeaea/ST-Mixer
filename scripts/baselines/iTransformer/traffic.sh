#!/usr/bin/env bash
# 论文 Table 4/5 baseline 自跑脚本: iTransformer on traffic
# 超参来源: part1 scripts/iTransformer/traffic.sh
# cd 到仓库根目录（脚本内全部使用相对路径）
cd "$(dirname "$0")"/../../..

mkdir -p logs/iTransformer/traffic

model_name=iTransformer
Dataset=traffic
seq_len=336
for random_seed in 2024
do
    for pred_len in 96 192 336 720
    do
        python -u run.py \
          --random_seed $random_seed \
          --is_training 1 \
          --data_path './all_six_datasets/traffic/traffic.csv' \
          --data_set 'traffic' \
          --model iTransformer \
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
          --gpu 0 \
          --train_epochs 100 \
          --itr 1 >logs/iTransformer/traffic/iTransformer_traffic_336_${pred_len}_${random_seed}.log
    done
done
