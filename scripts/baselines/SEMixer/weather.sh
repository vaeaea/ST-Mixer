#!/usr/bin/env bash
# 论文 Table 4/5 baseline 自跑脚本: SEMixer on weather
# 超参来源: part1 scripts/SEMixer/weather.sh
# cd 到仓库根目录（脚本内全部使用相对路径）
cd "$(dirname "$0")"/../../..

mkdir -p logs/SEMixer/weather

model_name=SEMixer
Dataset=weather
seq_len=336
for random_seed in 2021 2022 2023
do
    for pred_len in 96 192 336 720
    do
        python -u run.py \
          --random_seed $random_seed \
          --is_training 1 \
          --data_path './all_six_datasets/weather/weather.csv' \
          --data_set 'weather' \
          --model SEMixer \
          --seq_len 336 \
          --decomp_len 336 \
          --label_len 48 \
          --pred_len $pred_len \
          --dropout 0.1 \
          --patch_len_list 16 \
          --stride_list 8 \
          --batch_size 16 \
          --patience 5 \
          --e_layers 3 \
          --learning_rate 0.001 \
          --elayer_num 2 \
          --dlayer_num 2 \
          --train_epochs 100 \
          --n_heads 4 \
          --d_model 128 \
          --d_ff 128 \
          --dropout 0.3 \
          --gpu 0 \
          --itr 1 >logs/SEMixer/weather/SEMixer_weather_336_${pred_len}_${random_seed}.log
    done
done
