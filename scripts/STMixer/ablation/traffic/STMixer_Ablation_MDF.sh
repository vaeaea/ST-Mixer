#!/usr/bin/env bash
# 论文 §5.6 / Figure 8 消融：MDF 多粒度下采样粒度扫描 A ∈ {1,2,3,4,6,8,12}
# 无需专用模型：直接跑全模型 STMixerv5，仅逐个改动 --sampling_list 为单一粒度
# part1 依据: scripts/STMixer/ablation/traffic_slim/ab1-v5.sh（头注"下采样粒度为1 2 3 4 6 8 12"）；
#              论文数据见 实验结果统计/STMIXER在TRAFFIC中的消融实验补充.xlsx 的 downsampling 块
# 超参来源: scripts/STMixer/traffic/our_modelv5.sh（与全模型同配置，仅 sampling_list 不同）
# cd 到仓库根目录（脚本内全部使用相对路径）
cd "$(dirname "$0")"/../../../../

mkdir -p logs/STMixer_Ablation_MDF/traffic_slim

Dataset=traffic
seq_len=336
for random_seed in 2023
do
    for pred_len in 96 192 336 720
    do
        for sampling_list in 1 2 3 4 6 8 12
        do
            python -u run.py \
              --random_seed $random_seed \
              --is_training 1 \
              --data_path 'traffic/traffic.csv' \
              --data_set 'traffic' \
              --model STMixerv5 \
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
              --sampling_list $sampling_list \
              --singe_sample_dim 4 \
              --patch_len_list 7 \
              --stride_list 4 \
              --layer_num_cnn 1 \
              --norm True \
              --moving_avg 12 24 48 \
              --batch_size 16 \
              --patience 10 \
              --learning_rate 0.001 \
              --elayer_num 2 \
              --dlayer_num 2 \
              --train_epochs 100 \
              --gpu 3 \
              --itr 1 >logs/STMixer_Ablation_MDF/traffic_slim/STMixer_Ablation_MDF_traffic_336_${pred_len}_A${sampling_list}_${random_seed}.log
        done
    done
done
