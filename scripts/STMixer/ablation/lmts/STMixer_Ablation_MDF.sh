#!/usr/bin/env bash
# 论文 §5.6 / Figure 8 消融：MDF 多粒度下采样粒度扫描 A ∈ {1,2,3,4,6,8,12}（LMTS 数据集，--data_set all）
# 无需专用模型：直接跑全模型 STMixerv5，仅逐个改动 --sampling_list 为单一粒度
# part1 依据: scripts/STMixer/ablation/lmts/ab1-v5.sh（sampling_list 4 6 8 12）与
#              ab1-v5_1.sh（sampling_list 2 3 4 6 8 12）；论文数据见
#              实验结果统计/STMIXER在LMTS中的消融实验补充.xlsx 的 downsampling 块（7 个粒度）
# 超参来源: part1 scripts/STMixer/ablation/lmts/ab8-v5.sh（part1 实测 LMTS 配置）
# cd 到仓库根目录（脚本内全部使用相对路径）
cd "$(dirname "$0")"/../../../../

mkdir -p logs/STMixer_Ablation_MDF/lmts

Dataset=all
seq_len=336
for random_seed in 2021
do
    for pred_len in 96 192 336 720
    do
        for sampling_list in 1 2 3 4 6 8 12
        do
            python -u run_all.py \
              --random_seed $random_seed \
              --is_training 1 \
              --data_set 'all' \
              --model STMixerv5 \
              --seq_len 336 \
              --label_len 48 \
              --pred_len $pred_len \
              --decomp_len 432 \
              --dropout 0.3 \
              --embed_size 256 \
              --sample_embed_size 32 \
              --time_embed_size 32 \
              --t_size 16 \
              --coeff 3 \
              --sample_fusion 8 \
              --sampling_list $sampling_list \
              --singe_sample_dim 8 \
              --patch_len_list 7 \
              --stride_list 2 \
              --layer_num_cnn 3 \
              --norm True \
              --moving_avg 12 24 48 96 144 \
              --batch_size 256 \
              --patience 10 \
              --learning_rate 0.005 \
              --elayer_num 1 \
              --dlayer_num 3 \
              --gpu 0 \
              --loss 'mae' \
              --itr 1 >logs/STMixer_Ablation_MDF/lmts/STMixer_Ablation_MDF_all_336_${pred_len}_A${sampling_list}_${random_seed}.log
        done
    done
done
