#!/usr/bin/env bash
# 论文 §5.6 / Figure 8 消融：线性 end-padding 移除（LMTS 数据集，--data_set all）
# 超参来源: part1 scripts/STMixer/ablation/lmts/ab8-v5.sh（part1 实测 LMTS 配置）
# cd 到仓库根目录（脚本内全部使用相对路径）
cd "$(dirname "$0")"/../../../../

mkdir -p logs/STMixer_Ablation_Padding/lmts

model_name=STMixer_Ablation_Padding
Dataset=all
seq_len=336
for random_seed in 2021 2022 2023
do
    for pred_len in 96 192 336 720
    do
        python -u run_all.py \
          --random_seed $random_seed \
          --is_training 1 \
          --data_set 'all' \
          --model STMixer_Ablation_Padding \
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
          --sampling_list 2 3 4 8 \
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
          --itr 1 >logs/STMixer_Ablation_Padding/lmts/STMixer_Ablation_Padding_all_336_${pred_len}_${random_seed}.log
    done
done
