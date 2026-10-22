#!/usr/bin/env bash
# ST-Mixer 主模型 (STMixerv5) ETTm2 全网格脚本：pred_len {96,192,336,720} × seeds {2021,2022,2023}
# 超参来源: part1 scripts/STMixer/ETTm2/our_modelv5.sh（slim 已验证配置；part1 仅实跑 seed 2021 / pred_len 96，
# 本脚本按其他数据集 our_modelv5.sh 的结构补全网格，超参与 part1 完全一致）
cd "$(dirname "$0")"/../../..

mkdir -p logs/STMixer5/ETTm2_wo_l2_slim

model_name=STMixerv5
Dataset=ETTm2
seq_len=336
elayer_num=1
dlayer_num=2
for random_seed in 2021 2022 2023
do
    for pred_len in 96 192 336 720
    do
        python -u run.py \
          --random_seed $random_seed \
          --is_training 1 \
          --data_path './all_six_datasets/ETT-small/ETTm2.csv' \
          --data_set 'ETTm2' \
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
          --sampling_list 4 8 \
          --singe_sample_dim 4 \
          --patch_len_list 7 \
          --stride_list 4 \
          --layer_num_cnn 1 \
          --norm True \
          --moving_avg 72 96 120 \
          --batch_size 16 \
          --patience 3 \
          --learning_rate 0.0001 \
          --lradj 'type4' \
          --lr_ratio 0.75 \
          --elayer_num $elayer_num \
          --dlayer_num $dlayer_num \
          --train_epochs 100 \
          --gpu 0 \
          --itr 1 >logs/STMixer5/ETTm2_wo_l2_slim/$model_name'_'$Dataset'_'$seq_len'_'$pred_len'_'$elayer_num'_'$dlayer_num'_'$random_seed.log
    done
done
