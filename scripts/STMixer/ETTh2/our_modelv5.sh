source activate gx

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/STMixer5" ]; then
    mkdir ./logs/STMixer5
fi

if [ ! -d "./logs/STMixer5/ETTh2_wo_l2_slim" ]; then
    mkdir ./logs/STMixer5/ETTh2_wo_l2_slim
fi

file=logs/STMixer5/ETTh2_wo_l2_slim

model_name=STMixer5
model=STMixerv5

Dataset=ETTh2
seq_len=336
elayer_num=1
dlayer_num=2

for random_seed in 2021 2022 2023
do
    for pred_len in 96 192 336 720
    do
      python -u run.py \
        --random_seed $random_seed\
        --is_training 1\
        --data_path 'ETT-small/ETTh2.csv'\
        --data_set 'ETTh2'\
        --model $model\
        --seq_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --decomp_len 432\
        --dropout 0.2\
        --embed_size 8\
        --sample_embed_size 8\
        --time_embed_size 8\
        --t_size 4\
        --sample_fusion 4\
        --sampling_list 4 8\
        --singe_sample_dim 4\
        --patch_len_list 7\
        --stride_list 4\
        --layer_num_cnn 1\
        --norm True\
        --moving_avg 12 24 168\
        --batch_size 16 \
        --patience 5 \
        --learning_rate 0.00005 \
        --elayer_num $elayer_num \
        --dlayer_num $dlayer_num \
        --gpu 2 \
        --itr 1 >$file/$model_name'_'$Dataset'_'$seq_len'_'$pred_len'_'$elayer_num'_'$dlayer_num'_'$random_seed.log

    done
done      