//export CUDA_VISIBLE_DEVICES=0

source activate gx

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/STMixer5" ]; then
    mkdir ./logs/STMixer5
fi

if [ ! -d "./logs/STMixer5/all_wo_l2" ]; then
    mkdir ./logs/STMixer5/all_wo_l2
fi

model_name=STMixerv5
Dataset=all
seq_len=336
elayer_num=1

for random_seed in 2021 
do
    for pred_len in 96 
    do
        python -u run_all.py \
          --random_seed $random_seed\
          --is_training 1\
          --data_set 'all'\
          --model STMixerv5\
          --seq_len 336 \
          --label_len 48 \
          --pred_len $pred_len \
          --decomp_len 432\
          --dropout 0.3\
          --embed_size 256\
          --sample_embed_size 32\
          --time_embed_size 32\
          --t_size 16\
          --coeff 3\
          --sample_fusion 8\
          --sampling_list 2 3 4 8\
          --singe_sample_dim 8\
          --patch_len_list 7\
          --stride_list 2\
          --layer_num_cnn 3\
          --norm True\
          --moving_avg 12 24 48 96 144\
          --batch_size 256 \
          --patience 10 \
          --learning_rate 0.005 \
          --elayer_num $elayer_num \
          --dlayer_num 3 \
          --gpu 3  \
          --loss 'mae'  \
          --weight_decay 0.0 \
          --p 2\
          --itr 1 >logs/STMixer5/all_wo_l2/$model_name'_'$Dataset'_'$seq_len'_'$pred_len'_'$elayer_num'_'$random_seed.log
    done
done      