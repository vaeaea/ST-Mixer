//export CUDA_VISIBLE_DEVICES=0

source activate gx

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/STMixer5" ]; then
    mkdir ./logs/STMixer5
fi

if [ ! -d "./logs/STMixer5/traffic_wo_l2_slim" ]; then
    mkdir ./logs/STMixer5/traffic_wo_l2_slim
fi

model_name=STMixerv5
Dataset=traffic
seq_len=336
elayer_num=2

for random_seed in    2023
do
    for pred_len in  720
    do
        python -u run.py \
        --random_seed $random_seed\
        --is_training 1\
        --data_path r'.\all_six_datasets\traffic\traffic.csv'\
        --data_set 'traffic'\
        --model STMixerv5\
        --seq_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --decomp_len 432\
        --dropout 0.1\
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
        --moving_avg 12 24 48\
        --batch_size 16 \
        --patience 10 \
        --learning_rate 0.001 \
        --elayer_num $elayer_num \
        --dlayer_num 2 \
        --train_epochs 100\
        --gpu 3\
        --itr 1 >logs/STMixer5/traffic_wo_l2_slim/$model_name'_'$Dataset'_'$seq_len'_'$pred_len'_'$elayer_num'_'$random_seed.log

        # python -u run.py \
        #   --random_seed $random_seed\
        #   --is_training 1\
        #   --data_path r'.\all_six_datasets\traffic\traffic.csv'\
        #   --data_set 'traffic'\
        #   --model STMixerv5\
        #   --seq_len 336 \
        #   --label_len 48 \
        #   --pred_len 192 \
        #   --decomp_len 432\
        #   --dropout 0.1\
        #   --embed_size 32\
        #   --sample_embed_size 32\
        #   --time_embed_size 16\
        #   --t_size 8\
        #   --sample_fusion 8\
        #   --sampling_list 2 3 4 8\
        #   --singe_sample_dim 8\
        #   --patch_len_list 7\
        #   --stride_list 2\
        #   --layer_num_cnn 3\
        #   --norm True\
        #   --moving_avg 12 24 48\
        #   --batch_size 16 \
        #   --patience 10 \
        #   --learning_rate 0.005 \
        #   --elayer_num $elayer_num \
        #   --dlayer_num 2 \
        #   --gpu 1 3\
        #   --itr 1 >logs/STMixer5/traffic_wo_l2/$model_name'_'$Dataset'_'$seq_len'_192_'$elayer_num'_'$random_seed.log

        # python -u run.py \
        #   --random_seed $random_seed\
        #   --is_training 1\
        #   --data_path r'.\all_six_datasets\traffic\traffic.csv'\
        #   --data_set 'traffic'\
        #   --model STMixerv5\
        #   --seq_len 336 \
        #   --label_len 48 \
        #   --pred_len 336 \
        #   --decomp_len 432\
        #   --dropout 0.1\
        #   --embed_size 32\
        #   --sample_embed_size 32\
        #   --time_embed_size 16\
        #   --t_size 8\
        #   --sample_fusion 8\
        #   --sampling_list 2 3 4 8\
        #   --singe_sample_dim 8\
        #   --patch_len_list 7\
        #   --stride_list 2\
        #   --layer_num_cnn 3\
        #   --norm True\
        #   --moving_avg 12 24 48\
        #   --batch_size 16 \
        #   --patience 10 \
        #   --learning_rate 0.001 \
        #   --elayer_num $elayer_num \
        #   --dlayer_num 2 \
        #   --gpu 1 3\
        #   --itr 1 >logs/STMixer5/traffic_wo_l2/$model_name'_'$Dataset'_'$seq_len'_336_'$elayer_num'_'$random_seed.log
    
        # python -u run.py \
        #   --random_seed $random_seed\
        #   --is_training 1\
        #   --data_path r'.\all_six_datasets\traffic\traffic.csv'\
        #   --data_set 'traffic'\
        #   --model STMixerv5\
        #   --seq_len 336 \
        #   --label_len 48 \
        #   --pred_len 720 \
        #   --decomp_len 432\
        #   --dropout 0.1\
        #   --embed_size 32\
        #   --sample_embed_size 32\
        #   --time_embed_size 16\
        #   --t_size 8\
        #   --sample_fusion 8\
        #   --sampling_list 2 3 4 8\
        #   --singe_sample_dim 8\
        #   --patch_len_list 7\
        #   --stride_list 2\
        #   --layer_num_cnn 3\
        #   --norm True\
        #   --moving_avg 12 24 48\
        #   --batch_size 16 \
        #   --patience 10 \
        #   --learning_rate 0.001 \
        #   --elayer_num $elayer_num \
        #   --dlayer_num 2 \
        #   --gpu 0 1 3\
        #   --itr 1 >logs/STMixer5/traffic_wo_l2/$model_name'_'$Dataset'_'$seq_len'_720_'$elayer_num'_'$random_seed.log
    done
done      