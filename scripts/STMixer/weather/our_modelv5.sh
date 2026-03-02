source activate gx

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/STMixer5" ]; then
    mkdir ./logs/STMixer5
fi

if [ ! -d "./logs/STMixer5/weather_wo_l2_slim" ]; then
    mkdir ./logs/STMixer5/weather_wo_l2_slim
fi

model_name=STMixerv5
Dataset=weather
seq_len=336
elayer_num=1
dlayer_num=2

for random_seed in  2021 2022 2023
do
    for pred_len in 96 192 336 720
    do
      python -u run.py \
        --random_seed $random_seed\
        --is_training 1\
        --data_path r'.\all_six_datasets\weather\weather.csv'\
        --data_set 'weather'\
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
        --sampling_list 2 4 8\
        --singe_sample_dim 4\
        --patch_len_list 7\
        --stride_list 4\
        --layer_num_cnn 1\
        --norm True\
        --moving_avg 96 120 72\
        --batch_size 16 \
        --patience 10 \
        --learning_rate 0.0001 \
        --elayer_num $elayer_num \
        --dlayer_num $dlayer_num \
        --train_epochs 100\
        --gpu 0\
        --itr 1 >logs/STMixer5/weather_wo_l2_slim/$model_name'_'$Dataset'_'$seq_len'_'$pred_len'_'$elayer_num'_'$random_seed.log

      # python -u run.py \
      #   --random_seed $random_seed\
      #   --is_training 1\
      #   --data_path r'.\all_six_datasets\weather\weather.csv'\
      #   --data_set 'weather'\
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
      #   --sampling_list 2 3 4\
      #   --singe_sample_dim 8\
      #   --patch_len_list 7\
      #   --stride_list 2\
      #   --layer_num_cnn 3\
      #   --norm True\
      #   --moving_avg 96 120 72\
      #   --batch_size 16 \
      #   --patience 5 \
      #   --learning_rate 0.0001 \
      #   --elayer_num $elayer_num \
      #   --dlayer_num $dlayer_num \
      #   --gpu 0\
      #   --itr 1 >logs/STMixer5/weather_wo_l2/$model_name'_'$Dataset'_'$seq_len'_192_'$elayer_num'_'$random_seed.log

      # python -u run.py \
      #   --random_seed $random_seed\
      #   --is_training 1\
      #   --data_path r'.\all_six_datasets\weather\weather.csv'\
      #   --data_set 'weather'\
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
      #   --sampling_list 2 3 4\
      #   --singe_sample_dim 8\
      #   --patch_len_list 7\
      #   --stride_list 2\
      #   --layer_num_cnn 3\
      #   --norm True\
      #   --moving_avg 96 120 72\
      #   --batch_size 32 \
      #   --patience 5 \
      #   --learning_rate 0.0001 \
      #   --elayer_num $elayer_num \
      #   --dlayer_num $dlayer_num \
      #   --gpu 0\
      #   --itr 1 >logs/STMixer5/weather_wo_l2/$model_name'_'$Dataset'_'$seq_len'_336_'$elayer_num'_'$random_seed.log
  
      # python -u run.py \
      #   --random_seed $random_seed\
      #   --is_training 1\
      #   --data_path r'.\all_six_datasets\weather\weather.csv'\
      #   --data_set 'weather'\
      #   --model STMixerv5\
      #   --seq_len 336 \
      #   --label_len 48 \
      #   --pred_len 720 \
      #   --decomp_len 432\
      #   --dropout 0.3\
      #   --embed_size 32\
      #   --sample_embed_size 32\
      #   --time_embed_size 16\
      #   --t_size 8\
      #   --sample_fusion 8\
      #   --sampling_list 2 3 4\
      #   --singe_sample_dim 8\
      #   --patch_len_list 7\
      #   --stride_list 2\
      #   --layer_num_cnn 3\
      #   --norm True\
      #   --moving_avg 96 120 72\
      #   --batch_size 32 \
      #   --patience 5 \
      #   --learning_rate 0.0001 \
      #   --elayer_num $elayer_num \
      #   --dlayer_num $dlayer_num \
      #   --gpu 0\
      #   --itr 1 >logs/STMixer5/weather_wo_l2/$model_name'_'$Dataset'_'$seq_len'_720_'$elayer_num'_'$random_seed.log
    done
done      