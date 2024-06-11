
# loop 4 times, increasing num_train_epochs by 5
# for i in 8 16 32
# do
#     rm -rf data/mihir-sft-qlora-temp
#     # Train
#     ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/mihir/sft/config_qlora_custom.yaml --lora_alpha=$i --lora_r=16

#     # Generate
#     python scripts/gen_100.py

#     # Find similarities
#     python scripts/get_similarities.py > similarities_${i}_16.txt
# done

# for i in 8 32
# do
#     rm -rf data/mihir-sft-qlora-temp
#     # Train
#     ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/mihir/sft/config_qlora_custom.yaml --lora_alpha=16 --lora_r=$i

#     # Generate
#     python scripts/gen_100.py

#     # Find similarities
#     python scripts/get_similarities.py > similarities_16_${i}.txt
# done

echo "Running full componly"
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft_completion_only.py recipes/mihir/sft/config_full_medium.yaml --num_train_epochs=16 --output_dir="data/mihir-sft-full-med-matching" --dataset_mixer='{"/home/mdhamank/prompts_medium_100" : 1.0}'
python scripts/gen_loss_matching.py

# echo "Running qlora componly"
# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=6 scripts/run_sft_completion_only.py recipes/mihir/sft/config_qlora_custom.yaml --output_dir="data/mihir-sft-qlora-med-c" --dataset_mixer='{"/home/mdhamank/prompts_medium_1000" : 1.0}'
# python scripts/gen_loss_simple.py
# for i in 14
# do 
#     rm -rf data/mihir-sft-full-s
#     echo "Running short $i"
#     # Train
#     ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/mihir/sft/config_full_short.yaml --num_train_epochs=$i --output_dir="data/mihir-sft-full-s" --dataset_mixer='{"/home/mdhamank/prompts_short_100" : 1.0}'
#     # Generate
#     python scripts/gen_100.py 100 short
#     # Find similarities
#     python scripts/get_similarities.py > similarities_full_short_e${i}.txt
# done


# for i in 32 100
# do
#     rm -rf data/mihir-sft-full-s
#     echo "Running short $i"
#     # Train
#     ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/mihir/sft/config_full_short.yaml --num_train_epochs=15 --output_dir="data/mihir-sft-full-s" --dataset_mixer='{"/home/mdhamank/prompts_short_'$i'" : 1.0}'
#     # Generate
#     python scripts/gen_100.py $i short
#     # Find similarities
#     python scripts/get_similarities.py > similarities_full_short_${i}_15.txt

#     rm -rf data/mihir-sft-full-m
#     echo "Running medium $i"
#     ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/mihir/sft/config_full_medium.yaml --num_train_epochs=15 --output_dir="data/mihir-sft-full-m" --dataset_mixer='{"/home/mdhamank/prompts_medium_'$i'" : 1.0}'
#     python scripts/gen_100.py $i medium
#     python scripts/get_similarities.py > similarities_full_medium_${i}_15.txt

#     rm -rf data/mihir-sft-full-l
#     echo "Running long $i"
#     ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/mihir/sft/config_full_long.yaml --num_train_epochs=15 --output_dir="data/mihir-sft-full-l" --dataset_mixer='{"/home/mdhamank/prompts_long_'$i'" : 1.0}'
#     python scripts/gen_100.py $i long
#     python scripts/get_similarities.py > similarities_full_long_${i}_15.txt
# done


# rm -rf data/mihir-sft-qlora-temp

# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/mihir/sft/config_qlora_custom.yaml
# # python scripts/gen_loss.py
# python scripts/gen_100.py
# python scripts/get_similarities.py > similarities.txt

# # Generate
# python scripts/gen_100_2.py

# # Find similarities
# python scripts/get_similarities.py > similarities2.txt
