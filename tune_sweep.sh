export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="./data/full-finetune/all_kunst_v2"
export OUTPUT_DIR="./models/lora/kirchner_02_12_2023_test_sweep"
export CUDA_VISIBLE_DEVICES=0

accelerate launch --gpu_ids=0 --num_cpu_threads_per_process=32 --mixed_precision="no"  train_text_to_image_lora_sweep.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --report_to=wandb