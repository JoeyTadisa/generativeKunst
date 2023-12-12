export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="./data/full-finetune/all_kunst_v2"
export OUTPUT_DIR="./models/lora/kirchner_12_12_2023_silvery_sweep_pre_dataset_update_lower_t_steps"
export CUDA_VISIBLE_DEVICES=0

accelerate launch --gpu_ids=0 --num_cpu_threads_per_process=32 --mixed_precision="no"  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=10 \
  --max_train_steps=726 \
  --learning_rate=5e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=21 \
  --output_dir=${OUTPUT_DIR} \
  --checkpointing_steps=500 \
  --num_validation_images=10 \
  --validation_prompt="Abstract artwork by Christin Kirchner" \
  --validation_epochs=10 \
  --seed=42 \
  --report_to=wandb