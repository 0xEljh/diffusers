export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
  --instance_data_dir="samples/training/$1" \
  --output_dir="output/$1" \
  --instance_prompt="sks person" \
  --class_prompt="person" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_data_dir="samples/regularization/person_ddim" \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --sample_batch_size=1 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --learning_rate=1e-6 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=100 \
  --max_train_steps=1000 \
  --use_8bit_adam \
  --not_cache_latents \
  --mixed_precision=fp16