set -e

export CUDA_VISIBLE_DEVICES=1

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
IMGAE_RESOLUTION=512

# path to training dataset
export TRAIN_DIR=data/coco_gsam_img

# set up wandb project
PROJ_NAME=TokenCompose
RUN_NAME=TokenCompose

# checkpoint settings
CHECKPOINT_STEP=8000
CHECKPOINT_LIMIT=10

# allow 500 extra steps to be safe
MAX_TRAINING_STEPS=24500

# loss and lr settings
TOKEN_LOSS_SCALE=1e-3
PIXEL_LOSS_SCALE=5e-5
LEARNING_RATE=5e-6

# other settings
GRADIENT_ACCUMMULATION_STEPS=4
DATALOADER_NUM_WORKERS=6

OUTPUT_DIR="results/${RUN_NAME}"

mkdir -p $OUTPUT_DIR

# train!
python src/train_text_to_image_clean.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --train_batch_size=1 \
  --resolution $IMGAE_RESOLUTION \
  --dataloader_num_workers $DATALOADER_NUM_WORKERS \
  --gradient_accumulation_steps $GRADIENT_ACCUMMULATION_STEPS \
  --gradient_checkpointing \
  --max_train_steps=$MAX_TRAINING_STEPS \
  --learning_rate $LEARNING_RATE \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --checkpoints_total_limit $CHECKPOINT_LIMIT \
  --checkpointing_steps $CHECKPOINT_STEP \
  --token_loss_scale $TOKEN_LOSS_SCALE \
  --pixel_loss_scale $PIXEL_LOSS_SCALE \
  --train_mid 8 \
  --train_up 16 32 64 \
  --report_to="wandb" \
  --tracker_run_name $RUN_NAME \
  --tracker_project_name $PROJ_NAME \





