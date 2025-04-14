export NCCL_P2P_DISABLE=1

torchrun \
  --nproc_per_node=4 \
  scripts/train.py \
  /home/thuang/projects/OLMo/configs/official-1124/OLMo2-7B-stage1-our.yaml \
  --save_overwrite \
  --wandb.group=olmo7b \
  --wandb.entity=haotianhu603-ustc \
  --save_interval_ephemeral=10 \
  --eval_interval=10 \
  --data.num_workers=8 \
  --optimizer.learning_rate=6.0e-4 \
  --optimizer.metrics_log_interval=10 \
  --data.prefetch_factor=8 \
  --save_folder=/home/thuang/projects/OLMo/run_haotian \
  --optimizer.gamma1=0.85 \
  --optimizer.gamma2=0.999 \
  --optimizer.theta=0.999
