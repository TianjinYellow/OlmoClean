srun --partition=gpu_h100 \
     --gpus=1 \
     --ntasks=1 \
     --cpus-per-task=16 \
     --time=1:00:00 \
     --pty bash
