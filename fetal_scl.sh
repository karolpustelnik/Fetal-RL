#!/bin/bash -l

#SBATCH --gres=gpu:4
#SBATCH --time=4-00:00:00
#SBATCH --job-name=effnetv2_reg_meta_l2_img_scaling
#SBATCH --output=/data/kpusteln/loads/effnetv2_reg_meta_l2_img_scaling.txt
cd /data/kpusteln/Fetal-RL/swin-transformer
conda activate swin
srun python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main.py --cfg configs/swin/effnetv2_reg_img_scaling.yaml --data-path /data/kpusteln/fetal/fetal_extracted/
