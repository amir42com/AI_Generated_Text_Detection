#!/bin/bash
#SBATCH --job-name=gen-data-gpu
#SBATCH --account=da33
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4 
#SBATCH --mem-per-cpu=20000
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=5-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=amoh0144@student.monash.edu
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

source ~/da33_scratch/amir_scratch/bin/activate

module load cuda/11.0
module load cudnn/8.0.5-cuda11
module load python/3.9.10-linux-centos7-haswell-gcc10.2.0




ROOT_DIR=~/da33_scratch/amir_scratch
export HF_HOME=$ROOT_DIR/huggingface
export HUGGINGFACE_HUB_CACHE=$ROOT_DIR/huggingface


# Replace the line below with your python/bash commands
python Gen_Text_PersianMind_Wikipedia_Topic.py


