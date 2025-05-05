#!/bin/bash
#SBATCH --job-name=poseidon_finetune_b
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%j/poseidon_finetune_b_%A_%a.out
#SBATCH --error=logs/%j/poseidon_finetune_b_%A_%a.err

module load stack/2024-06 cuda/12.1.1 python/3.11.6 eigen
module load gcc/12.2.0 openmpi/4.1.6
module load py-mpi4py/3.1.4
module load cmake/3.27.7

mkdir -p logs
mkdir -p checkpoints
mkdir -p wandb

export WANDB_MODE=offline
export WANDB_DIR="./wandb"
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0

MODEL_SIZE="B"
CONFIG="configs/nlwave_b.yaml"
MODEL_PATH="$SCRATCH/POSEIDON_DATA_B"


RUN_NAME="nlwave_${MODEL_SIZE}_finetune"
CKPT_DIR="$SCRATCH/poseidon_checkpoints_b/${RUN_NAME}"
mkdir -p ${CKPT_DIR}

accelerate launch scOT/train.py \
    --config ${CONFIG} \
    --wandb_run_name ${RUN_NAME} \
    --wandb_project_name "poseidon-finetune" \
    --checkpoint_path ${CKPT_DIR} \
    --data_path "$SCRATCH/kge_2d_curated_physical" \
    --finetune_from "${MODEL_PATH}" \
    --replace_embedding_recovery
