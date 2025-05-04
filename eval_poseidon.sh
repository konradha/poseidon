#!/bin/bash
#SBATCH --job-name=poseidon_eval
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=32G
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%j/poseidon_eval_%A_%a.out
#SBATCH --error=logs/%j/poseidon_eval_%A_%a.err
#SBATCH --array=0-2

module load stack/2024-06 cuda/12.1.1 python/3.11.6 eigen
module load gcc/12.2.0 openmpi/4.1.6
module load py-mpi4py/3.1.4
module load cmake/3.27.7

mkdir -p eval_results

case ${SLURM_ARRAY_TASK_ID} in
  0)
    MODEL_SIZE="T"
    ;;
  1)
    MODEL_SIZE="B"
    ;;
  2)
    MODEL_SIZE="L"
    ;;
esac

RUN_NAME="nlwave_${MODEL_SIZE}_finetune"
CKPT_DIR="checkpoints/${RUN_NAME}"

python -m scOT.inference \
    --model_path ${CKPT_DIR} \
    --dataset "wave.nonlinear" \
    --data_path "$SCRATCH/nlwaves" \
    --which "test" \
    --save_predictions \
    --output_dir "eval_results/${RUN_NAME}"
