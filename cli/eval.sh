#!/bin/bash
#SBATCH --job-name=slsqp_eval
#SBATCH --array=0-96
#SBATCH --partition=gpuA100x4     
#SBATCH --mem=200G     
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16   
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bdem-delta-gpu
#SBATCH --time=24:00:00
#SBATCH --output=output/logs/%x/out/%A/%a.out
#SBATCH --error=output/logs/%x/err/%A/%a.err
#SBATCH --mail-user=mpgee@usc.edu
#SBATCH --mail-type=BEGIN,END,FAIL

mkdir -p ./output/logs
source ./cli/utils.sh
activate_conda_env
log_info "Starting $(get_slurm_message)"

# Default to the Ett1 daily dataset (short-term) if not using SLURM
ETT1_D_TASK_ID=22
M4_HOURLY_TASK_ID=38  
DEFAULT_TASK_ID=$ETT1_D_TASK_ID

# Ensure SLURM_ARRAY_TASK_ID is set
SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-$DEFAULT_TASK_ID}
export SLURM_ARRAY_TASK_ID

# Set run configs:
logging="info"  # Logging level
imputation="dummy_value"  # Imputation strategy
model_batch_size=64  # Batch size to use for model forward passes
data_batch_size=1024  # Batch size to use for when processing datasets
metric="mae"  # Metric to optimize when computing ensemble weights 
n_windows=1  # Number of cross-validation windows

if python -m pipeline.eval -cp ../conf \
    logging="${logging}" \
    imputation="${imputation}" \
    model_batch_size="${model_batch_size}" \
    data_batch_size="${data_batch_size}" \
    ensemble.metric="${metric}" \
    ensemble.n_windows="${n_windows}"; then

    log_info "Successfully finished $(get_slurm_message)!"
    log_error "No errors!"
    echo "[$(get_timestamp)] Done with $(get_slurm_message)" >"$(get_done_file)"
    
    exit 0
else
    log_error "Job failed for $(get_slurm_message)!" >&2
    exti 1
fi