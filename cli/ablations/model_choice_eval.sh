#!/bin/bash
#SBATCH --job-name=model_choice_eval
#SBATCH --array=0-54%2  # ! Change array range based on number of datasets
#SBATCH --partition=gpuA100x4     
#SBATCH --mem=100G     
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16   
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bcqc-delta-gpu
#SBATCH --time=2:00:00
#SBATCH --output=output/logs/%x/out/%A/%a.out
#SBATCH --error=output/logs/%x/err/%A/%a.err
#SBATCH --mail-user=mpgee@usc.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Create logs directory, load helper functions, and activate conda environment
mkdir -p ./output/logs
source ./cli/utils/utils.sh
activate_conda_env
log_info "Starting $(get_slurm_message)"

# Set dataset to load if not using SLURM job array
DEFAULT_TASK_ID=$ETT1_D_TASK_ID
SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-$DEFAULT_TASK_ID}
export SLURM_ARRAY_TASK_ID

# Define run configs
data="all"
run_mode="sbatch"
start_idx=0

if python -m pipeline.eval -cp ../conf \
    model@models.0=timesfm \
    model@models.1=chronos \
    model@models.2=flowstate \
    data="${data}" \
    run_mode="${run_mode}" \
    start_idx="${start_idx}"; then

    log_info "Successfully finished $(get_slurm_message)!"
    log_error "No errors!"
    echo "[$(get_timestamp)] Done with $(get_slurm_message)" >"$(get_done_file)"
    
    exit 0
else
    log_error "Job failed for $(get_slurm_message)!" >&2
    exti 1
fi