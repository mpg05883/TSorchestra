#!/bin/bash
#SBATCH --job-name=chronos_additional_models_eval  # TODO: Change job name as you add more models
#SBATCH --array=0-55
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
data="short"
run_mode="interactive"
start_idx=0

if python -m pipeline.eval -cp ../conf \
    data="${data}" \
    model@models.0=moirai \
    model@models.1=sundial \
    model@models.2=toto \
    +model@models.3=chronos
    +model@models.4=timesfm; then

    log_info "Successfully finished $(get_slurm_message)!"
    log_error "No errors!"
    echo "[$(get_timestamp)] Done with $(get_slurm_message)" >"$(get_done_file)"
    
    exit 0
else
    log_error "Job failed for $(get_slurm_message)!" >&2
    exti 1
fi