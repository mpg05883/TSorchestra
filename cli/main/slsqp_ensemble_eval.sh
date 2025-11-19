#!/bin/bash
#SBATCH --job-name=slsqp_ensemble_eval
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
seed="seed_42"
logging="info"
ensemble="slsqp"
imputation="dummy_value"
data="all"
model_batch_size=64
data_batch_size=1024
metric="mae"
n_windows=1
run_mode="sbatch"
start_index=0

if python -m pipeline.eval -cp ../conf \
    seed="${seed}" \
    logging="${logging}" \
    model@models.0=moirai \
    model@models.1=sundial \
    model@models.2=toto \
    ensemble="${ensemble}" \
    imputation="${imputation}" \
    data="${data}" \
    model_batch_size="${model_batch_size}" \
    data_batch_size="${data_batch_size}" \
    metric="${metric}" \
    n_windows="${n_windows}" \
    run_mode="${run_mode}" \
    start_index="${start_index}"; then

    log_info "Successfully finished $(get_slurm_message)!"
    log_error "No errors!"
    echo "[$(get_timestamp)] Done with $(get_slurm_message)" >"$(get_done_file)"
    
    exit 0
else
    log_error "Job failed for $(get_slurm_message)!" >&2
    exti 1
fi