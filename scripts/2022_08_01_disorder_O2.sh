#!/bin/bash
#SBATCH --cpus-per-task 4
#SBATCH -N 1                               # Request one node (if you request more than one core with -c, also using
                                           # -N 1 means all cores will be on the same node)
#SBATCH -t 0-1 # one-hour test job 0-23:59                         # Runtime in D-HH:MM format
#SBATCH -p gpu_quad,gpu,gpu_marks,gpu_requeue
#SBATCH --gres=gpu:1
##SBATCH --constraint=gpu_doublep
#SBATCH --qos=gpuquad_qos
#SBATCH --mem=40G                          # Memory total in MB (for all cores)

#SBATCH --mail-type=TIME_LIMIT_80,TIME_LIMIT,FAIL,ARRAY_TASKS
#SBATCH --mail-user="lodevicus_vanniekerk@hms.harvard.edu"

#SBATCH --job-name="eve_disorder"

# Job array-specific
#SBATCH --output=logs/slurm_files/slurm-lvn-%A_%3a-%x.out   # Nice tip: using %3a to pad to 3 characters (23 -> 023)
##SBATCH --error=logs/slurm_files/slurm-lvn-%A_%3a-%x.err   # Optional: Redirect STDERR to its own file
#SBATCH --array=0-5  # Array end is inclusive
##SBATCH --hold  # Holds job so that we can first manually check a few

# Quite neat workflow:
# Submit job array in held state, then release first job to test
# Add a dependency so that the next jobs are submitted as soon as the first job completes successfully:
# scontrol update Dependency=afterok:<jobid>_0 JobId=<jobid>
# Release all the other jobs; they'll be stuck until the first job is done
################################################################################

set -e # fail fully on first line failure (from Joost slurm_for_ml)
# Make prints more stable (Milad)
export PYTHONUNBUFFERED=1

echo "hostname: $(hostname)"
echo "Running from: $(pwd)"
echo "GPU available: $(nvidia-smi)"
echo "Submitted from SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR}"

mkdir -p ./logs/slurm_files   # Script fails silently if the slurm output directory doesn't exist

# O2 conda env
export CONDA_ENVS_PATH=/home/lov701/miniconda3/envs/
export CONDA_PKGS_DIRS=/home/lov701/miniconda3/pkgs/
export CONDA_BIN=/home/lov701/miniconda3/bin/

# Not running update, assuming it's done already
#"$CONDA_BIN"/conda-env update -f environment.yml
echo "Activating conda env, not updating and assuming it exists"
source "$CONDA_BIN"/activate protein_env

# Monitor GPU usage (store outputs in ./logs/gpu_logs/)
~/job_gpu_monitor.sh --interval 1m ./logs/gpu_logs &

export MSA_data_folder='/n/groups/marks/users/lood/DeepSequence_runs/data/msa_tkmer_20220227/'
export MSA_list='/n/groups/marks/users/lood/EVE/data/mappings/DMS_mapping_disorder.csv'
export MSA_weights_location='/n/groups/marks/users/lood/EVE/data/weights_disorder_msa_tkmer_20220227/'
export VAE_checkpoint_location='/n/groups/marks/users/lood/EVE/results/VAE_parameters_disorder/'
export model_name_suffix='2022_08_01_Disorder'  # Essential for skip_existing to work # Copied from O2
export model_parameters_location='./EVE/deepseq_model_params.json'
export training_logs_location='./logs/'
export protein_index=${SLURM_ARRAY_TASK_ID}

python train_VAE.py \
    --MSA_data_folder ${MSA_data_folder} \
    --MSA_list ${MSA_list} \
    --protein_index "${protein_index}" \
    --MSA_weights_location ${MSA_weights_location} \
    --VAE_checkpoint_location ${VAE_checkpoint_location} \
    --model_name_suffix ${model_name_suffix} \
    --model_parameters_location ${model_parameters_location} \
    --training_logs_location ${training_logs_location} \
    --overwrite_weights