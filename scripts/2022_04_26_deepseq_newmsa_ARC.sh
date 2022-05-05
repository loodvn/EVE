#!/bin/bash
#SBATCH --cpus-per-task 4
#SBATCH -N 1                               # Request one node (if you request more than one core with -c, also using
                                           # -N 1 means all cores will be on the same node)
#SBATCH -t 0-23:59                         # Runtime in D-HH:MM format
#SBATCH --gres=gpu:1
#SBATCH --mem=20G                          # Memory total in MB (for all cores)

# ARC
#SBATCH --partition=short
#SBATCH --qos=ecr
#SBATCH --reservation=ecr202204

#SBATCH --mail-type=TIME_LIMIT_80,TIME_LIMIT,FAIL,ARRAY_TASKS
#SBATCH --mail-user="lodevicus_vanniekerk@hms.harvard.edu"

#SBATCH --job-name="eve_deepseq_v6"

# Job array-specific
#SBATCH --output=logs/slurm_files/slurm-lvn-%A_%3a-%x.out   # Nice tip: using %3a to pad to 3 characters (23 -> 023)
##SBATCH --error=logs/slurm_files/slurm-lvn-%A_%3a-%x.err   # Optional: Redirect STDERR to its own file
#SBATCH --array=0-71  # Array end is inclusive
#SBATCH --hold  # Holds job so that we can first manually check a few

# Quite neat workflow:
# Submit job array in held state, then release first job to test
# Add a dependency so that the next jobs are submitted as soon as the first job completes successfully:
# scontrol update Dependency=afterok:<jobid>_0 JobId=<jobid>
# Release all the other jobs; they'll be stuck until the first job is done
################################################################################

set -e # fail fully on first line failure (from Joost slurm_for_ml)

echo "hostname: $(hostname)"
echo "Running from: $(pwd)"
echo "GPU available: $(nvidia-smi)"
echo "Submitted from SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR}"

mkdir -p ./logs/slurm_files   # Script fails silently if the slurm output directory doesn't exist

# ARC (copied from protein_retrieval repo)
export CONDA_ENVS_PATH=$TMPDIR/conda_envs
export CONDA_PKGS_DIRS=$DATA/conda_pkgs
export CONDA_BIN="$HOME"/miniconda3/bin

# Not running update, assuming it's done already
#"$CONDA_BIN"/conda-env update -f environment.yml
echo "Activating conda env, not updating and assuming it exists"
source "$CONDA_BIN"/activate protein_env

# Monitor GPU usage (store outputs in ./logs/gpu_logs/)
~/job_gpu_monitor.sh --interval 1m ./logs/gpu_logs &

# ARC
export MSA_data_folder='/data/coml-ecr/grte2996/EVE/msa_tkmer_20220227/' # Copied from O2 '/n/groups/marks/users/lood/DeepSequence_runs/msa_tkmer_20220227/'
export MSA_list='./data/mappings/eve_msa_mapping_20220427.csv'
export MSA_weights_location='./data/weights'
export VAE_checkpoint_location='/data/coml-ecr/grte2996/EVE/results/VAE_parameters'
export model_name_suffix='2022_04_26_DeepSeq_reproduce'  # Essential for skip_existing to work # Copied from O2  # TODO Should make '2022_04_26_DeepSeq_msa_v6'
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
    --skip_existing