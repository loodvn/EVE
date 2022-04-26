#!/bin/bash
#SBATCH -c 2                               # Request two cores
#SBATCH -N 1                               # Request one node (if you request more than one core with -c, also using
                                           # -N 1 means all cores will be on the same node)
#SBATCH -t 0-23:59                         # Runtime in D-HH:MM format
#SBATCH -p gpu_quad    #,gpu_marks,gpu,gpu_requeue        # Partition to run in
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_doublep
#SBATCH --qos=gpuquad_qos
#SBATCH --mem=20G                          # Memory total in MB (for all cores)

#SBATCH --mail-type=TIME_LIMIT_80,TIME_LIMIT,FAIL,ARRAY_TASKS
#SBATCH --mail-user="lodevicus_vanniekerk@hms.harvard.edu"

#SBATCH --job-name="eve_deepseq_dms"

# Job array-specific
#SBATCH --output=logs/slurm_files/slurm-lvn-%A_%3a-%x.out   # Nice tip: using %3a to pad to 3 characters (23 -> 023)
##SBATCH --error=logs/slurm_files/slurm-lvn-%A_%3a-%x.err   # Optional: Redirect STDERR to its own file
#SBATCH --array=0-3  # Array end is inclusive
#SBATCH --hold  # Holds job so that we can first manually check a few

# Quite neat workflow:
# Submit job array in held state, then release first job to test
# Add a dependency so that the next jobs are submitted as soon as the first job completes successfully:
# scontrol update Dependency=afterok:<jobid>_0 JobId=<jobid>
# Release all the other jobs; they'll be stuck until the first job is done
################################################################################

set -e # fail fully on first line failure (from Joost slurm_for_ml)

# Note: Remember to clear ~/.theano cache before running this script

echo "hostname: $(hostname)"
echo "Running from: $(pwd)"
echo "GPU available: $(nvidia-smi)"
module load gcc/6.2.0 cuda/10.2

export CONDA_ENVS_PATH=/home/lov701/miniconda3/envs/
export CONDA_PKGS_DIRS=/home/lov701/miniconda3/pkgs/
export CONDA_BIN=~/miniconda3/bin/

# Not running update, assuming it's done already
source "$CONDA_BIN"/activate protein_env

#export WEIGHTS_DIR=weights_msa_tkmer_20220227
#export ALIGNMENTS_DIR=msa_tkmer_20220227

# Monitor GPU usage (store outputs in ./logs/gpu_logs/)
/home/lov701/job_gpu_monitor.sh --interval 1m logs/gpu_logs &

export MSA_data_folder='./data/MSA'
export MSA_list='./data/mappings/eve_msa_mapping_20220227.csv'
export MSA_weights_location='./data/weights'
export VAE_checkpoint_location='./results/VAE_parameters'
export model_name_suffix='2022_04_26_DeepSeq_reproduce'
export model_parameters_location='./EVE/default_model_params.json'
export training_logs_location='./logs/'
export protein_index=${SLURM_ARRAY_TASK_ID}

export computation_mode='all_singles'
export all_singles_mutations_folder='./data/mutations'
export output_evol_indices_location='./results/evol_indices'
export num_samples_compute_evol_indices=20000
export batch_size=2048

python compute_evol_indices.py \
    --MSA_data_folder ${MSA_data_folder} \
    --MSA_list ${MSA_list} \
    --protein_index ${protein_index} \
    --MSA_weights_location ${MSA_weights_location} \
    --VAE_checkpoint_location ${VAE_checkpoint_location} \
    --model_name_suffix ${model_name_suffix} \
    --model_parameters_location ${model_parameters_location} \
    --computation_mode ${computation_mode} \
    --all_singles_mutations_folder ${all_singles_mutations_folder} \
    --output_evol_indices_location ${output_evol_indices_location} \
    --num_samples_compute_evol_indices ${num_samples_compute_evol_indices} \
    --batch_size ${batch_size}