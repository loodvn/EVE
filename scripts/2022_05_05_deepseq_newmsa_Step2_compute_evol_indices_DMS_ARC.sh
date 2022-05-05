#!/bin/bash
#SBATCH --cpus-per-task=4
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

#SBATCH --job-name="eve_deepseq_dms_v7"

# Job array-specific
#SBATCH --output=./logs/slurm_files/slurm-lvn-%A_%3a-%x.out   # Nice tip: using %3a to pad to 3 characters (23 -> 023)
##SBATCH --error=./logs/slurm_files/slurm-lvn-%A_%3a-%x.err   # Optional: Redirect STDERR to its own file
##SBATCH --array=0-87  # 88 DMSs, 72 MSAs # Array end is inclusive
#SBATCH --hold  # Holds job so that we can first manually check a few

# Only running new MSAs and new DMSs
# New MSAs: ["AACC1_PSEAI_Dandage_2018", "GFP_AEQVI_Sarkisyan_2016", "P53_HUMAN_Giacomelli_2018_NULL_Etoposide", "P53_HUMAN_Giacomelli_2018_NULL_Nutlin", "P53_HUMAN_Giacomelli_2018_WT_Nutlin", "PA_I34A1_Wu_2015", "POLG_CXB3N_Mattenberger_2021", "Q2N0S5_9HIV1_Haddox_2018", "REV_HV1H2_Fernandes_2016",  "SYUA_HUMAN_Newberry_2020", "TAT_HV1BR_Fernandes_2016", "DLG4_HUMAN_Faure_2021", "GRB2_HUMAN_Faure_2021"]
#SBATCH --array=8,24,33,34,48,49,50,54,55,59,63,74,76  # Careful to use DMS indices, not MSA indices

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

# Note that the MSAs are in $DATA on ARC
export MSA_data_folder='/data/coml-ecr/grte2996/EVE/msa_tkmer_20220505_v7' # Copied from O2 '/n/groups/marks/users/lood/DeepSequence_runs/msa_tkmer_20220505_v7/'
export MSA_list='./data/mappings/DMS_mapping_20220505.csv'  # Created using Javier's DMS_mapping_20220227.csv
export MSA_weights_location='./data/weights_tkmer_20220505_v7'
export VAE_checkpoint_location='/data/coml-ecr/grte2996/EVE/results/VAE_parameters_tkmer_20220505_v7'
export model_name_suffix='2022_05_05_DeepSeq_reproduce_v7'  # Essential for skip_existing to work # Copied from O2
export model_parameters_location='./EVE/deepseq_model_params.json'
export training_logs_location='./logs/'
export protein_index=${SLURM_ARRAY_TASK_ID}

export computation_mode='DMS'
#export all_singles_mutations_folder='./data/mutations'
export mutations_location='/data/coml-ecr/grte2996/EVE/DMS/DMS_Benchmarking_Dataset_v5_20220227_20220505_v7'
export output_evol_indices_location='./results/evol_indices_20220505_v7'
export output_evol_indices_filename_suffix='_2022_05_05_DeepSeq_reproduce_v7'
export num_samples_compute_evol_indices=20000
export batch_size=2048

python compute_evol_indices_DMS.py \
    --MSA_data_folder ${MSA_data_folder} \
    --MSA_list ${MSA_list} \
    --protein_index ${protein_index} \
    --MSA_weights_location ${MSA_weights_location} \
    --VAE_checkpoint_location ${VAE_checkpoint_location} \
    --model_name_suffix ${model_name_suffix} \
    --model_parameters_location ${model_parameters_location} \
    --computation_mode ${computation_mode} \
    --mutations_location ${mutations_location} \
    --output_evol_indices_location ${output_evol_indices_location} \
    --num_samples_compute_evol_indices ${num_samples_compute_evol_indices} \
    --batch_size ${batch_size} \
    --skip_existing