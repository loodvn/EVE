#!/bin/bash
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1 
#SBATCH --job-name="EVE"
#SBATCH --time=72:00:00
#SBATCH --partition=short
#SBATCH --qos=ecr
#SBATCH --output=./slurm_stdout/slurm-pn-%j.out
#SBATCH --error=./slurm_stdout/slurm-pn-%j.err
#SBATCH --array=57
#SBATCH --reservation=ecr202204
####SBATCH --dependency=afterok:745359

export CONDA_ENVS_PATH=/data/stat-ecr/scro3775/conda/conda_envs
export CONDA_PKGS_DIRS=/data/stat-ecr/scro3775/conda/conda_pkgs

#/home/scro3775/miniconda3/bin/conda-env update -f ./protein_env.yml
source /home/scro3775/miniconda3/bin/activate protein_env

export MSA_data_folder='/data/coml-ecr/grte2996/'
export MSA_list='/home/scro3775/projects/EVE/data/mappings/transfokmer_mapping_20220227_DMS.csv'
export MSA_weights_location=/home/scro3775/projects/EVE/data/weights
export VAE_checkpoint_location=/data/coml-ecr/scro3775/protein/Protein_transformer/model_checkpoints/EVE
export model_name_suffix='Jan10_seed_5000'
export output_evol_indices_filename_suffix='_Jan10_seed_0000_TEST'
export model_parameters_location='/home/scro3775/projects/EVE/EVE/default_model_params.json'

export computation_mode='DMS'
#export all_singles_mutations_folder='/home/scro3775/projects/EVE/data/mutations'
export mutations_location='/data/coml-ecr/scro3775/protein/Protein_transformer/DMS/DMS_Benchmarking_Dataset_v5_20220227'
export output_evol_indices_location='/home/scro3775/projects/EVE/results/evol_indices'
export num_samples_compute_evol_indices=20000

srun \
    python compute_evol_indices_DMS.py \
        --MSA_data_folder ${MSA_data_folder} \
        --MSA_list ${MSA_list} \
        --protein_index $SLURM_ARRAY_TASK_ID \
        --MSA_weights_location ${MSA_weights_location} \
        --VAE_checkpoint_location ${VAE_checkpoint_location} \
        --model_name_suffix ${model_name_suffix} \
        --model_parameters_location ${model_parameters_location} \
        --computation_mode ${computation_mode} \
        --mutations_location ${mutations_location} \
        --output_evol_indices_location ${output_evol_indices_location} \
        --num_samples_compute_evol_indices ${num_samples_compute_evol_indices} \
        --output_evol_indices_filename_suffix ${output_evol_indices_filename_suffix} \
        --batch_size 1024