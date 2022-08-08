import datetime
import os,sys
import json
import argparse
from resource import getrusage, RUSAGE_SELF

import pandas as pd
import torch

from EVE import VAE_model
from utils import data_utils

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Evol indices')
    parser.add_argument('--MSA_data_folder', type=str, help='Folder where MSAs are stored')
    parser.add_argument('--MSA_list', type=str, help='List of proteins and corresponding MSA file name')
    parser.add_argument('--protein_index', type=int, help='Row index of protein in input mapping file')
    parser.add_argument('--MSA_weights_location', type=str, help='Location where weights for each sequence in the MSA will be stored')
    parser.add_argument('--theta_reweighting', type=float, help='Parameters for MSA sequence re-weighting')
    parser.add_argument('--VAE_checkpoint_location', type=str, help='Location where VAE model checkpoints will be stored')
    parser.add_argument('--model_name_suffix', default='Jan1', type=str, help='model checkpoint name is the protein name followed by this suffix')
    parser.add_argument('--model_parameters_location', type=str, help='Location of VAE model parameters')
    parser.add_argument('--computation_mode', type=str, help='Computes evol indices for all single AA mutations or for a passed in list of mutations (singles or multiples) [all_singles,input_mutations_list]')
    parser.add_argument('--all_singles_mutations_folder', type=str, help='Location for the list of generated single AA mutations')
    parser.add_argument('--mutations_location', type=str, help='Location of all mutations to compute the evol indices for')
    parser.add_argument('--output_evol_indices_location', type=str, help='Output location of computed evol indices')
    parser.add_argument('--output_evol_indices_filename_suffix', default='', type=str, help='(Optional) Suffix to be added to output filename')
    parser.add_argument('--num_samples_compute_evol_indices', type=int, help='Num of samples to approximate delta elbo when computing evol indices')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size when computing evol indices')
    parser.add_argument("--skip_existing", action="store_true", help="Skip scoring if output file already exists")
    parser.add_argument("--aggregation_method", choices=["full", "batch", "online"], default="full", help="Method to aggregate evol indices")
    parser.add_argument("--threshold_focus_cols_frac_gaps", type=float,
                        help="Maximum fraction of gaps allowed in focus columns - see data_utils.MSA_processing")
    args = parser.parse_args()

    print("Arguments:", args)

    assert os.path.isfile(args.MSA_list), 'MSA list file does not exist: {}'.format(args.MSA_list)
    mapping_file = pd.read_csv(args.MSA_list)
    print("Mapping file head:\n", mapping_file.head())
    DMS_id = mapping_file['DMS_id'][args.protein_index]
    protein_name = mapping_file['UniProt_ID'][args.protein_index]  # Using Javier's mapping file
    DMS_filename = mapping_file['DMS_filename'][args.protein_index]
    mutant = mapping_file['DMS_filename'][args.protein_index]
    msa_location = args.MSA_data_folder + os.sep + mapping_file['MSA_filename'][args.protein_index]
    DMS_mutant_column = mapping_file['DMS_mutant_column'][args.protein_index]  # Using Javier's mapping file
    print("Protein name: "+str(protein_name))
    print("MSA file: "+str(msa_location))
    print("DMS id: "+str(DMS_id))
    assert DMS_filename.startswith(DMS_id), 'DMS id does not match DMS filename: {} vs {}'.format(DMS_id, DMS_filename)

    # Check filepaths are valid
    evol_indices_output_filename = os.path.join(args.output_evol_indices_location, DMS_id + '_' + protein_name + '_' + str(
        args.num_samples_compute_evol_indices) + '_samples' + args.output_evol_indices_filename_suffix + '.csv')

    if os.path.isfile(evol_indices_output_filename):
        print("Output file already exists: " + str(evol_indices_output_filename))

        if args.skip_existing:
            print("Skipping scoring since args.skip_existing is True")
            sys.exit(0)
        else:
            print("Overwriting existing file: " + str(evol_indices_output_filename))
            print("To skip scoring for existing files, use --skip_existing")
    # Check if surrounding directory exists
    else:
        assert os.path.isdir(os.path.dirname(evol_indices_output_filename)), \
            'Output directory does not exist: {}. Please create directory before running script.\nOutput filename given: {}.\nDebugging curdir={}'\
            .format(os.path.dirname(evol_indices_output_filename), evol_indices_output_filename, os.getcwd())

    if args.theta_reweighting is not None:
        theta = args.theta_reweighting
    else:
        try:
            theta = float(mapping_file['theta'][args.protein_index])
        except:
            theta = 0.2
    print("Theta MSA re-weighting: "+str(theta))

    # Using data_kwargs so that if options aren't set, they'll be set to default values
    data_kwargs = {}
    if args.threshold_focus_cols_frac_gaps is not None:
        print("Using custom threshold_focus_cols_frac_gaps: ", args.threshold_focus_cols_frac_gaps)
        data_kwargs['threshold_focus_cols_frac_gaps'] = args.threshold_focus_cols_frac_gaps

    data = data_utils.MSA_processing(
            MSA_location=msa_location,
            theta=theta,
            use_weights=False,  # Don't need weights for evol indices
            weights_location=args.MSA_weights_location + os.sep + protein_name + '_theta_' + str(theta) + '.npy',
            **data_kwargs,
    )

    print("MSA data object loaded.")
    print(f"Current time: {datetime.datetime.now()}, peak memory usage: {getrusage(RUSAGE_SELF).ru_maxrss}")

    if args.computation_mode=="all_singles":
        data.save_all_singles(output_filename=args.all_singles_mutations_folder + os.sep + protein_name + "_all_singles.csv")
        args.mutations_location = args.all_singles_mutations_folder + os.sep + protein_name + "_all_singles.csv"
    else:
        args.mutations_location = args.mutations_location + os.sep + DMS_filename

    model_name = protein_name + "_" + args.model_name_suffix
    print("Model name: "+str(model_name))

    model_params = json.load(open(args.model_parameters_location))

    model = VAE_model.VAE_model(
                    model_name=model_name,
                    data=data,
                    encoder_parameters=model_params["encoder_parameters"],
                    decoder_parameters=model_params["decoder_parameters"],
                    random_seed=42
    )
    model = model.to(model.device)

    checkpoint_name = str(args.VAE_checkpoint_location) + os.sep + model_name + "_final"
    assert os.path.isfile(checkpoint_name), 'Checkpoint file does not exist: {}'.format(checkpoint_name)

    try:
        checkpoint = torch.load(checkpoint_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Initialized VAE with checkpoint '{}' ".format(checkpoint_name))
    except Exception as e:
        print("Unable to load VAE model checkpoint {}".format(checkpoint_name))
        raise e

    print(f"Current time: {datetime.datetime.now()}, Peak memory in GB: {getrusage(RUSAGE_SELF).ru_maxrss / 1024**2:.3f}")

    list_valid_mutations, evol_indices, _, _ = model.compute_evol_indices(
        msa_data=data,
        list_mutations_location=args.mutations_location,
        mutant_column=DMS_mutant_column,
        num_samples=args.num_samples_compute_evol_indices,
        batch_size=args.batch_size,
        aggregation_method=args.aggregation_method,
    )

    df = {}
    df['protein_name'] = protein_name
    df['DMS_id'] = DMS_id
    df['mutant'] = list_valid_mutations
    df['evol_indices'] = evol_indices
    df = pd.DataFrame(df)

    try:
        keep_header = os.stat(evol_indices_output_filename).st_size == 0
    except:
        keep_header=True

    if not keep_header:
        print("Warning: File already exists, appending scores below.")

    df.to_csv(path_or_buf=evol_indices_output_filename, index=False, mode='a', header=keep_header)

    print(f"Peak memory in GB: {getrusage(RUSAGE_SELF).ru_maxrss / 1024**2:.3f}")

    print("Done")
