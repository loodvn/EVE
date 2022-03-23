# Basically train_VAE.py but just calculating the weights
import argparse
import os
import time

import numpy as np
import pandas as pd

from utils import data_utils


def create_argparser():
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--MSA_data_folder', type=str, help='Folder where MSAs are stored', required=True)
    parser.add_argument('--MSA_list', type=str, help='List of proteins and corresponding MSA file name', required=True)
    parser.add_argument('--protein_index', type=int, help='Row index of protein in input mapping file', required=True)
    parser.add_argument('--MSA_weights_location', type=str,
                        help='Location where weights for each sequence in the MSA will be stored', required=True)
    parser.add_argument('--theta_reweighting', type=float, help='Parameters for MSA sequence re-weighting')
    parser.add_argument("--num_cpus", type=int, help="Number of CPUs to use", default=1)
    # Note: It would be nicer to have an overwrite flag, but I don't want to change the MSAProcessing code too much
    parser.add_argument("--skip_existing", help="Will quit gracefully if weights file already exists", action="store_true", default=False)
    parser.add_argument("--calc_method", choices=["evcouplings", "eve", "both"], help="Method to use for calculating weights. Note: Both produce the same results as we modified the evcouplings numba code to mirror the eve calculation", default="evcouplings")
    return parser


def main(args):
    print("Arguments:", args)

    assert os.path.isfile(args.MSA_list), f"MSA file list {args.MSA_list} doesn't seem to exist"
    mapping_file = pd.read_csv(args.MSA_list)
    protein_name = mapping_file['protein_name'][args.protein_index]
    msa_location = args.MSA_data_folder + os.sep + mapping_file['msa_location'][args.protein_index]
    print("Protein name: " + str(protein_name))
    print("MSA file: " + str(msa_location))

    if args.theta_reweighting is not None:
        theta = args.theta_reweighting
        print(f"Using custom theta value {theta} instead of loading from mapping file.")
    else:
        try:
            theta = float(mapping_file['theta'][args.protein_index])
        except KeyError as e:
            # Overriding previous errors is bad, but we're being nice to the user
            raise KeyError("Couldn't load theta from mapping file. "
                           "NOT using default value of theta=0.2; please specify theta manually. Specific line:",
                           mapping_file[args.protein_index],
                           "Previous error:", e)
        assert not np.isnan(theta), "Theta is NaN, please provide a custom theta value"

    print("Theta MSA re-weighting: " + str(theta))

    if not os.path.isdir(args.MSA_weights_location):
        # exist_ok=True: Otherwise we'll get some race conditions between concurrent jobs
        os.makedirs(args.MSA_weights_location, exist_ok=True)
        print(f"{args.MSA_weights_location} is not a directory. "
              f"Being nice and creating it for you, but this might be a mistake.")
        # raise NotADirectoryError(f"{args.MSA_weights_location} is not a directory."
        #                          f"Could create it automatically, but at the moment raising an error.")

    weights_file = args.MSA_weights_location + os.sep + protein_name + '_theta_' + str(theta) + '.npy'
    # First check that the weights file doesn't exist
    if os.path.isfile(weights_file):
        if args.skip_existing:
            print("Weights file already exists, skipping, since --skip_existing was specified")
            exit(0)
        else:
            raise FileExistsError(f"File {weights_file} already exists. "
                                  f"Please delete it if you want to re-calculate it. "
                                  f"If you want to skip existing files, use --skip_existing.")

    # The msa_data processing has a side effect of saving a weights file
    _ = data_utils.MSA_processing(
        MSA_location=msa_location,
        theta=theta,
        use_weights=True,
        weights_location=weights_file,
        num_cpus=args.num_cpus,
        weights_calc_method=args.calc_method,
    )


if __name__ == '__main__':
    start = time.perf_counter()
    parser = create_argparser()
    args = parser.parse_args()
    main(args)
    end = time.perf_counter()
    print(f"calc_weights.py took {end-start:.2f} seconds in total.")
