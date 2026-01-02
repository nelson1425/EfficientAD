"""
Run the evaluation script for multiple experiments.

This is a wrapper around evaluate_experiment.py which is called for each
experiment specified in the config file passed to this script.

Run `print_metrics.py` afterwards in order to print the results.

For usage, see: python main.py --help or have a look at the README file.
"""

__author__ = "Kilian Batzner, Paul Bergmann, Michael Fauser, David Sattlegger"
__copyright__ = "2022, MVTec Software GmbH"

import argparse
import json
import subprocess
from os import path

from evaluate_experiment import OBJECT_NAMES


def parse_user_arguments():
    """
    Parse user arguments.

    Returns: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="""Parse user arguments.""")

    parser.add_argument(
        '--experiment_configs',
        default='experiment_configs.json',
        help='Path to the config file that contains the locations of all'
             ' experiments that should be evaluated.')

    parser.add_argument(
        '--dataset_base_dir',
        required=True,
        help='Path to the directory that contains the dataset images of the'
             ' MVTec LOCO dataset.')

    parser.add_argument(
        '--output_dir',
        default='./metrics/',
        help='Path to write evaluation results to.')

    parser.add_argument(
        '--curve_max_distance',
        default=0.001,
        type=float,
        help='Maximum distance between two points on the overall FPR-sPRO'
             ' curve. Will be used for selecting anomaly thresholds.'
             ' Decrease this value to increase the overall accuracy of'
             ' results.')

    parser.add_argument(
        '--dry_run',
        choices=['True', 'False'],
        default='False',
        help='If set to \'True\', the script is run without'
             ' perfoming the actual evalutions. Instead, the'
             ' experiments to be evaluated are simply printed'
             ' to the standard output.')

    return parser.parse_args()


def main():
    """
    Run the evaluation script for multiple experiments.
    """
    # Parse user arguments.
    args = parse_user_arguments()

    # Read the experiment configurations to be evaluated.
    with open(args.experiment_configs) as file:
        experiment_configs = json.load(file)

    # Call the evaluation script for each experiment separately.
    for experiment_id in experiment_configs['anomaly_maps_dirs']:
        print(f"\n=== Evaluate experiment: {experiment_id} ===")
        for object_name in OBJECT_NAMES:
            print(f"\nEvaluate object: {object_name}")

            # Anomaly maps for this experiment are located in this directory.
            anomaly_maps_dir = path.join(
                experiment_configs['exp_base_dir'],
                experiment_configs['anomaly_maps_dirs'][experiment_id])

            output_dir = path.join(args.output_dir, experiment_id, object_name)

            # Set up python call for the evaluation script.
            call = ['python', 'evaluate_experiment.py',
                    '--object_name', object_name,
                    '--dataset_base_dir', args.dataset_base_dir,
                    '--anomaly_maps_dir', anomaly_maps_dir,
                    '--output_dir', output_dir,
                    '--num_parallel_workers', str(4),
                    '--curve_max_distance', str(args.curve_max_distance),
                    '--niceness', str(19)]

            if args.dry_run == 'True':
                print(f"Would call: {call}\n")
            else:
                subprocess.run(call, check=True)


if __name__ == '__main__':
    main()
