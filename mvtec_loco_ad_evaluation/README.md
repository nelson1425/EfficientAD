# Evaluate your experiments on the MVTec LOCO AD dataset

The evaluation scripts can be used to assess the performance of a method on the MVTec Logical Costraints Anomaly Detection (MVTec LOCO AD) dataset.
Given a directory with anomaly maps, the scripts compute the area under the sPRO curve for anomaly localization. 
Additionally, the area under the ROC curve for anomaly classification is computed.

## Installation.
Our evaluation scripts require a python 3.7 installation as well as the following
packages:
- numpy
- tifffile
- pillow
- tabulate
- tqdm

For Linux, we provide an conda environment file. It can be used to create a new conda environment with all required packages readily installed:
```
conda env create --name mvtec_loco_eval --file=conda_environment.yml
conda activate mvtec_loco_eval
```

## Evaluating a single experiment.
The evaluation script requires an anomaly map to be present for each test sample in our dataset in `.tiff` format. 
Anomaly maps must contain real-valued anomaly scores and their size must match the one of the corresponding dataset images. 
Anomaly maps must all share the same base directory and adhere to the following folder structure: 
`<anomaly_maps_dir>/<object_name>/test/<defect_name>/<image_id>.tiff`

To evaluate a single experiment on one of the dataset objects, the script `evaluate_experiment.py` can be used.  
It requires the following user arguments:
- `object_name`: Name of the dataset object to be evaluated.
- `dataset_base_dir`: Base directory that contains the MVTec LOCO AD dataset.
- `anomaly_maps_dir`: Base directory that contains the corresponding anomaly maps.
- `output_dir`: Directory to store evaluation results as `.json` files.

A possible example call to this script would be:
```
python evaluate_experiment.py \
    --object_name pushpins \
    --dataset_base_dir 'path/to/dataset/' \
    --anomaly_maps_dir 'path/to/anomaly_maps/' \
    --output_dir 'metrics/'
```

The evaluation script computes the area under the sPRO curve up to a limited false positive rate as described in our paper. 
The integration limits are specified by the variable `MAX_FPRS`.

## Evaluate multiple experiments

If more than one experiment should be evaluated simultaneously, the script `evaluate_multiple_experiments.py` can be used. 
Multiple directories conatining anomaly maps should be specified in a `config.json` file with the following structure:
```
{
    "exp_base_dir": "/path/to/all/experiments/",
    "anomaly_maps_dirs": {
    "experiment_id_1": "eg/model_1/anomaly_maps/",
    "experiment_id_2": "eg/model_2/anomaly_maps/",
    "experiment_id_3": "eg/model_3/anomaly_maps/",
    "...": "..."
    }
}
```
- `exp_base_dir`: Base directory that contains all experimental results for each evaluated method.
- `anomaly_maps_dirs`: Dictionary that contains an identifier for each evaluated experiment and the location of its anomaly maps relative to the `exp_base_dir`.

The evaluation is run by calling `evaluate_multiple_experiments.py` with the following user arguments:
- `dataset_base_dir`: Base directory that contains the MVTec LOCO dataset.
- `experiment_configs`: Path to the above `config.json` that contains all experiments to be evaluated.
- `output_dir`: Directory to store evaluation results as `.json` files.

A possible example call to this script would be:
```
  python evaluate_multiple_experiments.py \
    --dataset_base_dir 'path/to/dataset/' \
    --experiment_configs 'configs.json' \
    --output_dir 'metrics/'
```

## Visualize the evaluation results.
After running `evaluate_experiment.py` or `evaluate_multiple_experiments.py`, the script `print_metrics.py`  can be used to visualize all computed metrics in a table. 
In total, three tables are printed to the standard output. The first two tables display the performance for the structural and logical anomalies, respectively. 
The third table shows the mean performance over both anomaly types.

The script requires the following user arguments:
- `metrics_folder`: The base directory that contains the computed metrics for each evaluated method. This directory is usually identical to the output directory specified in `evaluate_experiment.py` or `evaluate_multiple_experiments.py`.
- `metric_type`: Select either `localization` or `classification`. When selecting `localization`,
the AUC-sPRO results for the pixelwise localization of anomalies is shown. When selecting `classification`, the image level AUC-ROC for anomaly classification is shown.
- `integration_limit`: The integration limit until which the area under the sPRO curve is computed. This parameter is only applicable when `metric_type` is set to `localization`.

# License
The license agreement for our evaluation code is found in the accompanying
`LICENSE.txt` file.

The version of this evaluation script is: 2.0
