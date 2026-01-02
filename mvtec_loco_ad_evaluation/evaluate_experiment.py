"""
Evaluate a single experiment on a single object of the MVTec LOCO AD dataset.

For more information, see: python main.py --help or have a look at the README.
"""

__author__ = "Kilian Batzner, Paul Bergmann, Michael Fauser, David Sattlegger"
__copyright__ = "2022, MVTec Software GmbH"

import argparse
import glob
import json
import os
from typing import Optional, Iterable

import numpy as np
from tqdm import tqdm

from src.aggregation import MetricsAggregator, ThresholdMetrics
from src.image import GroundTruthMap, AnomalyMap, DefectsConfig
from src.util import get_auc_for_max_fpr, listdir, set_niceness, \
    compute_classification_auc_roc

TIFF_EXTS = ['.tif', '.tiff', '.TIF', '.TIFF']
OBJECT_NAMES = ['breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag',
                'splicing_connectors']

# The AU-sPRO is only computed up to a certain integration limit. Here, you
# can specify the integration limits that should be evaluated.
MAX_FPRS = [0.01, 0.05, 0.1, 0.3, 1.]


def parse_arguments():
    """
        Parse user arguments for the evaluation of a method on the MVTec LOCO AD
        dataset.

        returns:
            Parsed user arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--object_name',
        choices=OBJECT_NAMES,
        help='Name of the dataset object to be evaluated.')

    parser.add_argument(
        '--dataset_base_dir',
        help='Path to the directory that contains the dataset images of the'
             ' MVTec LOCO AD dataset.')

    parser.add_argument(
        '--anomaly_maps_dir',
        required=True,
        help='Path to the anomaly maps directory of the evaluated method.')

    parser.add_argument(
        '--output_dir',
        default=None,
        help='Path to the directory to store evaluation results. If no output'
             ' directory is specified, the results are not written to drive.')

    parser.add_argument(
        '--num_parallel_workers',
        default=None,
        type=optional_int,
        help='If None (default), nothing will be parallelized across CPUs.'
             ' Otherwise, the value denotes the number of CPUs to use for'
             ' parallelism. A value of 1 will result in suboptimal performance'
             ' compared to None.')

    parser.add_argument(
        '--curve_max_distance',
        default=0.001,
        type=float,
        help='Maximum distance between two points on the overall FPR-sPRO'
             ' curve. Will be used for selecting anomaly thresholds.'
             ' Decrease this value to increase the overall accuracy of'
             ' results.')

    parser.add_argument(
        '--niceness',
        type=int,
        default=19,
        choices=list(range(20)),
        help='UNIX niceness of all evaluation processes.')

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    set_niceness(args.niceness)

    # Read the defects config file of the evaluated object.
    defects_config_path = os.path.join(
        args.dataset_base_dir, args.object_name, 'defects_config.json')
    with open(defects_config_path) as defects_config_file:
        defects_list = json.load(defects_config_file)
    defects_config = DefectsConfig.create_from_list(defects_list)

    # Read the ground truth maps and the anomaly maps.
    gt_dir = os.path.join(
        args.dataset_base_dir, args.object_name, 'ground_truth')
    anomaly_maps_test_dir = os.path.join(
        args.anomaly_maps_dir, args.object_name, 'test')
    gt_maps, anomaly_maps = read_maps(
        gt_dir=gt_dir,
        anomaly_maps_test_dir=anomaly_maps_test_dir,
        defects_config=defects_config)

    # Collect relevant metrics based on the ground truth and anomaly maps.
    metrics_aggregator = MetricsAggregator(
        gt_maps=gt_maps,
        anomaly_maps=anomaly_maps,
        parallel_workers=args.num_parallel_workers,
        parallel_niceness=args.niceness)
    metrics = metrics_aggregator.run(
        curve_max_distance=args.curve_max_distance)

    # Fetch the anomaly localization results.
    localization_results = get_auc_spro_results(
        metrics=metrics,
        anomaly_maps_test_dir=anomaly_maps_test_dir)

    # Fetch the image-level anomaly detection results.
    classification_results = get_image_level_detection_metrics(
        gt_maps=gt_maps,
        anomaly_maps=anomaly_maps)

    # Create the dict to write to metrics.json.
    results = {
        'localization': localization_results,
        'classification': classification_results
    }

    # Write the results to the output directory.
    if args.output_dir is not None:
        print(f'Writing results to {args.output_dir}')
        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, 'metrics.json')
        with open(results_path, 'w') as results_file:
            json.dump(results, results_file, indent=4, sort_keys=True)


def read_maps(gt_dir: str,
              anomaly_maps_test_dir: str,
              defects_config: DefectsConfig):
    """Read the ground truth and the anomaly maps."""
    print('Reading ground truth and corresponding anomaly maps...')
    gt_maps = []
    anomaly_maps = []

    # Search for available relative paths to ground truth maps and to
    # anomaly maps.
    gt_map_rel_paths = set(get_available_gt_map_rel_paths(gt_dir))
    anomaly_rel_paths = list(
        get_available_test_image_rel_paths(anomaly_maps_test_dir))
    anomaly_rel_paths_no_ext = [os.path.splitext(p)[0]
                                for p in anomaly_rel_paths]
    # Check that there are no duplicates with different file endings.
    assert len(set(anomaly_rel_paths_no_ext)) == len(anomaly_rel_paths_no_ext)

    # Every ground truth image must have an anomaly image.
    skipped_gt_maps = gt_map_rel_paths.difference(anomaly_rel_paths_no_ext)
    if len(skipped_gt_maps) > 0:
        raise OSError(
            'These ground truth maps do not have corresponding anomaly'
            f' maps: {sorted(skipped_gt_maps)}')

    # For every relative path, read the ground truth (if available) and the
    # anomaly map.
    for rel_path, rel_path_no_ext in tqdm(
            zip(anomaly_rel_paths, anomaly_rel_paths_no_ext),
            total=len(anomaly_rel_paths)):
        anomaly_map_path = os.path.join(anomaly_maps_test_dir, rel_path)
        anomaly_map = AnomalyMap.read_from_tiff(anomaly_map_path)
        anomaly_maps.append(anomaly_map)

        if rel_path_no_ext in gt_map_rel_paths:
            gt_map_path = os.path.join(gt_dir, rel_path_no_ext)
            gt_map = GroundTruthMap.read_from_png_dir(
                png_dir=gt_map_path,
                defects_config=defects_config)
            gt_maps.append(gt_map)
        else:
            # This must be a good image. Hence, the path must start with
            # "good/".
            if not rel_path.startswith('good/'):
                raise OSError(
                    f'Anomaly map {rel_path} has no corresponding ground'
                    ' truth map, so it must be a good image. Good images'
                    ' must have a relative path starting with "good/" '
                    f' (relative to the test dir at {anomaly_maps_test_dir})'
                )
            gt_maps.append(None)

    return gt_maps, anomaly_maps


def get_available_gt_map_rel_paths(gt_dir: str) -> Iterable[str]:
    """Search for available relative paths to ground truth maps.

    Note that ground truth maps are represented as a directory containing
    .png files (one for each channel).

    The returned paths are the relative paths to the directories.
    """
    for defect_type_name in listdir(gt_dir):
        file_complaint_str = ('Ground truth directory must not contain files'
                              ' except for the .pngs nested in the single'
                              ' image directories.')

        # Get the logical_anomalies and structural_anomalies subdirectories.
        defect_type_dir = os.path.join(gt_dir, defect_type_name)
        if not os.path.isdir(defect_type_dir):
            raise OSError(file_complaint_str)

        # Raise if the directory name is not what we expect.
        valid_defect_type_names = ['logical_anomalies', 'structural_anomalies']
        if defect_type_name not in valid_defect_type_names:
            raise OSError(
                f'Subdirectory of ground truth maps'
                f' directory has name {defect_type_name}, but should be'
                f' "logical_anomalies" or "structural_anomalies".')

        # Get the list of all non-empty image directories (000, 001, etc.).
        for image_dir_name in listdir(defect_type_dir):
            image_dir_path = os.path.join(defect_type_dir, image_dir_name)
            # Image dir must be a dir.
            if not os.path.isdir(image_dir_path):
                raise OSError(file_complaint_str)
            # Image dir must contain pngs.
            if len(glob.glob(os.path.join(image_dir_path, '*.png'))) == 0:
                raise OSError(
                    f'Ground truth directory of single image'
                    f' at {image_dir_path} does not contain any pngs.')

            # Yield the relative paths to the image subdirectory.
            yield os.path.join(defect_type_name, image_dir_name)


def get_available_test_image_rel_paths(test_images_dir: str) -> Iterable[str]:
    """Search for available relative paths to test anomaly maps."""

    for defect_dir_name in listdir(test_images_dir):
        defect_dir = os.path.join(test_images_dir, defect_dir_name)
        if not os.path.isdir(defect_dir):
            continue

        for file_name in listdir(defect_dir):
            _, ext = os.path.splitext(file_name)
            if ext not in TIFF_EXTS:
                continue
            yield os.path.join(defect_dir_name, file_name)


def get_auc_spro_results(metrics: ThresholdMetrics,
                         anomaly_maps_test_dir: str):
    """Compute AUC sPRO values for all images, images in subdirectories and
    defect names.
    """
    # Compute the AUC sPRO for logical and structural anomalies.
    auc_spro = get_auc_spros_per_subdir(
        metrics=metrics,
        anomaly_maps_test_dir=anomaly_maps_test_dir,
        add_good_images=True)

    # Compute the mean performance over logical and structural anomalies.
    mean_spros = dict()
    for limit in auc_spro['structural_anomalies'].keys():
        auc_spro_structural = auc_spro['structural_anomalies'][limit]
        auc_spro_logical = auc_spro['logical_anomalies'][limit]
        mean = 0.5 * (auc_spro_structural + auc_spro_logical)
        mean_spros[limit] = mean
    auc_spro['mean'] = mean_spros

    return {'auc_spro': auc_spro}


def get_auc_spros_per_defect_type(metrics: ThresholdMetrics,
                                  defects_config: DefectsConfig,
                                  add_good_images):
    """Compute the AUC sPRO for images by defect type.

    For each defect type, all ground truth maps that contain at least one
    channel with this type will be considered. For the FPR computation, all
    channels of each image will be used. For the sPRO computation, however,
    only channels with the respective defect type will be used.

    If add_good_images is True, all good images in `metrics` will be added for
    the computation of each defect type's AUC sPRO value.
    """
    aucs_per_defect_type = {}
    defect_configs = defects_config.pixel_value_to_entry.values()
    defect_names = set(c.defect_name for c in defect_configs)

    for defect_name in defect_names:
        # Collect all anomaly maps, where the corresponding ground truth
        # maps has at least one channel with the defect type corresponding to
        # defect_name.
        reduced_anomaly_maps = []
        for anomaly_map, gt_map in zip(metrics.anomaly_maps, metrics.gt_maps):
            # Optionally add good images as well.
            if gt_map is None:
                if add_good_images:
                    reduced_anomaly_maps.append(anomaly_map)
                continue

            # Get the defect types in the ground truth map.
            defect_names_in_gt_map = \
                set(c.defect_config.defect_name for c in gt_map.channels)

            if defect_name in defect_names_in_gt_map:
                reduced_anomaly_maps.append(anomaly_map)

        # Reduce the threshold metrics and compute the AUC sPRO value.
        reduced_metrics = metrics.reduce_to_images(reduced_anomaly_maps)
        aucs_per_defect_type[defect_name] = get_auc_spros_for_metrics(
            metrics=reduced_metrics,
            filter_defect_names_for_spro=[defect_name])
    return aucs_per_defect_type


def get_auc_spros_per_subdir(metrics: ThresholdMetrics,
                             anomaly_maps_test_dir,
                             add_good_images):
    """Compute the AUC sPRO for images in subdirectories (usually "good",
    "structural_anomalies" and "logical_anomalies").

    If add_good_images is True, the images in the "good" subdirectory will be
    added to the images of each subdirectory for computing the corresponding
    AUC sPRO value. Hence, the result dict will not contain a "good" key.
    """
    aucs_per_subdir = {}
    subdir_names = listdir(anomaly_maps_test_dir)

    good_images = []
    if add_good_images:
        # Include the good images for each subdir.
        assert 'good' in subdir_names
        good_subdir = os.path.join(anomaly_maps_test_dir, 'good')
        good_subdir = os.path.realpath(good_subdir)
        good_images = [
            a for a in metrics.anomaly_maps
            if os.path.realpath(a.file_path).startswith(good_subdir)]
    # Regardless of add_good_images, we cannot compute an AUC sPRO value only
    # for the good images.
    if 'good' in subdir_names:
        subdir_names.remove('good')

    for subdir_name in subdir_names:
        subdir = os.path.join(anomaly_maps_test_dir, subdir_name)
        subdir = os.path.realpath(subdir)
        # Get all anomaly maps in here.
        subdir_anomaly_maps = [
            a for a in metrics.anomaly_maps
            if os.path.realpath(a.file_path).startswith(subdir)]
        if add_good_images:
            subdir_anomaly_maps += good_images

        subdir_metrics = metrics.reduce_to_images(subdir_anomaly_maps)

        aucs_per_subdir[subdir_name] = get_auc_spros_for_metrics(
            subdir_metrics)
    return aucs_per_subdir


def get_auc_spros_for_metrics(
        metrics: ThresholdMetrics,
        filter_defect_names_for_spro: Optional[Iterable[str]] = None):
    """Compute AUC sPRO values for a given ThresholdMetrics instance.

    Args:
        metrics: The ThresholdMetrics instance.
        filter_defect_names_for_spro: If not None, only the sPRO values from
            defect names in this sequence will be used. Does not affect the
            computation of FPRs!
    """
    auc_spros = {}
    for max_fpr in MAX_FPRS:
        try:
            fp_rates = metrics.get_fp_rates()
        except ZeroDivisionError:
            auc = None
        else:
            mean_spros = metrics.get_mean_spros(
                filter_defect_names=filter_defect_names_for_spro)
            auc = get_auc_for_max_fpr(fprs=fp_rates,
                                      y_values=mean_spros,
                                      max_fpr=max_fpr,
                                      scale_to_one=True)
        auc_spros[max_fpr] = auc
    return auc_spros


def get_image_level_detection_metrics(gt_maps, anomaly_maps):
    """Main function for computing all image-level anomaly detection results."""
    per_image_results = get_image_level_detection_metrics_per_image(
        gt_maps, anomaly_maps)
    auc_roc_classification = get_image_level_detection_metrics_aggregated(
        per_image_results)
    return auc_roc_classification


def get_image_level_detection_metrics_per_image(gt_maps, anomaly_maps):
    """Computes the image-level anomaly scores for an anomaly map.
    """
    per_image_results = []
    for gt_map, anomaly_map in zip(gt_maps, anomaly_maps):
        # Determine the image type (good, logical, or structural).
        parent_dir_path, _ = os.path.split(anomaly_map.file_path)
        _, image_type = os.path.split(parent_dir_path)

        image_results = {
            'file_path': anomaly_map.file_path,
            'gt_contains_defect': gt_map is not None,
            'anomaly_scores': {},
            'image_type': image_type
        }
        # Compute the image-level anomaly score by taking the maximum of the
        # anomaly map.
        image_level_score = float(np.max(anomaly_map.np_array))
        image_results['anomaly_scores']['max'] = image_level_score
        per_image_results.append(image_results)

    return per_image_results


def get_image_level_detection_metrics_aggregated(per_image_results):
    """Compute aggregated image-level anomaly detection metrics like the AUC-ROC

    Args:
         per_image_results: should be the result of
           get_image_level_detection_metrics_per_image
    """
    agg_results = {'auc_roc': {}}
    # Iterate through the different image-level score computation methods.

    # Collect image-level scores of good and anomalous images.
    anomaly_scores = \
        {'good': [], 'structural_anomalies': [], 'logical_anomalies': []}

    # Iterate through the images and collect the information required for
    # computing the AUC-ROC.
    for image_result in per_image_results:
        anomaly_scores[image_result['image_type']].append(
            image_result['anomaly_scores']['max'])

    # Compute auc-roc for structural and logical anomalies separately.
    auc_roc_structural = \
        compute_classification_auc_roc(
            anomaly_scores['good'], anomaly_scores['structural_anomalies'])

    auc_roc_logical = \
        compute_classification_auc_roc(
            anomaly_scores['good'], anomaly_scores['logical_anomalies'])

    agg_results['auc_roc']['logical_anomalies'] = auc_roc_logical
    agg_results['auc_roc']['structural_anomalies'] = auc_roc_structural
    agg_results['auc_roc']['mean'] = \
        0.5 * (auc_roc_logical + auc_roc_structural)
    return agg_results


def optional_int(str_value: Optional[str]) -> Optional[int]:
    """Helper function for parsing optional integer arguments with argparse."""
    if str_value is None or str_value.lower() == 'none':
        return None
    else:
        try:
            return int(str_value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                'An optional integer argument was given the value'
                f' {str_value}, but the value must be None or an integer.')


if __name__ == '__main__':
    main()
