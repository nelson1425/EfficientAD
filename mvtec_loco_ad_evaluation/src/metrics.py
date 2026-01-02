"""Metrics computed on a single image, but many anomaly thresholds.

At the bottom, there are two functions for computing sPRO values and false
positive rates efficiently for many images and many anomaly thresholds:
- get_spros_of_defects_of_images(...)
- get_fp_tn_areas_per_image(...)
"""

from concurrent.futures import ProcessPoolExecutor
from typing import Sequence, Optional, MutableMapping

import numpy as np
from tqdm import tqdm

from src.image import AnomalyMap, GroundTruthMap, GroundTruthChannel
from src.util import set_niceness


def get_spro(gt_channel: GroundTruthChannel,
             anomaly_map: AnomalyMap,
             anomaly_threshold: float) -> float:
    """Compute the saturated PRO metric for a single ground truth channel
    (i.e. defect) and a single threshold.

    Only use this function for testing and understanding. Do not use it
    repeatedly for different anomaly thresholds. Use get_spros(...) for that.
    """
    binary_anomaly_map = anomaly_map.get_binary_image(anomaly_threshold)
    tp = np.logical_and(binary_anomaly_map, gt_channel.bool_array)
    tp_area = np.sum(tp)
    saturation_area = gt_channel.get_saturation_area()
    return np.minimum(tp_area / saturation_area, 1.)


def get_spros_for_thresholds(gt_channel: GroundTruthChannel,
                             anomaly_map: AnomalyMap,
                             anomaly_thresholds: Sequence[float]):
    """Compute the saturated PRO metric for a single ground truth channel
    (i.e. defect) and multiple thresholds.

    Returns:
        A 1-D numpy array with the same length as anomaly_thresholds
            containing the sPRO values.
    """
    tp_areas = get_tp_areas_for_thresholds(
        gt_channel=gt_channel,
        anomaly_map=anomaly_map,
        anomaly_thresholds=anomaly_thresholds)
    saturation_area = gt_channel.get_saturation_area()
    return np.minimum(tp_areas / saturation_area, 1.)


def get_spros_per_defect_for_thresholds(gt_map: Optional[GroundTruthMap],
                                        anomaly_map: AnomalyMap,
                                        anomaly_thresholds: Sequence[float]):
    """Compute the saturated PRO metric for a single ground truth map
    (containing multiple defects / channels) and multiple thresholds.

    Returns:
        A tuple of 1-D numpy arrays. The length of the tuple is given by
            the number of channels in the ground truth map. Each numpy array
            has the same length as anomaly_thresholds and contains the sPRO
            values for the respective channel. If gt_map is None, the
            returned tuple is empty.
    """
    if gt_map is None:
        return []

    assert anomaly_map.np_array.shape == gt_map.size
    spros_per_defect = []
    for channel in gt_map.channels:
        spros = get_spros_for_thresholds(gt_channel=channel,
                                         anomaly_map=anomaly_map,
                                         anomaly_thresholds=anomaly_thresholds)
        spros_per_defect.append(spros)
    return tuple(spros_per_defect)


def get_tp_areas_for_thresholds(gt_channel: GroundTruthChannel,
                                anomaly_map: AnomalyMap,
                                anomaly_thresholds: Sequence[float]):
    """Compute the true positive areas for a single ground truth channel
    (i.e. defect) and multiple thresholds.

    Returns:
        A 1-D numpy array with the same length as anomaly_thresholds
            containing the true positive areas.
    """
    binary_anomaly_maps = anomaly_map.get_binary_images(anomaly_thresholds)
    tps = np.logical_and(binary_anomaly_maps, gt_channel.bool_array)
    tp_areas = np.sum(tps, axis=(1, 2))
    return tp_areas


def get_fp_areas_for_thresholds(gt_map: Optional[GroundTruthMap],
                                anomaly_map: AnomalyMap,
                                anomaly_thresholds: Sequence[float]):
    """Compute the false positive areas for a single ground truth map
    (containing multiple defects / channels) and multiple thresholds.

    Set gt_map to None for "good" images without ground truth annotations.

    Needs the whole GT maps to make sure that we do not mark a pixel a false
    positive that would be a true positive in another channel.

    A false positive pixel is a pixel that is defect-free in all channels of
    the ground truth map, but is a positive in the anomaly map.

    Returns:
        A 1-D numpy array with the same length as anomaly_thresholds
            containing the false positive areas.
    """

    binary_anomaly_maps = anomaly_map.get_binary_images(anomaly_thresholds)

    fp_areas: np.ndarray
    if gt_map is None:
        # This is a good image. All positive pixels are false positives.
        fp_areas = np.sum(binary_anomaly_maps, axis=(1, 2))
    else:
        # False positives do not depend on a single channel, like true
        # positives. Only pixels that are defect-free in all ground truth
        # channels can be a false positive.
        gt_combined = gt_map.get_or_over_channels()
        fps = np.logical_and(binary_anomaly_maps,
                             np.logical_not(gt_combined))
        fp_areas = np.sum(fps, axis=(1, 2))
    return fp_areas


def get_tn_areas_for_thresholds(gt_map: Optional[GroundTruthMap],
                                anomaly_map: AnomalyMap,
                                anomaly_thresholds: Sequence[float],
                                fp_areas: Optional[np.ndarray] = None):
    """Compute the true negative areas for a single ground truth map
    (containing multiple defects / channels) and multiple thresholds.

    Set gt_map to None for "good" images without ground truth annotations.

    A true negative pixel is a pixel that is defect-free in all channels of
    the ground truth map and is a negative in the anomaly map.

    The true negative area plus the false positive area equals the number of
    pixels that are defect-free in *all* channels of the ground truth map,
    see get_fp_areas_for_thresholds(...).

    The computation can be sped up significantly by setting fp_areas to the
    result of get_fp_areas_for_thresholds(...)!

    Returns:
        A 1-D numpy array with the same length as anomaly_thresholds
            containing the true negative areas.
    """

    binary_anomaly_maps = anomaly_map.get_binary_images(anomaly_thresholds)

    tn_areas: np.ndarray
    if gt_map is None:
        # This is a good image. All negative pixels are true negatives.
        tn_areas = np.sum(np.logical_not(binary_anomaly_maps), axis=(1, 2))
    else:
        # True negatives are pixels that have a negative prediction and are
        # marked as being defect-free in all channels of a ground truth map.
        gt_combined = gt_map.get_or_over_channels()

        if fp_areas is not None:
            # Compute the true negatives based on the false positives and the
            # total defect-free area.
            no_defect_area = np.sum(np.logical_not(gt_combined))
            tn_areas = no_defect_area - fp_areas
        else:
            # NOT prediction=True AND NOT defect=True can be replaced with
            # NOT (prediction=True OR defect=True)
            tns = np.logical_not(np.logical_or(binary_anomaly_maps,
                                               gt_combined))
            tn_areas = np.sum(tns, axis=(1, 2))
    return tn_areas


def _get_spros_per_defect_for_thresholds_kwargs(kwargs: MutableMapping):
    if 'niceness' in kwargs:
        set_niceness(kwargs['niceness'])
        del kwargs['niceness']
    return get_spros_per_defect_for_thresholds(**kwargs)


def get_spros_of_defects_of_images(
        gt_maps: Sequence[Optional[GroundTruthMap]],
        anomaly_maps: Sequence[AnomalyMap],
        anomaly_thresholds: Sequence[float],
        parallel_workers: Optional[int] = None,
        parallel_niceness: int = 19):
    """Compute the saturated PRO values for several images and anomaly
    thresholds, possibly in parallel.

    Args:
        gt_maps: Sequence of GroundTruthMap or None entries with the same
            length and ordering as anomaly_maps. Use None for "good" images
            without ground truth annotations.
        anomaly_maps: Must have the same length and ordering as gt_maps.
        anomaly_thresholds: Thresholds for obtaining binary anomaly maps.
        parallel_workers: If None (default), nothing will be parallelized
            across CPUs. Otherwise, the value denotes the number of CPUs to use
            for parallelism. A value of 1 will result in suboptimal performance
            compared to None.
        parallel_niceness: Niceness of child processes. Only applied in the
            parallelized setting.

    Returns:
        A list of tuples of numpy arrays. The outer list will have the same
            length as gt_maps and anomaly_maps. The length of each inner
            tuple is given by the number of defects per image. The length of
            each numpy array is given by the number of anomaly thresholds.
            "good" images will have an empty inner tuple.
    """
    assert len(gt_maps) == len(anomaly_maps)

    # Construct the kwargs for each call to get_spros_per_defect_for_thresholds
    # via _get_spros_per_defect_for_thresholds_kwargs.
    kwargs_list = []
    for gt_map, anomaly_map in zip(gt_maps, anomaly_maps):
        kwargs = {
            'gt_map': gt_map,
            'anomaly_map': anomaly_map,
            'anomaly_thresholds': anomaly_thresholds
        }
        if parallel_workers is not None:
            kwargs['niceness'] = parallel_niceness
        kwargs_list.append(kwargs)

    if parallel_workers is None:
        print(f'Computing mean sPROs for {len(anomaly_thresholds)} anomaly'
              f' thresholds...')
        spros_of_defects_of_images = [
            _get_spros_per_defect_for_thresholds_kwargs(kwargs)
            for kwargs in tqdm(kwargs_list)]
    else:
        print(f'Computing mean sPROs for {len(anomaly_thresholds)} anomaly'
              f' thresholds in parallel on {parallel_workers} CPUs...')
        pool = ProcessPoolExecutor(max_workers=parallel_workers)
        spros_of_defects_of_images = pool.map(
            _get_spros_per_defect_for_thresholds_kwargs,
            kwargs_list)
        spros_of_defects_of_images = list(spros_of_defects_of_images)
    return spros_of_defects_of_images


def _get_fp_areas_for_thresholds_kwargs(kwargs: MutableMapping):
    if 'niceness' in kwargs:
        set_niceness(kwargs['niceness'])
        del kwargs['niceness']
    return get_fp_areas_for_thresholds(**kwargs)


def get_fp_tn_areas_per_image(
        gt_maps: Sequence[Optional[GroundTruthMap]],
        anomaly_maps: Sequence[AnomalyMap],
        anomaly_thresholds: Sequence[float],
        parallel_workers: Optional[int] = None,
        parallel_niceness: int = 19):
    """Compute the false positive and the true negative areas for several
    images and anomaly thresholds, possibly in parallel.

    Args:
        gt_maps: Sequence of GroundTruthMap or None entries with the same
            length and ordering as anomaly_maps. Use for "good" images
            without ground truth annotations.
        anomaly_maps: Must have the same length and ordering as gt_maps.
        anomaly_thresholds: Thresholds for obtaining binary anomaly maps.
        parallel_workers: If None (default), nothing will be parallelized
            across CPUs. Otherwise, the value denotes the number of CPUs to use
            for parallelism. A value of 1 will result in suboptimal performance
            compared to None.
        parallel_niceness: Niceness of child processes. Only applied in the
            parallelized setting.

    Returns:
        A list of 1-D numpy arrays. The list has the same length as gt_maps
            and anomaly_maps. It contains the false positive areas for each
            image. Each numpy array has the same length as anomaly_thresholds.
        A list of 1-D numpy arrays. The list has the same length as gt_maps
            and anomaly_maps. It contains the true negative areas for each
            image. Each numpy array has the same length as anomaly_thresholds.
    """
    assert len(gt_maps) == len(anomaly_maps)

    # Construct the kwargs for each call to get_fp_areas_for_thresholds via
    # _get_fp_areas_for_thresholds_kwargs.
    kwargs_list = []
    for gt_map, anomaly_map in zip(gt_maps, anomaly_maps):
        kwargs = {
            'gt_map': gt_map,
            'anomaly_map': anomaly_map,
            'anomaly_thresholds': anomaly_thresholds
        }
        if parallel_workers is not None:
            kwargs['niceness'] = parallel_niceness
        kwargs_list.append(kwargs)

    # For each anomaly threshold, compute the FP areas per image.
    if parallel_workers is None:
        print(f'Computing FPRs for {len(anomaly_thresholds)} anomaly'
              f' thresholds...')
        fp_areas_per_image = [
            _get_fp_areas_for_thresholds_kwargs(kwargs)
            for kwargs in tqdm(kwargs_list)]
    else:
        print(f'Computing FPRs for {len(anomaly_thresholds)} anomaly'
              f' thresholds in parallel on {parallel_workers} CPUs...')
        pool = ProcessPoolExecutor(max_workers=parallel_workers)
        fp_areas_per_image = pool.map(
            _get_fp_areas_for_thresholds_kwargs,
            kwargs_list)
        fp_areas_per_image = list(fp_areas_per_image)

    # For each anomaly threshold, compute the TN areas per image.
    tn_areas_per_image = []
    for gt_map, anomaly_map, fp_areas in zip(
            gt_maps, anomaly_maps, fp_areas_per_image):
        # For each image, there is only one FP area and one TN area per
        # anomaly threshold.
        tn_areas = get_tn_areas_for_thresholds(
            gt_map=gt_map,
            anomaly_map=anomaly_map,
            anomaly_thresholds=anomaly_thresholds,
            fp_areas=fp_areas)
        tn_areas_per_image.append(tn_areas)
    return fp_areas_per_image, tn_areas_per_image


def get_fp_rates(fp_areas_per_image: Sequence[np.ndarray],
                 tn_areas_per_image: Sequence[np.ndarray]):
    """Compute false positive rates based on the results of
    get_fp_tn_areas_per_image(...).

    Args:
        fp_areas_per_image: See get_fp_tn_areas_per_image(...).
        tn_areas_per_image: See get_fp_tn_areas_per_image(...).

    Returns:
        A 1-D numpy array with the same length as each array in
            fp_areas_per_image and tn_areas_per_image. For each
            anomaly threshold, it contains the FPR computed over all images.

    Raises:
        ZeroDivisionError if there is no defect-free pixel in any of the
            images. This would result in a zero division when computing the
            FPR for any anomaly threshold.
    """
    total_fp_areas = np.zeros_like(fp_areas_per_image[0], dtype=np.int64)
    total_tn_areas = np.zeros_like(fp_areas_per_image[0], dtype=np.int64)
    for fp_areas, tn_areas in zip(fp_areas_per_image, tn_areas_per_image):
        assert len(fp_areas) == len(tn_areas)
        total_fp_areas += fp_areas
        total_tn_areas += tn_areas

    # If there is no defect-free pixel in any of the images, there cannot be
    # false positives or true negatives. Then, TN+FP will be zero, regardless
    # of the anomaly threshold. Otherwise, TN+FP will be positive, regardless
    # of the anomaly threshold.
    # Therefore, we prevent division by zero by checking if the sum of TN+FP
    # is zero for any of the thresholds.
    if total_tn_areas[0] + total_fp_areas[0] == 0:
        assert np.sum(total_fp_areas + total_tn_areas) == 0
        raise ZeroDivisionError
    fp_rates = total_fp_areas / (total_tn_areas + total_fp_areas)
    return fp_rates
