"""Wrapper around metrics.py for dynamically refining thresholds.

The important class for users of this module is MetricsAggregator.
"""

from typing import Sequence, Optional, Callable, Iterable

import numpy as np

from src.image import GroundTruthMap, AnomalyMap, get_file_path_repr
from src.metrics import get_fp_tn_areas_per_image, get_fp_rates
from src.metrics import get_spros_of_defects_of_images
from src.util import get_auc_for_max_fpr, take, flatten_2d
from src.util import get_sorted_nested_arrays, concat_nested_arrays


def binary_refinement(init_queries: Sequence,
                      init_values: Sequence,
                      max_distance: float,
                      get_values: Callable[[Sequence], Sequence],
                      min_queries_per_step: int,
                      max_queries_per_step: int,
                      max_steps: int):
    """Refine queries to maximize the resolution of a list of values.

    At each step, a query will be added between adjacent queries if the
    distance between the corresponding values is larger than max_distance.
    If there are still queries needed to reach min_queries_per_step, multiple
    queries will be inserted into intervals that have a large distance.

    Args:
        init_queries: The initial *sorted* queries. May be sorted in ascending
            or descending order. Can be of any type, but must support
            np.linspace(q1, q2, ...) for queries q1 and q2 and must support
            comparisons between q1 and q2. Must not contain duplicate queries.
        init_values: The initial values. Can be of any type, but
            np.linalg.norm(v1 - v2) must be a scalar for values v1 and v2.
        max_distance: The refinement will stop when all distances between
            adjacent values are smaller or equal to max_distance.
        get_values: A function that returns a list of values for a list of
            queries. This function will be used for refinement.
        min_queries_per_step: get_values will be called with at least this
            many queries per step.
        max_queries_per_step: get_values will be called with at most this
            many queries per step.
        max_steps: The maximum number of calls of get_values().

    Returns:
        A list with the final refined queries.
        A list with the final refined values.
    """
    # Input validation.
    assert len(init_queries) == len(init_values)
    assert len(set(init_queries)) == len(init_queries)
    pairwise_less = [init_queries[i] < init_queries[i + 1]
                     for i in range(len(init_queries) - 1)]
    if all(pairwise_less):
        ascending = True
    elif not any(pairwise_less):
        ascending = False
    else:
        raise AssertionError

    # Get all adjacent queries, whose corresponding values are further apart
    # than max_distance. As a secondary condition, the maximum tolerated area
    # between two points is derived from max_distance. The area is defined by
    # the triangle with side lengths delta_x and delta_y. Points with an area
    # smaller than the tolerated area are not considered as candidates for
    # further refinement of the curve.
    candidates = []
    max_area = 0.25 * max_distance**2
    for i in range(len(init_queries) - 1):
        distance = np.linalg.norm(init_values[i] - init_values[i + 1])

        # the area of the triangle defined by delta_x and delta_y
        area = 0.5 * np.prod(init_values[i] - init_values[i + 1])

        if distance <= max_distance or area <= max_area:
            continue
        query_left, query_right = init_queries[i], init_queries[i + 1]
        candidates.append((distance, (query_left, query_right)))
    if len(candidates) == 0 or max_steps < 1:
        return init_queries, init_values

    # Sort the candidate query intervals by distance.
    candidates = sorted(candidates, key=lambda c: c[0], reverse=True)
    print(f'Max Distance between points: {candidates[0][0]}')
    candidates = candidates[:max_queries_per_step]

    # Distribute the remaining queries to reach min_queries_per_step.
    # Distribute them weighted by each candidate's relative distance.
    num_remaining = max(min_queries_per_step - len(candidates), 0)
    total_distance_remaining = sum(distance for distance, _ in candidates)
    queries = []
    for distance, (query_left, query_right) in candidates:
        additional = distance / total_distance_remaining * num_remaining
        additional = int(np.round(additional))
        num_interval_queries = 1 + additional
        interval_queries = np.linspace(query_left, query_right,
                                       num=2 + num_interval_queries)[1:-1]
        queries.extend(interval_queries)
        num_remaining -= additional
        total_distance_remaining -= distance

    # Query the function.
    queried_values = get_values(queries)

    # Merge the new queries and values with the old queries and values.
    all_queries = list(init_queries) + list(queries)
    all_values = list(init_values) + list(queried_values)
    sort_indices = np.argsort(all_queries)
    all_queries = [all_queries[i] for i in sort_indices]
    all_values = [all_values[i] for i in sort_indices]

    if not ascending:
        all_queries = list(reversed(all_queries))
        all_values = list(reversed(all_values))

    # Start a new refinement step.
    return binary_refinement(init_queries=all_queries,
                             init_values=all_values,
                             max_distance=max_distance,
                             get_values=get_values,
                             min_queries_per_step=min_queries_per_step,
                             max_queries_per_step=max_queries_per_step,
                             max_steps=max_steps - 1)


class ThresholdMetrics:
    """Collection of metrics obtained for a list of anomaly thresholds and
    images.
    """

    def __init__(self,
                 gt_maps: Sequence[Optional[GroundTruthMap]],
                 anomaly_maps: Sequence[AnomalyMap],
                 anomaly_thresholds: np.ndarray,
                 spros_of_defects_of_images: Sequence[Sequence[np.ndarray]],
                 fp_areas_per_image: Sequence[np.array],
                 tn_areas_per_image: Sequence[np.array]):
        """
        Args:
            gt_maps: Sequence of GroundTruthMap or None entries with the
                same length and ordering as anomaly_maps. Use None for "good"
                images without ground truth annotations.
            anomaly_maps: Must have the same length and ordering as
                gt_maps.
            anomaly_thresholds: Thresholds for obtaining binary anomaly maps.
                Must be sorted in descending order so that the first
                threshold corresponds to a sPRO of 0 and an FPR of 0!
            spros_of_defects_of_images: See
                metrics.get_spros_of_defects_of_images(...).
            fp_areas_per_image: See metrics.get_fp_tn_areas_per_image(...)
            tn_areas_per_image: See metrics.get_fp_tn_areas_per_image(...)
        """

        # Input validation.
        assert len({len(gt_maps),
                    len(anomaly_maps),
                    len(spros_of_defects_of_images),
                    len(fp_areas_per_image),
                    len(tn_areas_per_image)}) == 1

        # Thresholds must be sorted in descending order.
        assert np.array_equal(np.sort(anomaly_thresholds)[::-1],
                              anomaly_thresholds)

        for per_threshold_array in (flatten_2d(spros_of_defects_of_images)
                                    + list(fp_areas_per_image)
                                    + list(tn_areas_per_image)):
            assert isinstance(per_threshold_array, np.ndarray)
            assert len(per_threshold_array) == len(anomaly_thresholds)

        self.gt_maps = gt_maps
        self.anomaly_maps = anomaly_maps
        self.anomaly_thresholds = anomaly_thresholds
        self.spros_of_defects_of_images = spros_of_defects_of_images
        self.fp_areas_per_image = fp_areas_per_image
        self.tn_areas_per_image = tn_areas_per_image

    def merge_with(self, other: 'ThresholdMetrics'):
        """Merge this collection of metrics with another one.

        The thresholds of both instances may differ, but they must have been
        computed on the same collection of ground truth and anomaly maps.
        Note that these image classes do not implement __eq__ at the moment,
        so while the lists themselves (gt_maps and anomaly_maps) may be
        different objects, the image objects must be identical for self and
        other.

        Returns:
            The merged ThresholdMetrics instance. The input instances will
                not be modified. The anomaly thresholds will be sorted such
                that the first threshold corresponds to a sPRO of 0 and an FPR
                of 0 and the last threshold to a sPRO of 1 and an FPR of 1. The
                entries in the metrics attributes will be ordered accordingly.
        """

        # Thresholds may differ, but must have been computed on the same
        # collection of ground truth and anomaly maps
        assert self.gt_maps == other.gt_maps
        assert self.anomaly_maps == other.anomaly_maps

        # Merge the 1-D numpy arrays.
        anomaly_thresholds = np.concatenate([self.anomaly_thresholds,
                                             other.anomaly_thresholds])
        # Merge the nested 1-D numpy arrays.
        fp_areas_per_image = concat_nested_arrays(self.fp_areas_per_image,
                                                  other.fp_areas_per_image)
        tn_areas_per_image = concat_nested_arrays(self.tn_areas_per_image,
                                                  other.tn_areas_per_image)
        spros_of_defects_of_images = concat_nested_arrays(
            self.spros_of_defects_of_images,
            other.spros_of_defects_of_images,
            nest_level=2)

        # Sort the anomaly thresholds in descending order so that the first
        # threshold corresponds to a sPRO of 0 and an FPR of 0.
        sort_indices = np.argsort(anomaly_thresholds)[::-1]
        anomaly_thresholds = anomaly_thresholds[sort_indices]
        fp_areas_per_image = get_sorted_nested_arrays(
            nested_arrays=fp_areas_per_image,
            sort_indices=sort_indices)
        tn_areas_per_image = get_sorted_nested_arrays(
            nested_arrays=tn_areas_per_image,
            sort_indices=sort_indices)
        spros_of_defects_of_images = get_sorted_nested_arrays(
            nested_arrays=spros_of_defects_of_images,
            sort_indices=sort_indices,
            nest_level=2)

        return ThresholdMetrics(
            gt_maps=self.gt_maps,
            anomaly_maps=self.anomaly_maps,
            anomaly_thresholds=anomaly_thresholds,
            spros_of_defects_of_images=spros_of_defects_of_images,
            fp_areas_per_image=fp_areas_per_image,
            tn_areas_per_image=tn_areas_per_image)

    def reduce_to_images(self, take_anomaly_maps: Sequence[AnomalyMap]):
        """Return a new ThresholdMetrics instance that only contains metrics
        for the given anomaly maps.

        The order of the anomaly maps is the same as take_anomaly_maps,
        which must be a subset of self.anomaly_maps.
        """
        take_indices = [self.anomaly_maps.index(image)
                        for image in take_anomaly_maps]
        return ThresholdMetrics(
            gt_maps=take(self.gt_maps, take_indices),
            anomaly_maps=take(self.anomaly_maps, take_indices),
            anomaly_thresholds=self.anomaly_thresholds,
            spros_of_defects_of_images=take(self.spros_of_defects_of_images,
                                            take_indices),
            fp_areas_per_image=take(self.fp_areas_per_image, take_indices),
            tn_areas_per_image=take(self.tn_areas_per_image, take_indices))

    def get_fp_rates(self):
        """Compute FPRs for this instance's anomaly thresholds and images.

        Raises:
            ZeroDivisionError if there is no defect-free pixel in any of the
                images. This would result in a zero division when computing the
                FPR for any anomaly threshold.
        """
        return get_fp_rates(fp_areas_per_image=self.fp_areas_per_image,
                            tn_areas_per_image=self.tn_areas_per_image)

    def get_mean_spros(self,
                       filter_defect_names: Optional[Iterable[str]] = None):
        """Compute the mean sPROs per threshold across all defects.

        Args:
            filter_defect_names: If not None, only the sPRO values from defect
                names in this sequence will be used.

        Returns:
            A 1-D numpy array containing the mean sPRO values averaged
                across all defects in self.spros_of_defect_of_images.
        """
        spros_of_defects = []

        # Flatten the list of lists of 1-D numpy arrays.
        for gt_image, image_spros in zip(self.gt_maps,
                                         self.spros_of_defects_of_images):
            if gt_image is None:
                assert len(image_spros) == 0
                continue
            for gt_channel, spros in zip(gt_image.channels, image_spros):
                defect_name = gt_channel.defect_config.defect_name
                if (filter_defect_names is None
                        or defect_name in filter_defect_names):
                    spros_of_defects.append(spros)

        if len(spros_of_defects) > 0:
            return np.mean(spros_of_defects, axis=0)
        else:
            return None

    def get_per_image_results_dicts(self, auc_max_fprs: Iterable[float]):
        """Yield dicts containing per-image metrics for json output.

        Args:
            auc_max_fprs: Maximum FPR values for computing AUC sPRO values.
        """
        for anomaly_map, fp_areas, tn_areas, spros_of_defects in zip(
                self.anomaly_maps,
                self.fp_areas_per_image,
                self.tn_areas_per_image,
                self.spros_of_defects_of_images):

            # Set the values that exist for "good" images as well.
            spros_of_defects = [spros.tolist() for spros in spros_of_defects]
            results_dict = {
                'path_full': anomaly_map.file_path,
                'path_short': get_file_path_repr(anomaly_map.file_path),
                'fp_areas': fp_areas.tolist(),
                'tn_areas': tn_areas.tolist(),
                'spros_of_defects': spros_of_defects
            }

            # Set the values that exist only for anomalous images.
            mean_spros = None
            if len(spros_of_defects) > 0:
                mean_spros = np.mean(spros_of_defects, axis=0)
                results_dict['mean_spros'] = mean_spros.tolist()

            # If there are no defect-free pixels, FP+TN will always be zero.
            # Then, we cannot compute the FPR.
            if np.sum(fp_areas + tn_areas) > 0:
                fp_rates = get_fp_rates(fp_areas_per_image=[fp_areas],
                                        tn_areas_per_image=[tn_areas])
                results_dict['fp_rates'] = fp_rates.tolist()

                if mean_spros is not None:
                    # Compute the AUC sPRO values for different max FPRs.
                    auc_spros = {}
                    for max_fpr in auc_max_fprs:
                        auc = get_auc_for_max_fpr(fprs=fp_rates,
                                                  y_values=mean_spros,
                                                  max_fpr=max_fpr,
                                                  scale_to_one=True)
                        auc_spros[max_fpr] = auc
                    results_dict['auc_spros'] = auc_spros

            yield results_dict


class MetricsAggregator:
    """Compute sPRO and FPR values for given ground truth and anomaly maps.

    The thresholds are computed dynamically by adding thresholds until the
    FPR-sPRO curve has a high resolution.

    Attributes (in addition to init arguments):
        threshold_metrics: The threshold metrics computed by the last run(...)
            call.
    """

    def __init__(self,
                 gt_maps: Sequence[Optional[GroundTruthMap]],
                 anomaly_maps: Sequence[AnomalyMap],
                 parallel_workers: Optional[int] = None,
                 parallel_niceness: int = 19):
        """
        Args:
            gt_maps: Sequence of GroundTruthMap or None entries with the
                same length and ordering as anomaly_maps. Use None for "good"
                images without ground truth annotations.
            anomaly_maps: Must have the same length and ordering as
                gt_maps.
            parallel_workers: If None (default), nothing will be parallelized
                across CPUs. Otherwise, the value denotes the number of CPUs to
                use for parallelism. A value of 1 will result in suboptimal
                performance compared to None.
            parallel_niceness: Niceness of child processes. Only applied in the
                parallelized setting.
        """
        self.gt_maps = gt_maps
        self.anomaly_maps = anomaly_maps
        self.parallel_workers = parallel_workers
        self.parallel_niceness = parallel_niceness
        self.threshold_metrics: Optional[ThresholdMetrics] = None

    def run(self, curve_max_distance: float):
        """Iteratively refine the anomaly thresholds and collect metrics.

        Args:
            curve_max_distance: Maximum distance between two points on the
                overall FPR-sPRO curve. Will be used for selecting anomaly
                thresholds.

        Returns:
            The resulting ThresholdMetrics instance.
        """
        initial_thresholds = self._get_initial_thresholds()
        initial_values = self._refinement_callback(initial_thresholds)
        binary_refinement(init_queries=initial_thresholds,
                          init_values=initial_values,
                          max_distance=curve_max_distance,
                          get_values=self._refinement_callback,
                          min_queries_per_step=20,
                          max_queries_per_step=100,
                          max_steps=10)
        return self.threshold_metrics

    def _refinement_callback(self, anomaly_thresholds: Sequence):
        # Sort the anomaly thresholds, but remember how to undo the sorting
        # for returning the points to binary_refinement in the right order.
        sort_indices = np.argsort(anomaly_thresholds)[::-1]
        unsort_indices = np.empty_like(sort_indices)
        unsort_indices[sort_indices] = np.arange(len(sort_indices))
        anomaly_thresholds_sorted = np.take(anomaly_thresholds, sort_indices)

        spros_of_defects_of_images = get_spros_of_defects_of_images(
            gt_maps=self.gt_maps,
            anomaly_maps=self.anomaly_maps,
            anomaly_thresholds=anomaly_thresholds_sorted,
            parallel_workers=self.parallel_workers,
            parallel_niceness=self.parallel_niceness)

        fp_areas_per_image, tn_areas_per_image = \
            get_fp_tn_areas_per_image(
                gt_maps=self.gt_maps,
                anomaly_maps=self.anomaly_maps,
                anomaly_thresholds=anomaly_thresholds_sorted,
                parallel_workers=self.parallel_workers,
                parallel_niceness=self.parallel_niceness)

        # Store the raw per-image metrics in a ThresholdMetrics instance.
        threshold_metrics_delta = ThresholdMetrics(
            gt_maps=self.gt_maps,
            anomaly_maps=self.anomaly_maps,
            anomaly_thresholds=np.array(anomaly_thresholds_sorted),
            spros_of_defects_of_images=spros_of_defects_of_images,
            fp_areas_per_image=fp_areas_per_image,
            tn_areas_per_image=tn_areas_per_image)

        # Compute metrics across images.
        mean_spros = threshold_metrics_delta.get_mean_spros()
        fp_rates = threshold_metrics_delta.get_fp_rates()

        # Update the threshold metrics collection.
        if self.threshold_metrics is None:
            self.threshold_metrics = threshold_metrics_delta
        else:
            self.threshold_metrics = self.threshold_metrics.merge_with(
                threshold_metrics_delta)

        result_values_sorted = [np.array([spro, fp_rate])
                                for spro, fp_rate in zip(mean_spros, fp_rates)]
        result_values = np.take(result_values_sorted, unsort_indices, axis=0)
        return result_values

    def _get_initial_thresholds(self, num_thresholds=50, epsilon=1e-6):
        """Returns initial anomaly thresholds for refining a sPRO curve.

        The thresholds are sorted in descending order. The first threshold is
        the maximum of all anomaly scores in self.anomaly_maps, plus a given
        epsilon. The last threshold is the minimum of all anomaly scores, minus
        a given epsilon. Thus, the first threshold corresponds to an FPR of 0
        and a sPRO of 0, while the last threshold corresponds to an FPR of 1 and
        a sPRO of 1.

        The thresholds in between are selected by sorting the anomaly scores
        and picking scores at equidistant indices. If the number of anomaly
        scores is large, this is done on a random subset of the anomaly scores.

        Args:
            num_thresholds: The length of the list of anomaly thresholds
                returned.
            epsilon: A small value to add to the maximum and subtract from the
                minimum of anomaly scores to reach an FPR and sPRO of 0 or 1,
                respectively.

        Returns:
            A list of floats sorted in descending order.
        """
        # sampled_scores should contain at most 10 million values to prevent
        # memory overflow and keep sorting (see below) fast.
        max_num_scores = 10_000_000
        # Therefore, we possibly need to sample random anomaly scores from each
        # anomaly map.
        num_images = len(self.anomaly_maps)
        num_scores_per_image = np.prod(self.anomaly_maps[0].np_array.shape)
        if num_images * num_scores_per_image <= max_num_scores:
            # We don't need to subsample. This corresponds to sampling
            # all scores (given that we sample without replacement).
            num_sampled_per_image = num_scores_per_image
        else:
            # We need to subsample.
            num_sampled_per_image = int(np.floor(max_num_scores / num_images))
        sampled_scores = []

        # Iterate through the images and keep track of the maximum and minimum
        # of all anomaly scores.
        some_score = self.anomaly_maps[0].np_array[0, 0]
        min_score, max_score = some_score, some_score
        for anomaly_map in self.anomaly_maps:
            min_score = min(min_score, np.min(anomaly_map.np_array))
            max_score = max(max_score, np.max(anomaly_map.np_array))

            # Sample scores from this image.
            flat_scores = anomaly_map.np_array.flatten()
            assert len(flat_scores) == num_scores_per_image
            sampled_scores_image = np.random.choice(flat_scores,
                                                    size=num_sampled_per_image,
                                                    replace=False)
            sampled_scores.append(sampled_scores_image)

        min_threshold = min_score - epsilon
        max_threshold = max_score + epsilon

        # Sort the sampled scores.
        sampled_scores = np.concatenate(sampled_scores).flatten()
        sampled_scores.sort()

        # From the sampled scores, take values at equidistant indices.
        equidistant_indices = np.linspace(0, len(sampled_scores) - 1,
                                          num=num_thresholds,
                                          endpoint=True,
                                          dtype=int)
        equidistant_indices = equidistant_indices[1:-1]
        equiheight_scores = sampled_scores[equidistant_indices]

        # Combine the minimum, the maximum and the equidistant anomaly scores
        # to form the list of thresholds, sorted in descending order.
        thresholds = equiheight_scores.tolist()[::-1]
        thresholds = [max_threshold] + thresholds + [min_threshold]
        unique_thresholds = []
        seen = set()
        for t in thresholds:
            if t in seen:
                continue
            unique_thresholds.append(t)
            seen.add(t)
        thresholds = unique_thresholds
        return thresholds
