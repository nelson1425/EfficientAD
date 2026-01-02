"""Collection of utility functions."""
import os
import platform
from bisect import bisect
from typing import Iterable, Sequence, List, Callable

import numpy as np


def is_dict_order_stable():
    """Returns true, if and only if dicts always iterate in the same order."""
    if platform.python_implementation() == 'CPython':
        required_minor = 6
    else:
        required_minor = 7
    major, minor, _ = platform.python_version_tuple()
    assert major == '3' and all(s.isdigit() for s in minor)
    return int(minor) >= required_minor


def listdir(path, sort=True, include_hidden=False):
    file_names = os.listdir(path)
    if sort:
        file_names = sorted(file_names)
    if not include_hidden:
        file_names = [f for f in file_names if not f.startswith('.')]
    return file_names


def set_niceness(niceness):
    # Same as os.nice, but takes an absolute niceness instead of an increment.
    current_niceness = os.nice(0)
    niceness_increment = niceness - current_niceness
    # Regular users are not allowed to decrease the niceness. Doing so would
    # raise an exception, even if the resulting niceness would be positive.
    niceness_increment = max(0, niceness_increment)
    return os.nice(niceness_increment)


def take(seq: Sequence, indices: Iterable[int]) -> List:
    return [seq[i] for i in indices]


def flatten_2d(seq: Sequence[Sequence]) -> List:
    return [elem for innerseq in seq for elem in innerseq]


def get_sorted_nested_arrays(nested_arrays, sort_indices, nest_level=1):
    return map_nested(nested_objects=nested_arrays,
                      fun=lambda a: a[sort_indices],
                      nest_level=nest_level)


def concat_nested_arrays(head_arrays: Sequence,
                         tail_arrays: Sequence,
                         nest_level=1):
    """Concatenate numpy arrays nested in a sequence (of sequences ...
    of sequences).

    Args:
        head_arrays:  Sequence (of sequences ... of sequences) of numpy arrays.
            The lengths of the nested numpy arrays may differ.
        tail_arrays:  Sequence (of sequences ... of sequences) of numpy arrays.
            Must have the same structure as head_arrays.
            The lengths of the nested numpy arrays may differ.
        nest_level: Number of sequence levels. 1 means there is a sequence of
            arrays. 2 means there is a sequence of sequences of arrays.
            Must be >= 1.

    Returns:
        A sequence (of sequences ... of sequences) of numpy arrays with the
        same structure as head_arrays and tail_arrays containing the
        concatenated arrays.
    """

    # Zip the heads and tails at the deepest level.
    head_tail_arrays = zip_nested(head_arrays, tail_arrays,
                                  nest_level=nest_level)

    def concat(args):
        head, tail = args
        return np.concatenate([head, tail])

    return map_nested(nested_objects=head_tail_arrays,
                      fun=concat,
                      nest_level=nest_level)


def map_nested(nested_objects: Sequence, fun: Callable, nest_level=1):
    """Apply a function to objects nested in a sequence (of sequences ...
     of sequences).

    Args:
        nested_objects: Sequence (of sequences ... of sequences) of objects.
        fun: Function to call for each object.
        nest_level: Number of sequence levels. 1 means there is a sequence of
            objects. 2 means there is a sequence of sequences of objects.
            Must be >= 1.

    Returns:
        A list (of lists ... of lists) of mapped objects. This list has the
            same structure as nested_objects. Each item is the result of
            applying fun to the corresponding nested object.
    """
    assert 1 <= nest_level
    if nest_level == 1:
        return [fun(o) for o in nested_objects]
    else:
        # Go one level deeper.
        mapped = []
        for lower_nested_objects in nested_objects:
            # Map the nested sequence of objects.
            lower_mapped = map_nested(
                nested_objects=lower_nested_objects,
                fun=fun,
                nest_level=nest_level - 1
            )
            mapped.append(lower_mapped)
        return mapped


def zip_nested(*seqs: Sequence, nest_level=1):
    """Zip sequences (of sequences ... of sequences) of objects at the deepest
    level.

    Args:
        seqs: Sequences (of sequences ... of sequences) of objects.
            All sequences must have the same structure (length, length of
            descending sequences etc.).
        nest_level: Number of sequence levels. 1 means each sequence is a
            sequence of objects. 2 means each is a sequence of sequences of
            objects. Must be >= 1.

    Returns:
        A list (of lists ... of lists) of tuples containing the zipped objects.
            This list has the same structure as each sequence in seqs.
    """
    assert 1 <= nest_level
    # All sequences must have the same length.
    seq_length = len(seqs[0])
    assert set(len(seq) for seq in seqs) == {seq_length}

    if nest_level == 1:
        return list(zip(*seqs))
    else:
        # Zip one level deeper.
        zipped = []
        for i in range(seq_length):
            # Get the i-th sequence in every sequence of sequences.
            nested_seqs = [seq[i] for seq in seqs]
            # Zip the i-th sequences.
            zipped_nested = zip_nested(*nested_seqs, nest_level=nest_level - 1)
            zipped.append(zipped_nested)
        return zipped


def get_auc_for_max_fpr(fprs, y_values, max_fpr, scale_to_one=True):
    """Compute AUCs for varying maximum FPRs."""
    assert fprs[0] == 0 and fprs[-1] == 1
    auc = trapz(x=fprs,
                y=y_values,
                x_max=max_fpr)
    if scale_to_one:
        auc /= max_fpr
    return auc


def trapz(x, y, x_max=None):
    """
    This function calculates the definit integral of a curve given by
    x- and corresponding y-values. In contrast to, e.g., 'numpy.trapz()',
    this function allows to define an upper bound to the integration range by
    setting a value x_max.

    Points that do not have a finite x or y value will be ignored with a
    warning.

    Args:
        x:     Samples from the domain of the function to integrate
               Need to be sorted in ascending order. May contain the same value
               multiple times. In that case, the order of the corresponding
               y values will affect the integration with the trapezoidal rule.
        y:     Values of the function corresponding to x values.
        x_max: Upper limit of the integration. The y value at max_x will be
               determined by interpolating between its neighbors. Must not lie
               outside the range of x.

    Returns:
        Area under the curve.
    """

    x = np.array(x)
    y = np.array(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not finite_mask.all():
        print("""WARNING: Not all x and y values passed to trapezoid(...)
                 are finite. Will continue with only the finite values.""")
    x = x[finite_mask]
    y = y[finite_mask]

    # Introduce a correction term if max_x is not an element of x.
    correction = 0.
    if x_max is not None:
        if x_max not in x:
            # Get the insertion index that would keep x sorted after
            # np.insert(x, ins, x_max).
            ins = bisect(x, x_max)
            # x_max must be between the minimum and the maximum, so the
            # insertion_point cannot be zero or len(x).
            assert 0 < ins < len(x)

            # Calculate the correction term which is the integral between
            # the last x[ins-1] and x_max. Since we do not know the exact value
            # of y at x_max, we interpolate between y[ins] and y[ins-1].
            y_interp = y[ins - 1] + ((y[ins] - y[ins - 1]) *
                                     (x_max - x[ins - 1]) /
                                     (x[ins] - x[ins - 1]))
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])

        # Cut off at x_max.
        mask = x <= x_max
        x = x[mask]
        y = y[mask]

    # Return area under the curve using the trapezoidal rule.
    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction


def compute_classification_roc(
        anomaly_scores_ok,
        anomaly_scores_nok):
    """
    Compute the ROC curve for anomaly classification on the image level.

    Args:
        anomaly_scores_ok:   List of real-valued anomaly scores of anomaly-free
                             samples.
        anomaly_scores_nok:  List of real-valued anomaly scores of anomalous
                             samples.

    Returns:
        fprs: List of false positive rates.
        tprs: List of correspoding true positive rates.
    """
    # Merge anomaly scores into a single list, keeping track of the GT label.
    # 0 = anomaly-free. 1 = anomalous.
    anomaly_scores = []

    anomaly_scores.extend([(x, 0) for x in anomaly_scores_ok])
    anomaly_scores.extend([(x, 1) for x in anomaly_scores_nok])

    # Sort anomaly scores.
    anomaly_scores = sorted(anomaly_scores, key=lambda x: x[0])

    # Fetch the number of ok and nok samples.
    num_scores = len(anomaly_scores)
    num_nok = len(anomaly_scores_nok)
    num_ok = len(anomaly_scores_ok)

    # Initially, every NOK sample is correctly classified as anomalous
    # (tpr = 1.0), and every OK sample is incorrectly classified as anomalous
    # (fpr = 1.0).
    fprs = [1.0]
    tprs = [1.0]

    # Keep track of the current number of false and true positive predictions.
    num_fp = num_ok
    num_tp = num_nok

    # Compute new true and false positive rates when successively increasing
    # the threshold. Add points to the curve only when anomaly scores change.
    prev_score = None
    for i, (score, label) in enumerate(anomaly_scores):
        if label == 0:
            num_fp -= 1
        else:
            num_tp -= 1

        if (prev_score is None) or (score != prev_score) or (
                i == num_scores - 1):
            fprs.append(num_fp / num_ok)
            tprs.append(num_tp / num_nok)
            prev_score = score

    # Return (FPR, TPR) pairs in increasing order.
    fprs = fprs[::-1]
    tprs = tprs[::-1]

    return fprs, tprs


def compute_classification_auc_roc(
        anomaly_scores_ok,
        anomaly_scores_nok):
    """
    Compute the area under the ROC curve for anomaly classification.

    Args:
        anomaly_scores_ok:   List of real-valued anomaly scores of anomaly-free
                             samples.
        anomaly_scores_nok:  List of real-valued anomaly scores of anomalous
                             samples.

    Returns:
        auc_roc: Area under the ROC curve.
    """
    # Compute the ROC curve.
    fprs, tprs = \
        compute_classification_roc(anomaly_scores_ok, anomaly_scores_nok)

    # Integrate its area.
    return trapz(fprs, tprs)
