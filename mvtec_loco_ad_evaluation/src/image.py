"""Classes for handling ground truth and anomaly maps."""

import glob
import os
from functools import lru_cache
from typing import Sequence, Optional, Mapping, Iterable, Tuple, Union, Any

import numpy as np
import tifffile
from PIL import Image


def get_file_path_repr(file_path: Optional[str]) -> str:
    if file_path is None:
        return 'no file path'
    else:
        parent_dir_path, file_name = os.path.split(file_path)
        _, parent_dir = os.path.split(parent_dir_path)
        return f'.../{parent_dir}/{file_name}'


class DefectConfig:
    def __init__(self,
                 defect_name: str,
                 pixel_value: int,
                 saturation_threshold: Union[int, float],
                 relative_saturation: bool):
        # Input validation.
        assert 1 <= pixel_value <= 255
        if relative_saturation:
            assert isinstance(saturation_threshold, float)
            assert 0. < saturation_threshold <= 1.
        else:
            assert isinstance(saturation_threshold, int)

        self.defect_name = defect_name
        self.pixel_value = pixel_value
        self.saturation_threshold = saturation_threshold
        self.relative_saturation = relative_saturation

    def __repr__(self):
        return f'DefectConfig({self.__dict__})'


class DefectsConfig:
    def __init__(self, entries: Sequence[DefectConfig]):
        # Create a pixel_value -> entry mapping for faster lookup.
        self.pixel_value_to_entry = {e.pixel_value: e for e in entries}

    @property
    def entries(self):
        return tuple(self.pixel_value_to_entry.values())

    @classmethod
    def create_from_list(cls, defects_list: Sequence[Mapping[str, Any]]):
        entries = []
        for defect_config in defects_list:
            entry = DefectConfig(
                defect_name=defect_config['defect_name'],
                pixel_value=defect_config['pixel_value'],
                saturation_threshold=defect_config['saturation_threshold'],
                relative_saturation=defect_config['relative_saturation'])
            entries.append(entry)
        return DefectsConfig(entries=entries)


class GroundTruthChannel:
    """A channel of a ground truth map.

    Corresponds to exactly one defect in a ground truth map. Must not be used
    to represent a defect-free image.
    """

    def __init__(self,
                 bool_array: np.ndarray,
                 defect_config: DefectConfig):
        """
        Args:
            bool_array: A 2-D numpy array with dtype np.bool_. A True value
                indicates an anomalous pixel.
            defect_config: The DefectConfig for this channel's defect type.
        """

        # Input validation.
        # numpy dtypes need to be checked with == instead of `is`, see
        # https://stackoverflow.com/a/26921882/2305095
        # We want np.bool_ for a fast computation of unions, intersections etc.
        assert len(bool_array.shape) == 2 and bool_array.dtype == np.bool_

        self.bool_array = bool_array
        self.defect_config = defect_config

    def get_defect_area(self):
        return np.sum(self.bool_array)

    def get_saturation_area(self):
        defect_area = self.get_defect_area()
        if self.defect_config.relative_saturation:
            return int(self.defect_config.saturation_threshold * defect_area)
        else:
            return np.minimum(self.defect_config.saturation_threshold,
                              defect_area)

    @classmethod
    def create_from_integer_array(cls,
                                  np_array: np.ndarray,
                                  defects_config: DefectsConfig):
        """Create a new GroundTruthChannel from an integer array.

        Args:
            np_array: A 2-D array with exactly one distinct positive value. All
                non-positive entries must be zero and correspond to defect-free
                pixels.
            defects_config: The defects configuration for the dataset object
                being evaluated.
        """
        assert np.issubdtype(np_array.dtype, np.integer)

        # Ensure that each channel has exactly one unique positive integer.
        sorted_unique = sorted(np.unique(np_array))
        if len(sorted_unique) == 1:
            defect_id = sorted_unique[0]
        else:
            zero, defect_id = sorted_unique
            assert zero == 0
        assert defect_id > 0
        # Cast np.uint8 etc. to int.
        defect_id = int(defect_id)

        # Convert to bool for faster logical operations with anomaly maps.
        bool_array = np_array.astype(np.bool_)

        # Look up the defect config for this defect id.
        defect_config = defects_config.pixel_value_to_entry[defect_id]
        return GroundTruthChannel(bool_array=bool_array,
                                  defect_config=defect_config)


class GroundTruthMap:
    """A ground truth map for an anomalous image.

    Each channel corresponds to one defect in the image.

    Use GroundTruthMap.read_from_tiff(...) to read a GroundTruthMap from a
    .tiff file.

    If defect_id_to_name is None, it is constructed based on the defect ids in
    the channels, using defect_id -> str(defect_id).
    """

    def __init__(self,
                 channels: Sequence[GroundTruthChannel],
                 file_path: Optional[str] = None):

        # Input validation.
        assert len(channels) > 0
        # Ensure that each channel has the same size.
        first_shape = channels[0].bool_array.shape
        assert set(c.bool_array.shape for c in channels) == {first_shape}
        # Check whether some channels have larger saturation thresholds than
        # defect areas.
        for i_channel, channel in enumerate(channels):
            if channel.defect_config.relative_saturation:
                continue
            threshold = channel.defect_config.saturation_threshold
            defect_area = channel.get_defect_area()
            if threshold > defect_area:
                print(f'WARNING: Channel {i_channel + 1} (1=first) of ground'
                      f' truth image {get_file_path_repr(file_path)} has a'
                      f' defect area of {defect_area}, but a saturation'
                      f' threshold of {threshold}. Corresponding defect'
                      f' config: {channel.defect_config}')

        self.channels = tuple(channels)
        self.file_path = file_path

    @property
    def size(self):
        return self.channels[0].bool_array.shape

    def get_or_over_channels(self) -> np.ndarray:
        """Combine the channels with a logical OR operation.

        Returns a numpy array of type np.bool_.
        """
        channels_np = tuple(c.bool_array for c in self.channels)
        return np.sum(channels_np, axis=0).astype(bool)

    @classmethod
    def read_from_png_dir(cls,
                          png_dir: str,
                          defects_config: DefectsConfig):
        """Read a GroundTruthMap from a directory containing one .png per
        channel.
        """
        gt_channels = []
        for png_path in sorted(glob.glob(os.path.join(png_dir, '*.png'))):
            image = Image.open(png_path)
            np_array = np.array(image)
            gt_channel = GroundTruthChannel.create_from_integer_array(
                np_array=np_array,
                defects_config=defects_config)
            gt_channels.append(gt_channel)

        return cls(channels=gt_channels, file_path=png_dir)


class AnomalyMap:
    """An anomaly map generated by a model.

    Use AnomalyMap.read_from_tiff(...) to read an AnomalyMap from a
    .tiff file.
    """

    def __init__(self,
                 np_array: np.ndarray,
                 file_path: Optional[str] = None):
        """
        Args:
            np_array: A 2-D numpy array containing the real-valued anomaly
                scores.
            file_path: (optional) file path of the image. Not used for I/O.
        """

        assert len(np_array.shape) == 2

        self.np_array = np_array
        self.file_path = file_path

    def __repr__(self):
        return f'AnomalyMap({get_file_path_repr(self.file_path)})'

    @property
    def size(self):
        return self.np_array.shape

    def get_binary_image(self, anomaly_threshold: float):
        """Return the binary anomaly map based on a given threshold.

        The result is a 2-D numpy array with dtype np.bool_.
        """
        return self.get_binary_images(
            anomaly_thresholds=[anomaly_threshold])[0]

    def get_binary_images(self, anomaly_thresholds: Iterable[float]):
        """Return binary anomaly maps based on given thresholds.

        The result is a 3-D numpy array with dtype np.bool_. The first
        dimension has the same length as the anomaly_thresholds.
        """
        return self._get_binary_images(
            anomaly_thresholds=tuple(anomaly_thresholds))

    @lru_cache(maxsize=3)
    def _get_binary_images(self, anomaly_thresholds: Tuple[float, ...]):
        thresholds = [[[t]] for t in anomaly_thresholds]
        return np.greater(self.np_array[np.newaxis, :, :], thresholds)

    @classmethod
    def read_from_tiff(cls, tiff_path: str):
        """Read an AnomalyMap from a TIFF-file."""
        np_array = tifffile.imread(tiff_path)
        assert len(np_array.shape) == 2
        return cls(np_array=np_array,
                   file_path=tiff_path)
