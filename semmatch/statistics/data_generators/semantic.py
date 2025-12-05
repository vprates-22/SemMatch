"""
Module: semmatch.statistics.data_generators.semantic
----------------------------------------------------

This module defines data generators specifically designed for semantic analysis
within the SemMatch statistics pipeline. These generators process raw input data
to produce structured semantic information such as object masks, which are then
consumed by data analyzers.

This module provides data generators for semantic analysis, specifically for
generating segmentation masks for keypoints using models like SAM (Segment Anything Model).
"""
from semmatch.utils.models import load_sam, get_object_mask

from semmatch.configs.base import Config
from semmatch.statistics.data_generators.base import DataGenerator
from semmatch.statistics.pipeline_data import RawDataInput, MaskData


class MaskGenerator(DataGenerator):
    """
    Generates semantic segmentation masks for matched keypoints using a specified model.

    This generator uses models like SAM (Segment Anything Model) to predict binary
    segmentation masks around keypoints in the input images.

    Parameters
    ----------
    config : Config, optional
        Configuration object for the generator.
        Expected keys:
        - 'mask_model' (str): The name of the mask generation model to use (e.g., 'sam').
          Defaults to 'sam'.
        - 'sam_model' (str): The specific SAM model checkpoint to load (e.g., 'sam2.1_l.pt').
          Defaults to 'sam2.1_l.pt'.
        - 'mask_batch_size' (int): The number of keypoints to process in a single batch
          when generating masks. Defaults to 200.
    """

    def __init__(self, config=None):
        default_config = Config({
            'mask_model': 'sam',
            'sam_model': 'sam2.1_l.pt',
            # 'sam_model': 'sam2.1_l.pt',
            # 'sam_model': 'sam2.1_l.pt',
            # 'sam_model': 'sam2.1_l.pt',
            'mask_batch_size': 200,
        })
        super().__init__(default_config.merge_config(config))

        if self._config.mask_model == 'sam':
            self.sam = load_sam(self._config.sam_model)
        else:
            pass

    def generate(self, raw_data: RawDataInput) -> list[MaskData]:
        """
        Generates semantic masks for keypoints in the second image (`raw_data.mkpts1`).

        This method uses the configured mask generation model (e.g., SAM) to predict
        segmentation masks for each keypoint in `raw_data.mkpts1` within `raw_data.image1`.
        Currently, it only generates masks for the second image.

        Parameters
        ----------
        raw_data : RawDataInput
            The raw input data containing image information and keypoints.
            - `raw_data.image1`: The second image as a NumPy array.
            - `raw_data.mkpts1`: Keypoints from the second image.

        Returns
        -------
        list[MaskData]
            A list containing a single `MaskData` object, which includes the
            generated masks for `mkpts1` in `image1`.
        """
        # masks0 = get_object_mask(
        #     self.sam, raw_data.image0, raw_data.mkpts0, self._config.mask_batch_size)
        masks1 = get_object_mask(
            self.sam, raw_data.image1, raw_data.mkpts1, self._config.mask_batch_size)

        return [
            MaskData(
                masks1=masks1
            )
        ]
