from semmatch.utils.models import load_sam, get_object_mask

from semmatch.configs.base import Config
from semmatch.statistics.data_generators.base import DataGenerator
from semmatch.statistics.pipeline_data import RawDataInput, MaskData


class MaskGenerator(DataGenerator):
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
        # masks0 = get_object_mask(
        #     self.sam, raw_data.image0, raw_data.mkpts0, self._config.mask_batch_size)
        masks1 = get_object_mask(
            self.sam, raw_data.image1, raw_data.mkpts1, self._config.mask_batch_size)

        return [
            MaskData(
                # masks0=masks0,
                masks1=masks1
            )
        ]
