import os
import tqdm
import torch
import numpy as np

from tqdm import tqdm
from ..settings import DATA_PATH
from .base_dataset_loader import BaseDatasetLoader
from ..utils.io import load_image, load_depth, combine_dicts

class Scannet(BaseDatasetLoader):
    def __init__(self, config):
        base_config = {
            'url': 'https://drive.google.com/uc?id=1wtl-mNicxGlXZ-UQJxFnKuWPvvssQBwd',
            'url_download_extension': '.tar',
            'pairs_path': DATA_PATH/'scannet1500_pairs_calibrated.txt',
        }
        config = combine_dicts(base_config, config)

        super().__init__(config)

        self._depth_cache = {}
        if self.config['cache_images']:
            self.load_images('depth')

    def load_images(self, attr = 'image') -> None:
        """
                
        """
        if attr not in ['image', 'depth']:
            raise NotImplementedError('The only attributes allowed are "image" and "depth"')

        attr_name = f'_{attr}_cache'
        cache = self.__getattribute__(attr_name)
        load_func = load_image if attr == 'image' else load_depth

        for pair in tqdm(self.pairs, desc=f'Caching {attr}s'):
            if pair[f'{attr}0'] not in cache:
                cache[pair[f'{attr}0']] = load_func(pair[f'{attr}0'])
            if pair[f'{attr}1'] not in cache:
                cache[pair[f'{attr}1']] = load_func(pair[f'{attr}1'])

    def read_image(self, path:str, attr:str = 'image') -> torch.Tensor:
        """
        
        """
        if self.config['cache_images']:
            return self.image_cache[path]
        else:
            load_func = load_image if attr == 'image' else load_depth
            return load_func(path)

    def read_gt(self) -> dict:
        """
        
        """
        pairs = []
        with self.config['pairs_path'].open() as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.replace('\n', '').replace('  ', '')
                all_info = line.split(' ')
                image0 = all_info[0]
                image1 = all_info[1]

                if len(all_info) != 38:
                    continue

                # scannet format
                K0 = np.array(all_info[4:13]).astype(float).reshape(3, 3)
                K1 = np.array(all_info[13:22]).astype(float).reshape(3, 3)
                T_0to1 = np.array(all_info[22:38]).astype(float).reshape(4, 4)
                
                image0 = os.path.join(self.config['data_path'], image0)
                image1 = os.path.join(self.config['data_path'], image1)
                depth0 = image0.replace('color', 'depth').replace('jpg', 'png')
                depth1 = image1.replace('color', 'depth').replace('jpg', 'png')

                pairs.append({
                    'image0': image0,
                    'image1': image1,
                    'depth0': depth0,
                    'depth1': depth1,
                    'K0': K0,
                    'K1': K1,
                    'T_0to1': T_0to1,
                })

            if self.config['max_pairs'] > 0:
                pairs = pairs[:self.config['max_pairs']]
        return pairs

    @property
    def depth_cache(self) -> dict:
        return self._depth_cache