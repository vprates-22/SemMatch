import os
import tqdm
import torch
import numpy as np

from tqdm import tqdm
from ..settings import DATA_PATH
from .base_dataset_loader import BaseDatasetLoader
from ..utils.io import load_image, load_depth, combine_dicts

class MegaDepth(BaseDatasetLoader):
    def __init__(self, config):
        base_config = {
            'url': 'https://cvg-data.inf.ethz.ch/megadepth/megadepth1500.zip',
            'url_download_extension': '.zip',
            'pairs_path': DATA_PATH/'megadepth1500_pairs_calibrated.txt',
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

                if len(all_info) != 32:
                    continue

                K0 = np.array(all_info[2:11]).astype(float).reshape(3, 3)
                K1 = np.array(all_info[11:20]).astype(float).reshape(3, 3)
                
                pose_elems = all_info[20:32]
                R, t = pose_elems[:9], pose_elems[9:12]
                R = np.array([float(x) for x in R]).reshape(3, 3).astype(np.float32)
                t = np.array([float(x) for x in t]).astype(np.float32)
                T_0to1 = np.eye(4)
                T_0to1[:3, :3] = R
                T_0to1[:3, 3] = t

                depth0 = os.path.join(self.config['data_path'], 'depths', image0).replace('.jpg', '.h5')
                depth1 = os.path.join(self.config['data_path'], 'depths', image1).replace('.jpg', '.h5')
                image0 = os.path.join(self.config['data_path'], 'images', image0)
                image1 = os.path.join(self.config['data_path'], 'images', image1)
                
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