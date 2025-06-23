import sys
import gdown
import torch
import urllib.request

from tqdm import tqdm
from pathlib import Path
from .utils import extract_archive
from ..utils.io import combine_dicts
from abc import ABCMeta, abstractmethod

DATASET_CONFIG_KEYS = [
    'data_path', 'pairs_path', 'cache_images', 'max_pairs', 
    'n_workers', 'resize', 'url', 'url_download_extension',
]


class BaseDatasetLoader(metaclass=ABCMeta):
    default_config = {
        'data_path': '',
        'pairs_path': '',
        'cache_images': False,
        'max_pairs': -1,
        'url': '',
        'url_download_extension': ''
    }

    def __init__(self, config:dict):
        self.config = combine_dicts(self.default_config, config)
        
        if not Path(self.config['data_path']).exists():
            if self.config.get('url'):
                self.download_dataset()
            else:
                print('Neither a valid "data_path" nor a "url" was provided to download the dataset.', file=sys.stderr)
                sys.exit(1)  

        self._pairs = self.read_gt()

        self._image_cache = {}
        if self.config['cache_images']:
            self.load_images()

    @abstractmethod
    def load_images(self, attr = 'image') -> None:
        raise NotImplementedError

    @abstractmethod
    def read_image(self, path, attr = 'image') -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def read_gt(self) -> dict:
        raise NotImplementedError

    def download_dataset(self, chunk_size: int = 1024) -> None:
        """
        Faz o download de um arquivo de dataset a partir de uma URL, exibe progresso,
        extrai seu conteúdo e remove o arquivo compactado.

        Args:
            chunk_size (int): Tamanho de leitura em bytes por chunk (padrão: 1024).
        """
        url = self.config.get('url')
        archive_ext = self.config.get('url_download_extension', '')

        output_dir = Path(self.config['data_path'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Definir nome temporário do arquivo
        archive_path = output_dir / f"temp_dataset{archive_ext}"     

        try:
            if 'drive.google' in url:
                gdown.download(url, str(archive_path), quiet=False)

            else:                
                with urllib.request.urlopen(url) as response:
                    total_size = int(response.headers.get("Content-Length", 0))

                    with open(archive_path, 'wb') as out_file, \
                        tqdm(
                            total=total_size,
                            unit='B',
                            unit_scale=True,
                            unit_divisor=1024,
                            desc="Baixando",
                            ncols=80
                        ) as progress_bar:
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            out_file.write(chunk)
                            progress_bar.update(len(chunk))


            extract_archive(
                archive_path,
                output_dir,
                remove_after_extraction=True
            )

        except Exception as e:
            print(f"[!] Erro ao baixar ou extrair dataset: {e}")
            if archive_path.exists():
                archive_path.unlink()
            output_dir.rmdir()

    @property
    def pairs(self) -> list:
        return self._pairs
    
    @property
    def image_cache(self) -> dict:
        return self._image_cache