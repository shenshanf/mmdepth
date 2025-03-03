from typing import Dict, Union, Tuple, Optional, List
import numpy as np

import mmcv
from mmcv.transforms import BaseTransform
from mmengine.fileio import get

from mmdepth.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImageFromFile(BaseTransform):
    """Load and process image from file using byte stream.

    Args:
        to_float32 (bool): Convert images to float32. Defaults to False.
        color_type (str): Color type processing. Options are:
            - 'color': Ensure image is in color (3 channels)
            - 'grayscale': Convert to grayscale
            - 'unchanged': Keep original format
            Defaults to 'color'.
        channel_order (str): Channel order for color images. Options are:
            - 'bgr': BGR order (OpenCV default)
            - 'rgb': RGB order
            Defaults to 'bgr'.
        imdecode_backend (str): The image decoding backend. Options are:
            - 'cv2': OpenCV backend
            - 'pillow': PIL backend
            - 'turbojpeg': libjpeg-turbo backend
            Defaults to 'cv2'.
    """

    def __init__(self,
                 img_key='imgs',
                 to_float32: bool = False,
                 color_type: str = 'unchanged',
                 channel_order: str = 'rgb',  # bgr -> rgb
                 imdecode_backend: str = 'cv2',
                 backend_args: dict = None, ) -> None:
        super().__init__()
        if color_type not in ['color', 'grayscale', 'unchanged']:
            raise ValueError(f'Invalid color_type {color_type}, must be one of: '
                             f'["color", "grayscale", "unchanged"]')

        if channel_order not in ['bgr', 'rgb']:
            raise ValueError(f'Invalid channel_order {channel_order}, '
                             f'must be either "bgr" or "rgb"')

        self.img_key = img_key

        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.imdecode_backend = imdecode_backend
        self.backend_args = backend_args

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """Transform function to load and process images.
        keys:
            - file_paths

        add keys:
            - self.img_key
            - img_shape
            - ori_shape
            - img_path

        """
        file_paths = results.get('file_paths', None)
        if file_paths is None or self.img_key not in file_paths:
            raise ValueError(f"Missing image file paths, "
                             f"please set '{self.img_key}' key")

        if not isinstance(file_paths, dict):
            raise ValueError("file_paths should be a dict")

        assert isinstance(file_paths, Dict), "'file_paths' must be a dict"
        img_paths = file_paths.get(self.img_key, None)
        assert isinstance(img_paths, List)

        # Load and process images
        imgs = []
        for img_path in img_paths:
            try:
                # Load as byte stream
                img_bytes = get(img_path, self.backend_args)

                # Decode image based on color type
                img = mmcv.imfrombytes(
                    img_bytes,
                    flag=self.color_type,
                    channel_order=self.channel_order,
                    backend=self.imdecode_backend
                )

                # Convert data type if needed
                if self.to_float32:
                    img = img.astype(np.float32)

                imgs.append(img)
            except Exception as e:
                raise ValueError(f'Failed to process image {img_path}: {str(e)}')

        # Update data sample
        results[self.img_key] = imgs
        # recode metainfo
        results['img_shape'] = imgs[0].shape[:2]
        results['ori_shape'] = imgs[0].shape[:2]
        results['img_path'] = img_paths[0]  # use left image path
        return results

    def __repr__(self) -> str:
        """Return string representation."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f'color_type="{self.color_type}", '
        repr_str += f'channel_order="{self.channel_order}", '
        repr_str += f'imdecode_backend="{self.imdecode_backend}")'
        return repr_str
