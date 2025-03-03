from typing import Dict, Union, List, Optional, Tuple
from mmcv.transforms import BaseTransform

from .disp_parser import BaseDispParser
from mmdepth.fileio import disp_read
from mmdepth.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadDisparityFromFile(BaseTransform):
    """Load disparity maps from file using byte stream."""

    def __init__(self,
                 disp_key: str = 'gt_disps',
                 imdecode_backend='cv2',
                 backend_args: dict = None,
                 parser: Dict = dict(type='LinearDispParser',
                                     scale=256.0,
                                     offset=0.0,
                                     operation='div',
                                     invalid_value=0.0,
                                     transform_first=False)) -> None:
        super().__init__()
        self.disp_key = disp_key
        self.imdecode_backend = imdecode_backend
        self.backend_args = backend_args
        self.disp_parser: BaseDispParser = TRANSFORMS.build(parser)

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """Transform function to load and parse disparity maps.

        keys:
            - file_paths

        add keys:
            - self.disp_key
            - disp_parser
            - disp_suffix
        """
        file_paths = results.get('file_paths', None)
        if file_paths is None or self.disp_key not in file_paths:
            raise ValueError(f"Missing disparity file paths, "
                             f"please set {self.disp_key}")

        assert isinstance(file_paths, Dict), "'file_paths' must be a dict"
        disp_paths = file_paths.get(self.disp_key, None)
        assert isinstance(disp_paths, List)

        assert len(disp_paths) > 0, "empty disp_paths"

        gt_disps = []
        ext = None
        for disp_path in disp_paths:
            # Load raw bytes
            try:
                raw_disp = disp_read(disp_path)
            except Exception as e:
                raise ValueError(f'Failed to process image {disp_path}: {str(e)}')

            # Parse using provided parser
            gt_disps.append(self.disp_parser(raw_disp))

        results[self.disp_key] = gt_disps

        # recorde metainfo
        results['disp_parser'] = type(self.disp_parser).__name__
        results['disp_suffix'] = ext

        return results
