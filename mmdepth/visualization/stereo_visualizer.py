from typing import Dict, List, Optional, Union, Tuple, Any
import os.path
import numpy as np
import mmcv
from mmengine.visualization import Visualizer
from mmdepth.registry import VISUALIZERS, TRANSFORMS
from mmdepth.structures import BaseDataSample, DispMap, MultiDispMap, ErrorMap

from matplotlib.cm import ScalarMappable


@VISUALIZERS.register_module()
class StereoVisualizer(Visualizer):

    def __init__(self,
                 name: str = 'stereo_visualizer',
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 transforms: Optional[Dict] = None,
                 disp_cmap: str = 'turbo',
                 disp_color_mapper: Optional[ScalarMappable] = None,
                 emap_mapper: Optional[ScalarMappable] = None,
                 vis_right_img: bool = True,
                 vis_right: bool = False,
                 vis_multi: bool = True) -> None:
        super().__init__(
            name=name,
            vis_backends=vis_backends,
            save_dir=save_dir)

        # transform before visualization
        self.transforms = TRANSFORMS.build(transforms) if transforms else None

        # whether visual right side data if exist
        self.vis_right = vis_right
        self.vis_right_img = vis_right_img

        # whether visual multi level prediction data if exist
        self.vis_multi = vis_multi

        self.disp_cmap = disp_cmap

        # todo: default
        self.disp_color_mapper = disp_color_mapper
        self.emap_mapper = emap_mapper

        # Set default value. When calling
        # `DetLocalVisualizer().dataset_meta=xxx`,
        # it will override the default value.
        self.dataset_meta = {}

    def draw_disp(self,
                  disp_data: Optional[Union[DispMap, MultiDispMap]],
                  name: str) -> Dict:
        """Draw and save disparity map visualization.

        Args:
            disp_data: Disparity data to visualize (DispMap or MultilevelData[DispMap])
            name: The visualization identifier


        Returns:
            Optional[Union[np.ndarray, List[np.ndarray]]]:
                - For DispMap: The pseudo-colored disparity image
                - For MultilevelData: List of pseudo-colored disparity images
                - None if no valid image was generated
        """
        if disp_data is None:
            return {}

        norm_max = self.dataset_meta.get('disp_max')
        norm_min = self.dataset_meta.get('disp_min')

        if isinstance(disp_data, DispMap) or (isinstance(disp_data, MultiDispMap) and not self.vis_multi):
            disp_data = disp_data[-1] if isinstance(disp_data, MultiDispMap) else disp_data
            psd_img = disp_data.to_psd_img(norm_max=norm_max,
                                           norm_min=norm_min,
                                           cmap=self.disp_cmap,
                                           color_mapper=self.disp_color_mapper)
            if psd_img is not None:
                return {name: psd_img}
            else:
                return {}

        elif isinstance(disp_data, MultiDispMap) and self.vis_multi:
            results = dict()
            for i, disp in enumerate(disp_data):
                psd_img = disp.to_psd_img(norm_max=norm_max,
                                          norm_min=norm_min,
                                          cmap=self.disp_cmap,
                                          color_mapper=self.disp_color_mapper)
                if psd_img is not None:
                    results[f'{name}_s{i}'] = psd_img
            return results
        else:
            raise TypeError(f'Unsupported disparity data type: {type(disp_data)}')

    def draw_error_map(self,
                       gt_disp: Optional[DispMap],
                       pr_disp: Optional[Union[DispMap, MultiDispMap]],
                       name: str) -> Dict:
        """Draw error map visualization between ground truth and prediction.

        Args:
            gt_disp: Ground truth disparity map
            pr_disp: Predicted disparity map, can be single level or multi-level
            name: The visualization identifier

        Returns:
            Optional[Union[np.ndarray, List[np.ndarray]]]:
                The pseudo-colored error map image(s), None if inputs invalid
        """
        if gt_disp is None or pr_disp is None:
            return dict()

        if isinstance(pr_disp, DispMap) or (isinstance(pr_disp, MultiDispMap) and not self.vis_multi):
            pr_disp = pr_disp[-1] if isinstance(pr_disp, MultiDispMap) else pr_disp
            # Single scale prediction
            error_map = ErrorMap.kitti_d1(gt_disp=gt_disp, pr_disp=pr_disp)
            psd_img = error_map.to_psd_img(self.emap_mapper)
            if psd_img is not None:
                # todo: draw metric text on psdcolor error map
                return {name: psd_img}
            else:
                return {}

        elif isinstance(pr_disp, MultiDispMap) and self.vis_multi:
            # Multi-scale prediction
            results = dict()
            for i, _pr_disp in enumerate(pr_disp):
                error_map = ErrorMap.kitti_d1(gt_disp=gt_disp, pr_disp=_pr_disp)
                psd_img = error_map.to_psd_img(self.emap_mapper)
                if psd_img is not None:
                    results[f'{name}_s{i}'] = psd_img
            return results

        else:
            raise TypeError(f'Unsupported prediction type: {type(pr_disp)}')

    def add_datasample(self,
                       name: str = '',
                       image: Optional[np.ndarray] = None,
                       data_sample: Optional[BaseDataSample] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       show: bool = False,
                       wait_time: int = 0,
                       step: int = 0) -> None:
        """Add a stereo data sample visualization.

        Args:
            name: Base name for the visualizations
            image: Ignored, stereo data uses imgs from data_sample
            data_sample: Stereo data sample containing images and disparities
            draw_gt: Whether to draw ground truth visualizations
            draw_pred: Whether to draw prediction visualizations
            show: Whether to display visualizations
            wait_time: Wait time for display
            step: Global step for logging
        """
        if image is not None:
            raise NotImplementedError(
                "Image input is ignored, use 'imgs' data field in data_sample")
        if data_sample is None:
            return

        if self.transforms is not None:
            data_sample = self.transforms(data_sample)

        draw_imgs = dict()

        # Process stereo image pairs
        draw_imgs[f'img_left'] = data_sample.img_left

        if self.vis_right_img:
            draw_imgs[f'img_right'] = data_sample.img_right

        # Process ground truth
        if draw_gt:
            draw_imgs.update(
                self.draw_disp(data_sample.gt_disp, f'gt_disp'))

            if self.vis_right:
                draw_imgs.update(
                    self.draw_disp(data_sample.gt_disp_right, f'gt_disp_right'))

        # Process predictions
        if draw_pred:
            # process pred disparity
            draw_imgs.update(
                self.draw_disp(data_sample.pr_disp, f'pred_disp'))

            # process error map
            draw_imgs.update(
                self.draw_error_map(data_sample.gt_disp,
                                    data_sample.pred_disp,
                                    f'emap'))

            if self.vis_right:
                draw_imgs.update(
                    self.draw_disp(data_sample.pr_disp_right, f'pred_disp_right'))

                draw_imgs.update(
                    self.draw_error_map(data_sample.gt_disp_right,
                                        data_sample.pred_disp_right,
                                        f'emap_right'))

        # Show visualizations
        # note: in show mode, the data will not save in visualization backend
        for k, img in draw_imgs.items():
            if show:
                mmcv.imshow(img, win_name=f"{self.instance_name}: {k}", wait_time=wait_time)
            else:
                self.add_image(name=f"{name}/{k}", image=img, step=step)
