from abc import ABC, abstractmethod
from typing import Union, List, Optional, Any

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmdepth.registry import BACKBONES, NECKS, PREDICT_HEADERS


class BaseDispHead(BaseModule, ABC):

    def __init__(self, init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(init_cfg)

    def loss(self, *args, data_samples: Optional[list] = None, get_results=False, **kwargs):
        """
        forward in loss mode
        Args:
            *args:
            data_samples:
            get_results
            **kwargs:

        Returns: loss dict

        """
        results = self.forward(*args, **kwargs)
        if not get_results:
            return self.loss_by_results(results, data_samples)
        else:
            return self.loss_by_results(results, data_samples), results

    @abstractmethod
    def loss_by_results(self, results, datasamples):
        """

        Args:
            results:
            datasamples:

        Returns:

        """
        ...

    def predict(self, *args, data_samples: Optional[list] = None, get_results=False, **kwargs):
        """
        forward in predict mode
        Args:
            *args:
            data_samples:
            get_results
            **kwargs:

        Returns: datasample contained with prediction

        """
        results = self.forward(*args, **kwargs)
        if not get_results:
            return self.pack_results(results, data_samples)
        else:
            return self.pack_results(results, data_samples), results

    @abstractmethod
    def pack_results(self, results, datasamples):
        """

        Args:
            results:
            datasamples:

        Returns:

        """
        ...
