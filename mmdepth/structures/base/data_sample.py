from typing import Any, Dict, Iterator, Optional, Tuple, List, Type, Union, Callable
from deprecated import deprecated
import copy
import torch
import numpy as np


class BaseDataSample:
    """Basic data structure for samples in training/testing/validation."""

    def __init__(self,
                 *,
                 metainfo: Optional[dict] = None,
                 assets: Optional[dict] = None,
                 **kwargs) -> None:
        self._metainfo_fields: set = set()
        self._asset_fields: set = set()
        self._result_fields: set = set()

        if metainfo is not None:
            self.set_metainfo(metainfo)
        if assets is not None:
            self.set_asset(assets)
        if kwargs:
            self.set_result(kwargs)

    def set_metainfo(self, metainfo: dict) -> None:
        """set data to metainfo"""
        assert isinstance(metainfo, dict), \
            f'metainfo should be a dict, but got {type(metainfo)}'
        meta = copy.deepcopy(metainfo)
        for k, v in meta.items():
            self.set_field(name=k, value=v, field_type='metainfo', dtype=None)

    def set_asset(self, assets: dict) -> None:
        """set data to asset
        note: property will set to asset field, see __setattr__ function
        """
        assert isinstance(assets, dict), \
            f'assets should be a dict, but got {type(assets)}'
        for k, v in assets.items():
            setattr(self, k, v)  # do not use set_field for property method

    def set_result(self, results: dict) -> None:
        """set data to result"""
        assert isinstance(results, dict), \
            f'results should be a dict, but got {type(results)}'
        for k, v in results.items():
            self.set_field(name=k, value=v, field_type='result', dtype=None)

    def set_field(self,
                  value: Any,
                  name: str,
                  dtype: Optional[Union[Type, Tuple[Type, ...]]] = None,
                  field_type: str = 'asset') -> None:
        assert field_type in ['metainfo', 'asset', 'result']

        # note: Different from mmengine, check elements type in container(List/Tuple/Dict)
        if dtype is not None:
            if isinstance(value, (List, Tuple)):
                for v in value:
                    assert isinstance(v, dtype), \
                        f'{name} should be a Sequence of {dtype}, but got {type(v)}'
            elif isinstance(value, Dict):
                for v in value.values():
                    assert isinstance(v, dtype), \
                        f'{name} should be a Dict of {dtype}, but got {type(v)}'
            else:
                assert isinstance(value, dtype), \
                    f'{name} should be a {dtype}, but got {type(value)}'

        # note: The key to be set in a domain cannot appear in other domains
        #       but can appear in its own domain, equivalent to updating the value
        if field_type == 'metainfo':
            if name in self._asset_fields | self._result_fields:
                raise AttributeError(
                    f'Cannot set {name} to be a field of metainfo '
                    f'because {name} is already in other field')
            self._metainfo_fields.add(name)
        elif field_type == 'asset':
            if name in self._metainfo_fields | self._result_fields:
                raise AttributeError(
                    f'Cannot set {name} to be a field of asset '
                    f'because {name} is already in other field')
            self._asset_fields.add(name)
        else:  # result
            if name in self._metainfo_fields | self._asset_fields:
                raise AttributeError(
                    f'Cannot set {name} to be a field of result '
                    f'because {name} is already in other field')
            self._result_fields.add(name)

        super().__setattr__(name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute in the sample."""
        if name in ('_metainfo_fields', '_asset_fields', '_result_fields'):
            if not hasattr(self, name):  # note: only can set in initial method
                super().__setattr__(name, value)
            else:
                raise AttributeError(f'{name} has been used as a '
                                     'private attribute, which is immutable.')
        else:
            self.set_field(name=name, value=value, field_type='asset', dtype=None)

    def __delattr__(self, item: str) -> None:
        """delete attribute"""
        if item in ('_metainfo_fields', '_asset_fields', '_result_fields'):
            raise AttributeError(f'{item} is used as a private attribute '
                                 'and is immutable')

        if item in self._metainfo_fields:
            self._metainfo_fields.remove(item)
            super().__delattr__(item)
        elif item in self._asset_fields:
            self._asset_fields.remove(item)
            super().__delattr__(item)
        elif item in self._result_fields:
            self._result_fields.remove(item)
            super().__delattr__(item)
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{item}'")

    __delitem__ = __delattr__

    def get(self, key: str, default: Any = None) -> Any:
        """get attribute"""
        return getattr(self, key, default)

    def metainfo_keys(self) -> list:
        """Get all keys from metainfo fields."""
        return list(self._metainfo_fields)

    def asset_keys(self) -> List:
        # We assume that the name of the attribute related to property is
        # '_' + the name of the property. We use this rule to filter out
        # private keys.
        # TODO: Use a more robust way to solve this problem
        private_keys = {
            '_' + key
            for key in self._asset_fields
            if isinstance(getattr(type(self), key, None), property)}
        return list(self._asset_fields - private_keys)

    def result_keys(self) -> list:
        """Get all keys from result fields."""
        # We assume that the name of the attribute related to property is
        # '_' + the name of the property. We use this rule to filter out
        # private keys.
        # TODO: Use a more robust way to solve this problem
        private_keys = {
            '_' + key
            for key in self._result_fields
            if isinstance(getattr(type(self), key, None), property)}
        return list(self._result_fields - private_keys)

    def keys(self) -> List:
        """ asset + result"""
        return self.asset_keys() + self.result_keys()

    def all_keys(self) -> List:
        """metainfo+asset+result"""
        return self.metainfo_keys() + self.asset_keys() + self.result_keys()

    def metainfo_values(self) -> list:
        """Get all values from metainfo fields."""
        return [getattr(self, k) for k in self.metainfo_keys()]

    def asset_values(self) -> list:
        """Get all values from result fields."""
        return [getattr(self, k) for k in self.asset_keys()]

    def result_values(self) -> list:
        """Get all values from result fields."""
        return [getattr(self, k) for k in self.result_keys()]

    def values(self) -> list:
        """Get all values from asset and result fields."""
        return [getattr(self, k) for k in self.keys()]

    def all_values(self):
        """ """
        return [getattr(self, k) for k in self.all_keys()]

    def metainfo_items(self) -> Iterator[Tuple[str, Any]]:
        """Get all items from metainfo fields."""
        for k in self.metainfo_keys():
            yield k, getattr(self, k)

    def asset_items(self) -> Iterator[Tuple[str, Any]]:
        """Get all items from result fields."""
        for k in self.asset_keys():
            yield k, getattr(self, k)

    def result_items(self) -> Iterator[Tuple[str, Any]]:
        """Get all items from result fields."""
        for k in self.result_keys():
            yield k, getattr(self, k)

    def items(self) -> Iterator[Tuple[str, Any]]:
        """Get all items from asset fields."""
        for k in self.keys():
            yield k, getattr(self, k)

    def all_items(self) -> Iterator[Tuple[str, Any]]:
        """Get all items from asset fields."""
        for k in self.all_keys():
            yield k, getattr(self, k)

    @property
    def metainfo(self) -> dict:
        return dict(self.metainfo_items())

    def pop(self, *args) -> Any:
        assert len(args) < 3, '`pop` got more than 2 arguments'
        name = args[0]

        if name in self._metainfo_fields:
            self._metainfo_fields.remove(name)
            return self.__dict__.pop(*args)
        elif name in self._asset_fields:
            self._asset_fields.remove(name)
            return self.__dict__.pop(*args)
        elif name in self._result_fields:
            self._result_fields.remove(name)
            return self.__dict__.pop(*args)
        elif len(args) == 2:
            return args[1]
        else:
            raise KeyError(f'{name} is not contained in any fields')

    def __contains__(self, item: str) -> bool:
        return (item in self._metainfo_fields or
                item in self._asset_fields or
                item in self._result_fields)

    def new(self,
            *,
            metainfo: Optional[dict] = None,
            assets: Optional[dict] = None,
            results: Optional[dict] = None) -> 'BaseDataSample':
        """new instance"""
        new_data = self.__class__()

        # copy or set metainfo
        if metainfo is not None:
            new_data.set_metainfo(metainfo)
        else:
            new_data.set_metainfo(dict(self.metainfo_items()))

        # asset
        if assets is not None:
            new_data.set_asset(assets)
        else:
            new_data.set_asset(dict(self.asset_items()))

        # result
        if results is not None:
            new_data.set_result(results)
        else:
            new_data.set_result(dict(self.result_items()))

        return new_data

    def new_empty(self, copy_metainfo=True):
        new_data = self.__class__()
        if copy_metainfo:
            new_data.set_metainfo(dict(self.metainfo_items()))
        return new_data

    def clone(self):
        """Deep copy the current data element."""

        def _deep_copy(x):
            if isinstance(x, torch.Tensor):
                return x.clone()
            if isinstance(x, np.ndarray):
                return x.copy()
            return copy.deepcopy(x)

        return self._convert(self, object, _deep_copy)

    # covert method
    def _convert(self, data: Any, apply_to: Type, func: Callable, *args, **kwargs) -> Any:
        """"""
        if data is None:
            return None

        if isinstance(data, (torch.Tensor, np.ndarray)):
            # note: skip if not 'apply_to' type
            return func(data, *args, **kwargs) if isinstance(data, apply_to) else data

        if isinstance(data, BaseDataSample):
            # new_data with empty asset and result field
            new_data = data.new_empty(copy_metainfo=True)
            for k, v in data.asset_items():
                new_data.set_field(value=self._convert(v, apply_to, func, *args, **kwargs),
                                   name=k, dtype=None, field_type='asset')
            for k, v in data.result_items():
                new_data.set_field(value=self._convert(v, apply_to, func, *args, **kwargs),
                                   name=k, dtype=None, field_type='result')
            return new_data

        if isinstance(data, (List, Tuple)):
            new_data = [self._convert(v, apply_to, func, *args, **kwargs) for v in data]
            return tuple(new_data) if isinstance(data, tuple) else new_data

        if isinstance(data, Dict):
            return {k: self._convert(v, apply_to, func, *args, **kwargs) for k, v in data.items()}

        raise TypeError(f'Unsupported type {type(data)} for conversion')

    def to(self, *args, **kwargs) -> 'BaseDataSample':
        """Apply same name function to all tensors in data_fields."""
        return self._convert(self, torch.Tensor, lambda x: x.to(*args, **kwargs))

    # Tensor-like methods
    def cpu(self) -> 'BaseDataSample':
        """Convert all tensors to CPU."""
        return self._convert(self, torch.Tensor, lambda x: x.cpu())

    def cuda(self) -> 'BaseDataSample':
        """Convert all tensors to CUDA."""
        return self._convert(self, torch.Tensor, lambda x: x.cuda())

    def musa(self) -> 'BaseDataSample':
        """Convert all tensors to MUSA."""
        return self._convert(self, torch.Tensor, lambda x: x.musa())

    # Tensor-like methods
    def npu(self) -> 'BaseDataSample':
        """Convert all tensors to NPU in data."""
        return self._convert(self, torch.Tensor, lambda x: x.npu())

    def mlu(self) -> 'BaseDataSample':
        """Convert all tensors to MLU in data."""
        return self._convert(self, torch.Tensor, lambda x: x.mlu())

    # Tensor-like methods
    def detach(self) -> 'BaseDataSample':
        """Detach all tensors in data."""
        return self._convert(self, torch.Tensor, lambda x: x.detach())

    # Tensor-like methods
    def numpy(self) -> 'BaseDataSample':
        """Convert all tensors to np.ndarray in data."""
        return self._convert(self, torch.Tensor, lambda x: x.detach().cpu().numpy())

    def to_tensor(self) -> 'BaseDataSample':
        """Convert all np.ndarray to tensor in data."""
        return self._convert(self, np.ndarray, lambda x: torch.from_numpy(x))

    def to_dict(self) -> dict:
        """Convert BaseDataSample to dict."""
        return {
            k: v.to_dict() if isinstance(v, BaseDataSample) else v
            for k, v in self.all_items()
        }
