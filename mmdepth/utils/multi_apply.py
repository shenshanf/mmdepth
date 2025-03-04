from typing import TypeVar, Sequence, Callable, Any, Tuple, Union, get_type_hints, get_args, get_origin
from collections.abc import Sequence as SequenceABC
from functools import wraps
import inspect
from itertools import repeat


def multi_apply(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any) -> Any:
        if not args:
            return func()

        has_sequence = any(isinstance(arg, SequenceABC) and
                           not isinstance(arg, (str, bytes)) for arg in args)

        if not has_sequence:
            return func(*args)

        max_len = max(len(arg) if isinstance(arg, SequenceABC) and
                                  not isinstance(arg, (str, bytes)) else 1
                      for arg in args)

        sequences = []
        for arg in args:
            if isinstance(arg, SequenceABC) and not isinstance(arg, (str, bytes)):
                if len(arg) == 1:
                    sequences.append(repeat(arg[0], max_len))
                elif len(arg) == max_len:
                    sequences.append(arg)
                else:
                    raise ValueError(f"Sequence length mismatch: {len(arg)} != {max_len}")
            else:
                sequences.append(repeat(arg, max_len))

        results = []
        for items in zip(*sequences):
            result = func(*items)
            results.append(result if isinstance(result, tuple) else (result,))

        return tuple(map(tuple, zip(*results)))

    # try:
    #     orig_hints = get_type_hints(func)
    # except TypeError:
    #     sig = inspect.signature(func)
    #     orig_hints = {param.name: Any for param in sig.parameters.values()}
    #     orig_hints['return'] = Any
    #
    # def make_union_type(t: type) -> type:
    #     if t == Any:
    #         return Any
    #     if get_origin(t) == tuple:
    #         args = get_args(t)
    #         return tuple(Sequence[arg] for arg in args)
    #     return Sequence[t]
    #
    # wrapper.__annotations__ = {
    #     name: make_union_type(t) for name, t in orig_hints.items()
    #     if name != 'return'
    # }
    # wrapper.__annotations__['return'] = make_union_type(orig_hints.get('return', Any))

    return wrapper


if __name__ == "__main__":
    @multi_apply
    def add_mul(a: int, b: int) -> Tuple[int, int]:
        return a + b, a * b


    print(add_mul(1, 2))  # (3, 2)
    print(add_mul([1, 2], [3, 4]))  # ((4, 3), (6, 8))
    print(add_mul([1, 2, 3], 4))  # ((5, 4), (6, 8), (7, 12))
    print(add_mul([1], [2, 3]))  # ((3, 2), (4, 3))