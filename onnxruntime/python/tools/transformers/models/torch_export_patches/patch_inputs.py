import inspect
from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from . import string_type
from .cache_helper import make_dynamic_cache


def _process_cache(k: str, v):
    assert k != "position_ids" or isinstance(k, torch.Tensor), (
        f"Unexpected type for parameter {k!r} {string_type(v, with_shape=True)}"
    )
    if isinstance(v, list) and all(isinstance(i, tuple) for i in v) and {len(t) for t in v} == {2}:
        # A dynamicCache
        cache = make_dynamic_cache(v)
        return cache
    if isinstance(v, torch.Tensor):
        return v
    raise NotImplementedError(f"Unable to process parameter {k!r} with v={string_type(v, with_shape=True)}")


def _make_shape(subset: dict, cls: type, value: Any) -> Any:
    if cls is DynamicCache:
        assert subset, "DynamicCache cannot be empty"
        values = set(map(str, subset.values()))
        assert len(values) == 1, (
            f"Inconsistencies in subset={subset}, found={values}, it cannot be a {cls}, value={string_type(value)}"
        )
        cache_length = len(value.key_cache)
        for v in subset.values():
            axes = v
            break
        new_shape = [[axes for i in range(cache_length)], [axes for i in range(cache_length)]]
        return new_shape
    raise NotImplementedError(f"_make_shape not implemented for cls={cls}, subset={subset}, value={string_type(value)}")


def convert_dynamic_axes_into_dynamic_shapes(
    model: torch.nn.Module,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    prefix_mapping: dict[str, str] | None = None,
    verbose: int = 0,
) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]:
    """
    Converts the input from an export to something :func:`torch.export.export` can handle.

    :param model: model to convert (used to extract the signature)
    :param args: positional arguments
    :param kwargs: named arguments
    :param dynamic_axes: dynamic axes
    :param prefix_mapping: prefix mapping
    :param verbose: verbosity
    :return: (args, kwargs, dynamic shapes)
    """
    new_kwargs = {}
    if args:
        assert hasattr(model, "forward"), f"Missing method 'forward' for {model!r}"
        plus = 0 if isinstance(model, torch.nn.Module) else 1
        print(
            f"[convert_dynamic_axes_into_dynamic_shapes] "
            f"mapping args to kwargs for model="
            f"{model if plus else model.__class__.__name__}"
        )
        pars = inspect.signature(model.forward).parameters
        assert len(pars) >= len(args), f"Length mismatch, len(args)={len(args)}, pars={list(pars)}"

        for i, p in enumerate(pars):
            if i < plus:
                continue
            if i - plus >= len(args):
                break
            if verbose:
                print(
                    f"[convert_dynamic_axes_into_dynamic_shapes] mapping args[{i - plus}] "
                    f"to {p!r} ({string_type(args[i - plus])})"
                )
            new_kwargs[p] = args[i - plus]

    if kwargs:
        for k, v in kwargs.items():
            assert k not in new_kwargs, f"Argument {k!r} from kwargs already present in args."
            new_kwargs[k] = v

    # process
    updated_kwargs = {}
    changes = {}
    for k, v in new_kwargs.items():
        if isinstance(v, torch.Tensor):
            updated_kwargs[k] = v
            continue
        if isinstance(v, list):
            # cache?
            updated_kwargs[k] = _process_cache(k, v)
            if type(updated_kwargs[k]) is not type(v):
                # A cache was introduced.
                if verbose:
                    print(
                        f"[convert_dynamic_axes_into_dynamic_shapes] parameter "
                        f"{k!r} was changed into {type(updated_kwargs[k])}"
                    )
                changes[k] = type(updated_kwargs[k])
                continue
        raise NotImplementedError(f"Unexpected type {type(v)} for parameter {k!r} ({string_type(v, with_shape=True)})")

    # process dynamic axes
    if changes:
        dynamic_shapes = {}
        done = set()
        for k, v in dynamic_axes.items():
            if k not in changes and k in updated_kwargs and isinstance(v, dict):
                dynamic_shapes[k] = v
                continue
            if "." in k:
                # something like present.0.key
                prefix = k.split(".")[0]
                if prefix in done:
                    continue
                args_prefix = prefix_mapping[prefix] if prefix_mapping and prefix in prefix_mapping else prefix
                if args_prefix in updated_kwargs and args_prefix in changes:
                    # A cache.
                    cls = changes[args_prefix]
                    dynamic_shapes[args_prefix] = _make_shape(
                        {_: __ for _, __ in dynamic_axes.items() if _.startswith(f"{prefix}.")},
                        cls,
                        updated_kwargs[args_prefix],
                    )
                    done.add(prefix)
                    continue
            if k not in updated_kwargs:
                # dynamic axes not in the given inputs, should be raise an exception?
                if verbose:
                    print(
                        f"[convert_dynamic_axes_into_dynamic_shapes] dropping axes "
                        f"{k!r}-{v!r}, not found in {set(updated_kwargs)}"
                    )
                continue
            raise NotImplementedError(
                f"Unable to process dynamic axes {k!r}, axes={v}, "
                f"value={string_type(updated_kwargs[k], with_shape=True)}, "
                f"dynamic axes={dynamic_axes}, "
                f"updated_kwargs={string_type(updated_kwargs, with_shape=True)}"
            )

    return (), updated_kwargs, dynamic_shapes


def replace_dynamic_shapes(ds, mapping, default_value):
    if isinstance(ds, dict) and all(isinstance(k, int) for k in ds):
        new_ds = {}
        for k, v in ds.items():
            if isinstance(v, str):
                new_ds[k] = mapping.get(v, default_value)
            else:
                new_ds[k] = v
        return new_ds
    if isinstance(ds, tuple):
        return tuple(replace_dynamic_shapes(d, mapping, default_value) for d in ds)
    if isinstance(ds, list):
        return [replace_dynamic_shapes(d, mapping, default_value) for d in ds]
    if isinstance(ds, dict):
        return {k: replace_dynamic_shapes(v, mapping, default_value) for k, v in ds.items()}
    return ds
