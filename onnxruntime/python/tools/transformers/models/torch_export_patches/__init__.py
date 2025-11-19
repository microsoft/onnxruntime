from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np
import packaging.version as pv
import torch
from transformers import __version__ as transformers_version
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from .onnx_export_errors import (
    bypass_export_some_errors,
    register_additional_serialization_functions,
)


def is_torchdynamo_exporting() -> bool:
    """Tells if torch is exporting a model."""
    import torch  # noqa: PLC0415

    if not hasattr(torch.compiler, "is_exporting"):
        # torch.compiler.is_exporting requires torch>=2.7
        return False

    try:
        return torch.compiler.is_exporting()
    except Exception:
        try:
            import torch._dynamo as dynamo  # noqa: PLC0415

            return dynamo.is_exporting()  # type: ignore
        except Exception:
            return False


def string_type(
    obj: Any,
    with_shape: bool = False,
    with_min_max: bool = False,
    with_device: bool = False,
    ignore: bool = False,
    limit: int = 20,
) -> str:
    """
    Displays the types of an object as a string.

    :param obj: any
    :param with_shape: displays shapes as well
    :param with_min_max: displays information about the values
    :param with_device: display the device
    :param ignore: if True, just prints the type for unknown types
    :param limit: do not print container with more than limit elements
    :return: str
    """
    if obj is None:
        return "None"

    # tuple
    if isinstance(obj, tuple):
        if len(obj) == 1:
            s = string_type(
                obj[0],
                with_shape=with_shape,
                with_min_max=with_min_max,
                with_device=with_device,
                ignore=ignore,
                limit=limit,
            )
            return f"({s},)"
        if len(obj) < limit:
            js = ",".join(
                string_type(
                    o,
                    with_shape=with_shape,
                    with_min_max=with_min_max,
                    with_device=with_device,
                    ignore=ignore,
                    limit=limit,
                )
                for o in obj
            )
            return f"({js})"
        tt = string_type(
            obj[0],
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            ignore=ignore,
            limit=limit,
        )
        if with_min_max and all(isinstance(_, (int, float, bool)) for _ in obj):
            mini, maxi, avg = min(obj), max(obj), sum(float(_) for _ in obj) / len(obj)
            return f"#{len(obj)}({tt},...)[{mini},{maxi}:A[{avg}]]"
        return f"#{len(obj)}({tt},...)"
    # list
    if isinstance(obj, list):
        if len(obj) < limit:
            js = ",".join(
                string_type(
                    o,
                    with_shape=with_shape,
                    with_min_max=with_min_max,
                    with_device=with_device,
                    ignore=ignore,
                    limit=limit,
                )
                for o in obj
            )
            return f"#{len(obj)}[{js}]"
        tt = string_type(
            obj[0],
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            ignore=ignore,
            limit=limit,
        )
        if with_min_max and all(isinstance(_, (int, float, bool)) for _ in obj):
            mini, maxi, avg = min(obj), max(obj), sum(float(_) for _ in obj) / len(obj)
            return f"#{len(obj)}[{tt},...][{mini},{maxi}:{avg}]"
        return f"#{len(obj)}[{tt},...]"
    # set
    if isinstance(obj, set):
        if len(obj) < 10:
            js = ",".join(
                string_type(
                    o,
                    with_shape=with_shape,
                    with_min_max=with_min_max,
                    with_device=with_device,
                    ignore=ignore,
                    limit=limit,
                )
                for o in obj
            )
            return f"{{{js}}}"
        if with_min_max and all(isinstance(_, (int, float, bool)) for _ in obj):
            mini, maxi, avg = min(obj), max(obj), sum(float(_) for _ in obj) / len(obj)
            return f"{{...}}#{len(obj)}[{mini},{maxi}:A{avg}]"
        return f"{{...}}#{len(obj)}" if with_shape else "{...}"
    # dict
    if isinstance(obj, dict) and type(obj) is dict:
        if len(obj) == 0:
            return "{}"

        import torch  # noqa: PLC0415

        if all(isinstance(k, int) for k in obj) and all(
            isinstance(
                v,
                (
                    str,
                    torch.export.dynamic_shapes._Dim,
                    torch.export.dynamic_shapes._DerivedDim,
                    torch.export.dynamic_shapes._DimHint,
                ),
            )
            for v in obj.values()
        ):
            # This is dynamic shapes
            rows = []
            for k, v in obj.items():
                if isinstance(v, str):
                    rows.append(f"{k}:DYN({v})")
                else:
                    rows.append(f"{k}:{string_type(v)}")
            return f"{{{','.join(rows)}}}"

        kws = dict(
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            ignore=ignore,
            limit=limit,
        )
        s = ",".join(f"{kv[0]}:{string_type(kv[1], **kws)}" for kv in obj.items())
        if all(isinstance(k, int) for k in obj):
            return f"{{{s}}}"
        return f"dict({s})"
    # array
    if isinstance(obj, np.ndarray):
        from .onnx_helper import np_dtype_to_tensor_dtype  # noqa: PLC0415

        if with_min_max:
            s = string_type(obj, with_shape=with_shape)
            if len(obj.shape) == 0:
                return f"{s}={obj}"
            if obj.size == 0:
                return f"{s}[empty]"
            n_nan = np.isnan(obj.reshape((-1,))).astype(int).sum()
            if n_nan > 0:
                nob = obj.ravel()
                nob = nob[~np.isnan(nob)]
                if nob.size == 0:
                    return f"{s}[N{n_nan}nans]"
                return f"{s}[{nob.min()},{nob.max()}:A{nob.astype(float).mean()}N{n_nan}nans]"
            return f"{s}[{obj.min()},{obj.max()}:A{obj.astype(float).mean()}]"
        i = np_dtype_to_tensor_dtype(obj.dtype)
        if not with_shape:
            return f"A{i}r{len(obj.shape)}"
        return f"A{i}s{'x'.join(map(str, obj.shape))}"

    import torch  # noqa: PLC0415

    # Dim, SymInt
    if isinstance(obj, torch.export.dynamic_shapes._DerivedDim):
        return "DerivedDim"
    if isinstance(obj, torch.export.dynamic_shapes._Dim):
        return f"Dim({obj.__name__})"
    if isinstance(obj, torch.SymInt):
        return "SymInt"
    if isinstance(obj, torch.SymFloat):
        return "SymFloat"

    if isinstance(obj, torch.export.dynamic_shapes._DimHint):
        cl = (
            torch.export.dynamic_shapes._DimHintType
            if hasattr(torch.export.dynamic_shapes, "_DimHintType")
            else torch.export.Dim
        )
        if obj in (torch.export.Dim.DYNAMIC, cl.DYNAMIC):
            return "DYNAMIC"
        if obj in (torch.export.Dim.AUTO, cl.AUTO):
            return "AUTO"
        return str(obj)

    if isinstance(obj, bool):
        if with_min_max:
            return f"bool={obj}"
        return "bool"
    if isinstance(obj, int):
        if with_min_max:
            return f"int={obj}"
        return "int"
    if isinstance(obj, float):
        if with_min_max:
            return f"float={obj}"
        return "float"
    if isinstance(obj, str):
        return "str"
    if isinstance(obj, slice):
        return "slice"

    if is_dataclass(obj):
        # That includes torch.export.Dim.AUTO, torch.export.Dim.DYNAMIC so they need to be
        # handled before that.
        values = {f.name: getattr(obj, f.name, None) for f in fields(obj)}
        values = {k: v for k, v in values.items() if v is not None}
        s = string_type(
            values,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            ignore=ignore,
            limit=limit,
        )
        return f"{obj.__class__.__name__}{s[4:]}"

    # Tensors
    if isinstance(obj, torch._subclasses.fake_tensor.FakeTensor):
        from .onnx_helper import torch_dtype_to_onnx_dtype  # noqa: PLC0415

        i = torch_dtype_to_onnx_dtype(obj.dtype)
        prefix = ("G" if obj.get_device() >= 0 else "C") if with_device else ""
        if not with_shape:
            return f"{prefix}F{i}r{len(obj.shape)}"
        return f"{prefix}F{i}s{'x'.join(map(str, obj.shape))}"
    if isinstance(obj, torch.Tensor):
        from .onnx_helper import torch_dtype_to_onnx_dtype  # noqa: PLC0415

        if with_min_max:
            s = string_type(obj, with_shape=with_shape, with_device=with_device)
            if len(obj.shape) == 0:
                return f"{s}={obj}"
            if obj.numel() == 0:
                return f"{s}[empty]"
            n_nan = obj.reshape((-1,)).isnan().to(int).sum()
            if n_nan > 0:
                nob = obj.reshape((-1,))
                nob = nob[~nob.isnan()]
                if obj.dtype in {torch.complex64, torch.complex128}:
                    return f"{s}[{nob.abs().min()},{nob.abs().max():A{nob.mean()}N{n_nan}nans}]"
                return f"{s}[{obj.min()},{obj.max()}:A{obj.to(float).mean()}N{n_nan}nans]"
            if obj.dtype in {torch.complex64, torch.complex128}:
                return f"{s}[{obj.abs().min()},{obj.abs().max()}:A{obj.abs().mean()}]"
            return f"{s}[{obj.min()},{obj.max()}:A{obj.to(float).mean()}]"
        i = torch_dtype_to_onnx_dtype(obj.dtype)
        prefix = ("G" if obj.get_device() >= 0 else "C") if with_device else ""
        if not with_shape:
            return f"{prefix}T{i}r{len(obj.shape)}"
        return f"{prefix}T{i}s{'x'.join(map(str, obj.shape))}"

    if obj.__class__.__name__ == "OrtValue":
        if not obj.has_value():
            return "OV(<novalue>)"
        if not obj.is_tensor():
            return "OV(NOTENSOR)"
        if with_min_max:
            try:
                t = obj.numpy()
            except Exception:
                # pass unable to convert into numpy (bfloat16, ...)
                return "OV(NO-NUMPY:FIXIT)"
            return f"OV({string_type(t, with_shape=with_shape, with_min_max=with_min_max)})"
        dt = obj.element_type()
        shape = obj.shape()
        if with_shape:
            return f"OV{dt}s{'x'.join(map(str, shape))}"
        return f"OV{dt}r{len(shape)}"

    # others classes

    if obj.__class__ in torch.utils._pytree.SUPPORTED_NODES:
        from .cache_helper import flatten_unflatten_for_dynamic_shapes  # noqa: PLC0415

        args = flatten_unflatten_for_dynamic_shapes(obj)
        att = string_type(
            args,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
        )
        return f"{obj.__class__.__name__}[serialized]({att})"

    if type(obj).__name__ == "Node" and hasattr(obj, "meta"):
        # torch.fx.node.Node
        return f"%{obj.target}"
    if type(obj).__name__ == "ValueInfoProto":
        return f"OT{obj.type.tensor_type.elem_type}"

    if obj.__class__.__name__ == "BatchFeature":
        s = string_type(
            obj.data,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
        )
        return f"BatchFeature(data={s})"

    if obj.__class__.__name__ == "BatchEncoding":
        s = string_type(
            obj.data,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
        )
        return f"BatchEncoding(data={s})"

    if obj.__class__.__name__ == "VirtualTensor":
        return f"{obj.__class__.__name__}(name={obj.name!r}, dtype={obj.dtype}, shape={obj.shape})"

    if isinstance(obj, torch.nn.Module):
        return f"{obj.__class__.__name__}(...)"

    if isinstance(obj, (torch.device, torch.dtype, torch.memory_format, torch.layout)):
        return f"{obj.__class__.__name__}({obj})"

    if isinstance(  # TreeSpec, MappingKey, SequenceKey
        obj,
        (
            torch.utils._pytree.TreeSpec,
            torch.utils._pytree.MappingKey,
            torch.utils._pytree.SequenceKey,
        ),
    ):
        return repr(obj).replace(" ", "").replace("\n", " ")

    # to avoid failures

    if obj.__class__.__name__ == "MambaCache":
        c = string_type(
            obj.conv_states,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
        )
        d = string_type(
            obj.ssm_states,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
        )
        return f"MambaCache(conv_states={c}, ssm_states={d})"

    if obj.__class__.__name__ == "DynamicCache":
        kc = string_type(
            obj.key_cache,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
        )
        vc = string_type(
            obj.value_cache,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
        )
        return f"{obj.__class__.__name__}(key_cache={kc}, value_cache={vc})"

    if obj.__class__.__name__ == "EncoderDecoderCache":
        att = string_type(
            obj.self_attention_cache,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
        )
        cross = string_type(
            obj.cross_attention_cache,
            with_shape=with_shape,
            with_min_max=with_min_max,
            with_device=with_device,
            limit=limit,
        )
        return f"{obj.__class__.__name__}(self_attention_cache={att}, cross_attention_cache={cross})"

    raise AssertionError(f"Unsupported type {type(obj).__name__!r} - {type(obj)}")


if pv.Version(transformers_version) > pv.Version("4.49.99999"):

    def make_dynamic_cache(
        key_value_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> DynamicCache:
        """
        Creates an instance of :class:`DynamicCache`.
        This version is valid for ``transformers >= 4.50``.

        :param key_value_pairs: list of pairs of (key, values)
        :return: :class:`DynamicCache`
        """
        return DynamicCache(key_value_pairs)

else:

    def make_dynamic_cache(
        key_value_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> DynamicCache:
        """
        Creates an instance of :class:`DynamicCache`.
        This version is valid for ``transformers < 4.50``.

        :param key_value_pairs: list of pairs of (key, values)
        :return: :class:`DynamicCache`
        """
        cache = DynamicCache(len(key_value_pairs))
        for i, (key, value) in enumerate(key_value_pairs):
            cache.update(key, value, i)
        return cache


def make_encoder_decoder_cache(
    self_attention_cache: DynamicCache,
    cross_attention_cache: DynamicCache,
) -> EncoderDecoderCache:
    "Creates an EncoderDecoderCache."
    return EncoderDecoderCache(self_attention_cache=self_attention_cache, cross_attention_cache=cross_attention_cache)
