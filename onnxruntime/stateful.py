# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Typed configuration for native stateful ONNX Runtime execution.

Ordinary InferenceSession objects remain stateless. Configurations capture their
collections by value; execution feeds retain caller storage without copying.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from os import PathLike, fspath
from threading import Lock
from types import MappingProxyType
from typing import Self

import numpy as np

from .capi import onnxruntime_pybind11_state as _C
from .capi.onnxruntime_inference_collection import InferenceSession, OrtValue


class ExitKind(IntEnum):
    """Explicit native loop termination conditions."""

    END_OF_INPUTS = 0
    MAX_ITERATIONS = 1
    MAX_GENERATED_TOKENS = 2
    EOS = 3
    BOOLEAN_OUTPUT = 4


class SourceKind(IntEnum):
    """Distinct chain input source namespaces."""

    EXTERNAL_INPUT = 0
    EXTERNAL_SEQUENCE = 1
    COMPONENT_OUTPUT = 2
    STATE = 3


class OutputKind(IntEnum):
    """Selected named chain output kinds."""

    MODEL_OUTPUT = 0
    GENERATED_TOKEN_IDS = 1


class OutputGroup(IntEnum):
    """Immutable result value collections."""

    SELECTED_OUTPUTS = 0
    COMPONENT_OUTPUTS = 1
    STATE_VALUES = 2


class CacheUpdate(IntEnum):
    """Native cache storage update semantics."""

    SHARE = 0
    COPY = 1
    APPEND = 2


class ElementType(IntEnum):
    """Supported cache tensor element types, using ONNX numeric identifiers."""

    FLOAT32 = 1
    FLOAT16 = 10
    BFLOAT16 = 16


class GenerationPhase(IntEnum):
    """Explicit native token-generation phase."""

    PREFILL = 1
    DECODE = 2


class SamplingAlgorithm(IntEnum):
    """Supported native token-sampling algorithms."""

    GREEDY = 0


class ProviderMode(IntEnum):
    """Independent stateful feature implementation selection."""

    CPU = 0
    PREFER = 1
    REQUIRE = 2


class DeviceType(IntEnum):
    """Native memory device categories."""

    CPU = 0
    GPU = 1
    FPGA = 2
    NPU = 3


class ProviderFeature(IntEnum):
    """Stateful provider capability families."""

    CACHE = 0
    SAMPLER = 1


class ProviderOperation(IntEnum):
    """Individually queryable stateful provider operations."""

    CACHE_INITIALIZE = 0
    CACHE_SHARE = 1
    CACHE_COPY = 2
    CACHE_APPEND = 3
    CACHE_RESERVE = 4
    CACHE_RESET = 5
    SAMPLE_GREEDY = 6


class GuidanceMode(IntEnum):
    """Classifier-free guidance invocation convention."""

    BATCHED = 0
    SEQUENTIAL = 1


class TimestepFormat(IntEnum):
    """Explicit DDIM model timestep encoding."""

    INT64_SCALAR = 0
    INT64_VECTOR = 1
    FLOAT32_SCALAR = 2
    FLOAT32_VECTOR = 3


def _name(value: str | None, field: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value or "\0" in value:
        raise ValueError(f"{field} must be nonempty and contain no NUL")
    value.encode("utf-8")


def _integer(value: int, field: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{field} must be an integer, not bool or a coerced scalar")
    if not -(1 << 63) <= value < (1 << 63):
        raise ValueError(f"{field} is outside int64 range")


def _enum(value: IntEnum, kind: type[IntEnum], field: str) -> None:
    if not isinstance(value, kind):
        raise TypeError(f"{field} must be {kind.__name__}")


def _records(value: object, kind: type, field: str) -> tuple:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{field} must be a list or tuple")
    result = tuple(value)
    if any(not isinstance(item, kind) for item in result):
        raise TypeError(f"{field} contains an invalid {kind.__name__}")
    return result


def _names(value: tuple[str, ...], field: str) -> tuple[str, ...]:
    result = _records(value, str, field)
    for item in result:
        _name(item, field)
    return result


def _integers(value: tuple[int, ...], field: str) -> tuple[int, ...]:
    result = _records(value, int, field)
    for item in result:
        _integer(item, field)
    return result


def _position(value: int | None) -> None:
    if value is not None:
        _integer(value, "logits_position")
        if value < 0:
            raise ValueError("logits_position must be nonnegative")


@dataclass(frozen=True)
class ExitCondition:
    """Explicit exit; unused fields must retain their empty defaults."""

    kind: ExitKind
    limit: int = 0
    output_name: str | None = None
    eos_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _enum(self.kind, ExitKind, "kind")
        _integer(self.limit, "limit")
        _name(self.output_name, "output_name", optional=True)
        object.__setattr__(self, "eos_token_ids", _integers(self.eos_token_ids, "eos_token_ids"))
        if self.kind in (ExitKind.MAX_ITERATIONS, ExitKind.MAX_GENERATED_TOKENS):
            if self.limit < 0:
                raise ValueError("Exit limits must be nonnegative")
        elif self.limit != 0:
            raise ValueError("This exit kind does not use a limit")
        if self.kind != ExitKind.BOOLEAN_OUTPUT and self.output_name is not None:
            raise ValueError("Only a boolean exit names a model output")
        if self.kind != ExitKind.EOS and self.eos_token_ids:
            raise ValueError("Only an EOS exit supplies token IDs")


@dataclass(frozen=True)
class FeedbackBinding:
    """Bind the preceding model output to the next invocation's input."""

    input_name: str
    output_name: str

    def __post_init__(self) -> None:
        _name(self.input_name, "input_name")
        _name(self.output_name, "output_name")


@dataclass(frozen=True)
class ValueSource:
    """Explicit named chain source; component_index applies to COMPONENT_OUTPUT."""

    kind: SourceKind
    name: str
    component_index: int = 0

    def __post_init__(self) -> None:
        _enum(self.kind, SourceKind, "kind")
        _name(self.name, "name")
        _integer(self.component_index, "component_index")


@dataclass(frozen=True)
class InputBinding:
    """Bind a source to one model or declared policy input."""

    input_name: str
    source: ValueSource

    def __post_init__(self) -> None:
        _name(self.input_name, "input_name")
        if not isinstance(self.source, ValueSource):
            raise TypeError("source must be ValueSource")


@dataclass(frozen=True)
class StateBinding:
    """Retain a component output under an explicit state name."""

    state_name: str
    output_name: str

    def __post_init__(self) -> None:
        _name(self.state_name, "state_name")
        _name(self.output_name, "output_name")


@dataclass(frozen=True)
class ChainOutput:
    """Select a component output or generated history under a public name."""

    name: str
    component_index: int
    output_name: str | None = None
    kind: OutputKind = OutputKind.MODEL_OUTPUT

    def __post_init__(self) -> None:
        _name(self.name, "name")
        _integer(self.component_index, "component_index")
        _name(self.output_name, "output_name", optional=True)
        _enum(self.kind, OutputKind, "kind")


@dataclass(frozen=True)
class ProviderSelection:
    """CPU by default; non-CPU requests explicitly identify provider and device."""

    mode: ProviderMode = ProviderMode.CPU
    provider_name: str | None = None
    device_type: DeviceType = DeviceType.CPU
    device_id: int = 0

    def __post_init__(self) -> None:
        _enum(self.mode, ProviderMode, "mode")
        _name(self.provider_name, "provider_name", optional=True)
        _enum(self.device_type, DeviceType, "device_type")
        _integer(self.device_id, "device_id")


@dataclass(frozen=True)
class CacheSpec:
    """Contiguous native cache layout; the sequence dimension starts at zero."""

    name: str
    element_type: ElementType
    shape: tuple[int, ...]
    sequence_axis: int
    initial_capacity: int = 0

    def __post_init__(self) -> None:
        _name(self.name, "name")
        _enum(self.element_type, ElementType, "element_type")
        object.__setattr__(self, "shape", _integers(self.shape, "shape"))
        _integer(self.sequence_axis, "sequence_axis")
        _integer(self.initial_capacity, "initial_capacity")


@dataclass(frozen=True)
class CacheBinding:
    """Connect one cache to model ports using an explicit update policy."""

    cache: CacheSpec
    input_name: str
    output_name: str
    update: CacheUpdate = CacheUpdate.SHARE

    def __post_init__(self) -> None:
        if not isinstance(self.cache, CacheSpec):
            raise TypeError("cache must be CacheSpec")
        _name(self.input_name, "input_name")
        _name(self.output_name, "output_name")
        _enum(self.update, CacheUpdate, "update")


@dataclass(frozen=True)
class SamplerConfig:
    """Greedy logits selection; None chooses the final row of rank-three logits."""

    algorithm: SamplingAlgorithm = SamplingAlgorithm.GREEDY
    logits_position: int | None = None

    def __post_init__(self) -> None:
        _enum(self.algorithm, SamplingAlgorithm, "algorithm")
        _position(self.logits_position)


@dataclass(frozen=True)
class GenerationConfig:
    """Native prefill/decode policy; graph port names are explicit."""

    phase: GenerationPhase
    input_ids: str
    logits_output: str
    attention_mask: str | None = None
    position_ids: str | None = None
    logits_position: int | None = None

    def __post_init__(self) -> None:
        _enum(self.phase, GenerationPhase, "phase")
        _name(self.input_ids, "input_ids")
        _name(self.logits_output, "logits_output")
        _name(self.attention_mask, "attention_mask", optional=True)
        _name(self.position_ids, "position_ids", optional=True)
        _position(self.logits_position)


@dataclass(frozen=True)
class GenerationBootstrap:
    """Initial decode logits plus original prompt; prefill itself does not sample."""

    logits: ValueSource
    prompt: ValueSource
    logits_position: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.logits, ValueSource) or not isinstance(self.prompt, ValueSource):
            raise TypeError("logits and prompt must be ValueSource")
        _position(self.logits_position)


@dataclass(frozen=True)
class DdimConfig:
    """Fixed deterministic CPU FP32 DDIM policy, including for CUDA denoisers."""

    latent_shape: tuple[int, ...]
    num_inference_steps: int
    guidance_scale: float
    latent_input: str
    timestep_input: str
    conditioning_input: str
    prediction_output: str
    initialization_input: str
    positive_conditioning_input: str
    negative_conditioning_input: str
    latent_output: str
    guidance_mode: GuidanceMode = GuidanceMode.BATCHED
    timestep_format: TimestepFormat = TimestepFormat.FLOAT32_SCALAR
    profile: str = "sd15_ddim_v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "latent_shape", _integers(self.latent_shape, "latent_shape"))
        _integer(self.num_inference_steps, "num_inference_steps")
        if type(self.guidance_scale) not in (int, float):
            raise TypeError("guidance_scale must be float or int, not a coerced scalar")
        for field in (
            "latent_input",
            "timestep_input",
            "conditioning_input",
            "prediction_output",
            "initialization_input",
            "positive_conditioning_input",
            "negative_conditioning_input",
            "latent_output",
            "profile",
        ):
            _name(getattr(self, field), field)
        _enum(self.guidance_mode, GuidanceMode, "guidance_mode")
        _enum(self.timestep_format, TimestepFormat, "timestep_format")


@dataclass(frozen=True)
class LoopConfig:
    """Copied loop definition; native validation governs policies and explicit exits."""

    output_names: tuple[str, ...]
    exit_conditions: tuple[ExitCondition, ...]
    finite_input_names: tuple[str, ...] = ()
    feedback: tuple[FeedbackBinding, ...] = ()
    generated_token_output: str | None = None
    nonempty_input: str | None = None
    kv_caches: tuple[CacheBinding, ...] = ()
    generation: GenerationConfig | None = None
    ddim: DdimConfig | None = None
    cache_provider: ProviderSelection = ProviderSelection()
    sampler_provider: ProviderSelection = ProviderSelection()

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_names", _names(self.output_names, "output_names"))
        object.__setattr__(self, "finite_input_names", _names(self.finite_input_names, "finite_input_names"))
        object.__setattr__(self, "exit_conditions", _records(self.exit_conditions, ExitCondition, "exit_conditions"))
        object.__setattr__(self, "feedback", _records(self.feedback, FeedbackBinding, "feedback"))
        object.__setattr__(self, "kv_caches", _records(self.kv_caches, CacheBinding, "kv_caches"))
        _name(self.generated_token_output, "generated_token_output", optional=True)
        _name(self.nonempty_input, "nonempty_input", optional=True)
        for field, kind in (
            ("generation", GenerationConfig),
            ("ddim", DdimConfig),
            ("cache_provider", ProviderSelection),
            ("sampler_provider", ProviderSelection),
        ):
            value = getattr(self, field)
            if value is None and field in ("generation", "ddim"):
                continue
            if not isinstance(value, kind):
                raise TypeError(f"{field} must be {kind.__name__}")


@dataclass(frozen=True)
class ChainComponent:
    """Named initialized Session or SessionLoop plus explicit connections."""

    name: str
    execution: InferenceSession | SessionLoop
    inputs: tuple[InputBinding, ...] = ()
    state_outputs: tuple[StateBinding, ...] = ()
    kv_caches: tuple[CacheBinding, ...] = ()
    cache_provider: ProviderSelection | None = None
    sampler_provider: ProviderSelection | None = None
    generation_bootstrap: GenerationBootstrap | None = None

    def __post_init__(self) -> None:
        _name(self.name, "name")
        object.__setattr__(self, "inputs", _records(self.inputs, InputBinding, "inputs"))
        object.__setattr__(self, "state_outputs", _records(self.state_outputs, StateBinding, "state_outputs"))
        object.__setattr__(self, "kv_caches", _records(self.kv_caches, CacheBinding, "kv_caches"))
        for field, kind in (
            ("cache_provider", ProviderSelection),
            ("sampler_provider", ProviderSelection),
            ("generation_bootstrap", GenerationBootstrap),
        ):
            value = getattr(self, field)
            if value is not None and not isinstance(value, kind):
                raise TypeError(f"{field} must be {kind.__name__}")


@dataclass(frozen=True)
class ChainConfig:
    """Ordered immutable component definitions and named final output ports."""

    components: tuple[ChainComponent, ...]
    outputs: tuple[ChainOutput, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", _records(self.components, ChainComponent, "components"))
        object.__setattr__(self, "outputs", _records(self.outputs, ChainOutput, "outputs"))


_CPU_PROVIDER = ProviderSelection()
_DEFAULT_SAMPLER = SamplerConfig()


class ResultKind(IntEnum):
    """Native progress or retained-state snapshot kind."""

    LOOP = 0
    CHAIN = 1
    STATE_SNAPSHOT = 2


class DispatchOrigin(IntEnum):
    """Observed implementation route, not model-node assignment."""

    CPU = 0
    PROVIDER = 1
    CPU_FALLBACK = 2
    NOT_INVOKED = 3


@dataclass(frozen=True)
class StepInfo:
    """Copied immutable native progress; executed=False does not imply completion."""

    kind: ResultKind
    executed: bool
    completed: bool
    invocations: int
    iterations: int
    generated_tokens: int
    components_completed: int
    component_name: str
    exit_reasons: tuple[ExitKind, ...]
    generated_token: int | None


@dataclass(frozen=True)
class ValueInfo:
    """Copied tensor metadata; element_type is the native ONNX element identifier."""

    shape: tuple[int, ...]
    element_type: int
    device_type: DeviceType
    device_id: int


@dataclass(frozen=True)
class StagingInfo:
    """CPU payload and snapshot-copy counters, not measured DMA transfers."""

    cpu_input_request_bytes: int
    cpu_output_bytes: int
    cpu_snapshot_copy_bytes: int


@dataclass(frozen=True)
class DispatchInfo:
    """Copied operation diagnostics; names and counts have no borrowed references."""

    provider_name: str
    requested_provider_name: str
    policy: ProviderMode
    origin: DispatchOrigin
    operation: ProviderOperation
    device_type: DeviceType
    device_id: int
    cpu_calls: int
    provider_calls: int
    cpu_delegations: int
    failures: int


@dataclass(frozen=True)
class ProviderCapabilities:
    """Native version/operation/element/device bitmasks; no capability is inferred."""

    version: int
    operations: int
    cache_element_types: int
    sampler_element_types: int
    memory_types: int
    device_id: int


@dataclass(frozen=True)
class CacheStats:
    """Native cache lifetime allocation/copy/share counters."""

    allocations: int
    bytes_copied: int
    shared_updates: int


@dataclass(frozen=True)
class CacheViewInfo:
    """Retained read-only logical tensor and copied length/capacity."""

    value: StatefulValue
    length: int
    capacity: int


@dataclass(frozen=True)
class PackageVariantInfo:
    """Copied declared variant name and optional execution-provider name."""

    name: str
    execution_provider: str | None


@dataclass(frozen=True)
class PackageComponentInfo:
    """Copied component name and declared variants."""

    name: str
    variants: tuple[PackageVariantInfo, ...]


@dataclass(frozen=True)
class ModelPackageInfo:
    """Copied native schema major version and component/variant metadata."""

    schema_version_major: int
    components: tuple[PackageComponentInfo, ...]


@dataclass(frozen=True)
class SelectedComponentInfo:
    """Copied selected variant identity and resolved folder."""

    variant_name: str
    variant_folder: str


def _dispatch_info(info: dict) -> DispatchInfo:
    info["policy"] = ProviderMode(info["policy"])
    info["origin"] = DispatchOrigin(info["origin"])
    info["operation"] = ProviderOperation(info["operation"])
    info["device_type"] = DeviceType(info["device_type"])
    return DispatchInfo(**info)


def _backend():
    backend = getattr(_C, "_stateful", None)
    if backend is None or not backend.available:
        raise _C.NotImplemented("Stateful execution is unavailable in this ONNX Runtime build")
    return backend


class _Lease:
    def __init__(self, native, owners: tuple[object, ...]):
        self.native = native
        self.owners = owners

    def __del__(self) -> None:
        # Tuple/dict cleanup order must not release a Session before its native children.
        native = self.__dict__.get("native")
        owners = self.__dict__.get("owners", ())
        self.native = None
        self.owners = ()
        del native, owners


class _Resource:
    def __init__(self, native, owners: tuple[object, ...]):
        self._native = native
        self._owners = owners
        self._state_lock = Lock()
        self._busy = False

    def __del__(self) -> None:
        # Preserve dependency ownership during native destruction, including partial construction.
        native = self.__dict__.get("_native")
        owners = self.__dict__.get("_owners", ())
        self._native = None
        self._owners = ()
        del native, owners

    @contextmanager
    def _operation(self):
        with self._state_lock:
            if self._native is None:
                raise RuntimeError(f"{type(self).__name__} is closed")
            if self._busy:
                raise RuntimeError(f"{type(self).__name__} is in use")
            self._busy = True
            native, owners = self._native, self._owners
        try:
            yield native, owners
        finally:
            with self._state_lock:
                self._busy = False

    def close(self) -> None:
        """Release this handle; independently retained children remain valid."""
        with self._state_lock:
            if self._busy:
                raise RuntimeError(f"{type(self).__name__} is in use")
            native, owners = self._native, self._owners
            self._native = None
            self._owners = ()
        # Native Sessions/callbacks can destroy Python state; release outside the lock with the GIL.
        del native, owners

    def __enter__(self) -> Self:
        with self._operation():
            return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class _CopiedOrtValue(OrtValue):
    def __init__(self, native, owners: tuple[object, ...]):
        super().__init__(native)
        self._stateful_owners = owners

    def __del__(self) -> None:
        native = self.__dict__.pop("_ortvalue", None)
        owners = self.__dict__.pop("_stateful_owners", ())
        del native, owners


class StatefulValue(_Resource):
    """Read-only retained tensor; writable conversions always create new storage.

    Direct GPU numpy() is rejected. Request copy_to_cpu() for explicit readback.
    No raw pointer, writable buffer, array protocol or DLPack alias is exposed.
    """

    def get_info(self) -> ValueInfo:
        """Return copied shape/type/device metadata without copying tensor data."""
        with self._operation() as (native, _):
            info = native.metadata()
            return ValueInfo(
                tuple(info["shape"]), info["element_type"], DeviceType(info["device_type"]), info["device_id"]
            )

    def to_ortvalue(self) -> OrtValue:
        """Return an independent mutable copy on the original device."""
        with self._operation() as (native, owners):
            return _CopiedOrtValue(native.copy(False), (_Lease(native, owners),))

    def copy_to_cpu(self) -> OrtValue:
        """Explicitly copy to an independent mutable CPU OrtValue."""
        with self._operation() as (native, owners):
            return _CopiedOrtValue(native.copy(True), (_Lease(native, owners),))

    def numpy(self) -> np.ndarray:
        """Return an independent writable CPU array; device values require copy_to_cpu()."""
        with self._operation() as (native, owners):
            if native.metadata()["device_type"] != DeviceType.CPU:
                raise ValueError("Device values require an explicit copy_to_cpu() before numpy conversion")
            return _CopiedOrtValue(native.copy(False), (_Lease(native, owners),)).numpy()


class StepResult(_Resource):
    """Immutable retained result; copied metadata and returned value handles outlive it."""

    def __init__(self, native, owners: tuple[object, ...]):
        super().__init__(native, owners)
        fields = native.info()
        fields["kind"] = ResultKind(fields["kind"])
        fields["exit_reasons"] = tuple(ExitKind(reason) for reason in fields["exit_reasons"])
        self._info = StepInfo(**fields)

    def get_info(self) -> StepInfo:
        """Return immutable copied native progress."""
        with self._operation():
            return self._info

    def get_outputs(self, group: OutputGroup = OutputGroup.SELECTED_OUTPUTS) -> Mapping[str, StatefulValue]:
        """Return independently retained read-only values without copying tensor data."""
        _enum(group, OutputGroup, "group")
        with self._operation() as (native, owners):
            return MappingProxyType(
                {
                    name: StatefulValue(value, (_Lease(native, owners),))
                    for name, value in native.outputs(int(group)).items()
                }
            )


TensorInput = np.ndarray | OrtValue | StatefulValue


def _input(value: TensorInput) -> tuple[object, tuple[object, ...]]:
    if isinstance(value, StatefulValue):
        with value._operation() as (native, owners):
            return native, (_Lease(native, owners),)
    elif isinstance(value, np.ndarray):
        if not value.flags.c_contiguous or not value.dtype.isnative or value.dtype.kind not in "biuf":
            raise ValueError("Stateful NumPy inputs require contiguous, native-endian, numeric/bool storage")
        tensor = OrtValue.ortvalue_from_numpy(value)
        return tensor._get_c_value(), (tensor,)
    elif isinstance(value, OrtValue):
        return value._get_c_value(), (value,)
    raise TypeError("Stateful inputs must be NumPy ndarray, OrtValue or StatefulValue")


def _feeds(values: Mapping[str, TensorInput] | None) -> tuple[dict[str, object], tuple[object, ...]]:
    if values is None:
        return {}, ()
    if not isinstance(values, Mapping):
        raise TypeError("Named feeds must be a mapping")
    result: dict[str, object] = {}
    owners: list[object] = []
    for name, value in values.items():
        _name(name, "feed name")
        tensor, retained = _input(value)
        result[name] = tensor
        owners.extend(retained)
    return result, tuple(owners)


class StatefulExecution(_Resource):
    """Sequential execution context; only cancel() may overlap a native operation.

    Input buffers are retained, not copied. Do not mutate them while this context
    can use them, including Reset replay. Failed/possibly advanced work requires
    reset(). Closing a parent definition does not invalidate this context.
    """

    def __init__(self, native, owners: tuple[object, ...]):
        super().__init__(native, owners)
        self._delivery_failed = False

    def _advance(self, run: bool) -> StepResult:
        with self._operation() as (native, owners):
            if self._delivery_failed:
                raise _C.Fail("Stateful execution requires Reset after a failed result delivery")
            lease = _Lease(native, owners)
            self._delivery_failed = True
            result = StepResult(native.advance(run), (lease,))
            self._delivery_failed = False
            return result

    def step(self) -> StepResult:
        """Perform one native step, including a sample-only token or complete DDIM update."""
        return self._advance(False)

    def run(self) -> StepResult:
        """Run to native completion and return only the final result."""
        return self._advance(True)

    def reset(self) -> None:
        """Reset native state and reuse the originally retained feeds."""
        with self._operation() as (native, _):
            self._delivery_failed = True
            native.reset()
            self._delivery_failed = False

    def cancel(self) -> None:
        """Request native boundary cancellation; safe from another Python thread."""
        with self._state_lock:
            if self._native is None:
                raise RuntimeError("StatefulExecution is closed")
            native = self._native
            lease = _Lease(native, self._owners)
        native.cancel()
        del lease

    def requires_reset(self) -> bool:
        """Report native or binding delivery invalidation."""
        with self._operation() as (native, _):
            return self._delivery_failed or native.requires_reset()

    def get_state(self) -> StepResult:
        """Return an immutable native state snapshot; reject invalidated contexts."""
        with self._operation() as (native, owners):
            if self._delivery_failed:
                raise _C.Fail("Stateful execution requires Reset")
            return StepResult(native.state(), (_Lease(native, owners),))

    def get_staging_info(self, component: int = 0) -> StagingInfo:
        """Return CPU payload counters, not measured host/device DMA."""
        _integer(component, "component")
        if component < 0:
            raise ValueError("component must be nonnegative")
        with self._operation() as (native, _):
            return StagingInfo(**native.staging(component))

    def get_dispatch_info(
        self, feature: ProviderFeature, operation: ProviderOperation, component: int = 0
    ) -> DispatchInfo:
        """Return copied native operation diagnostics, including after failure."""
        _enum(feature, ProviderFeature, "feature")
        _enum(operation, ProviderOperation, "operation")
        _integer(component, "component")
        if component < 0:
            raise ValueError("component must be nonnegative")
        with self._operation() as (native, _):
            return _dispatch_info(native.dispatch(int(feature), int(operation), component))


class SessionLoop(_Resource):
    """Immutable native loop retaining the exact initialized Python Session.

    Later set_providers() on the original InferenceSession affects only newly
    constructed definitions. Executions retain original arrays/OrtValues.
    """

    def __init__(self, session: InferenceSession, config: LoopConfig):
        if not isinstance(session, InferenceSession) or not isinstance(config, LoopConfig):
            raise TypeError("SessionLoop requires InferenceSession and LoopConfig")
        original = session._sess
        if original is None:
            raise ValueError("SessionLoop requires an initialized Session")
        super().__init__(_backend().Loop.create(original, config), (original,))

    def create_execution(
        self,
        inputs: Mapping[str, TensorInput] | None = None,
        *,
        batches: Sequence[Mapping[str, TensorInput]] = (),
    ) -> StatefulExecution:
        """Create an independent context retaining feeds without binding-level copies."""
        values, owners = _feeds(inputs)
        if not isinstance(batches, (tuple, list)):
            raise TypeError("batches must be a list or tuple of named feeds")
        finite, retained = [], list(owners)
        for batch in batches:
            if batch is None:
                raise TypeError("A finite batch must be a named feed mapping")
            feeds, batch_owners = _feeds(batch)
            finite.append(feeds)
            retained.extend(batch_owners)
        with self._operation() as (native, dependencies):
            execution = native.create_execution(values, tuple(finite))
            return StatefulExecution(execution, (_Lease(native, dependencies), tuple(retained)))


class SessionChain(_Resource):
    """Immutable ordered native chain retaining exact Session/loop dependencies."""

    def __init__(self, config: ChainConfig):
        if not isinstance(config, ChainConfig):
            raise TypeError("SessionChain requires ChainConfig")
        executions, owners = [], []
        for component in config.components:
            execution = component.execution
            if isinstance(execution, SessionLoop):
                with execution._operation() as (native, dependencies):
                    executions.append(native)
                    owners.append(_Lease(native, dependencies))
            elif isinstance(execution, InferenceSession):
                original = execution._sess
                if original is None:
                    raise ValueError("Chain components require an initialized Session")
                executions.append(original)
                owners.append(original)
            else:
                raise TypeError("Chain components require initialized InferenceSession or SessionLoop")
        super().__init__(_backend().Chain.create(config, tuple(executions)), tuple(owners))

    def create_execution(
        self,
        inputs: Mapping[str, TensorInput] | None = None,
        *,
        sequences: Mapping[str, Sequence[TensorInput]] | None = None,
        initial_state: Mapping[str, TensorInput] | None = None,
    ) -> StatefulExecution:
        """Capture explicitly named feeds/sequences/state; Reset reuses them."""
        values, owners = _feeds(inputs)
        state, state_owners = _feeds(initial_state)
        if sequences is not None and not isinstance(sequences, Mapping):
            raise TypeError("sequences must be a mapping")
        finite, retained = {}, list(owners + state_owners)
        for name, sequence in () if sequences is None else sequences.items():
            _name(name, "sequence name")
            if not isinstance(sequence, (tuple, list)):
                raise TypeError("A finite sequence must be a list or tuple")
            tensors = []
            for value in sequence:
                tensor, dependencies = _input(value)
                tensors.append(tensor)
                retained.extend(dependencies)
            finite[name] = tuple(tensors)
        with self._operation() as (native, dependencies):
            execution = native.create_execution(values, finite, state)
            return StatefulExecution(execution, (_Lease(native, dependencies), tuple(retained)))


def _native_session(session: InferenceSession | None) -> tuple[object | None, tuple[object, ...]]:
    if session is None:
        return None, ()
    if not isinstance(session, InferenceSession):
        raise TypeError("session must be InferenceSession or None")
    original = session._sess
    if original is None:
        raise ValueError("An initialized Session is required")
    return original, (original,)


class KVCacheView(_Resource):
    """Immutable cache snapshot; returned values survive update/reset/manager close."""

    def get_info(self) -> CacheViewInfo:
        """Return a retained value and copied logical/allocation extents."""
        with self._operation() as (native, owners):
            length, capacity, value = native.info()
            return CacheViewInfo(StatefulValue(value, (_Lease(native, owners),)), length, capacity)


class KVCacheManager(_Resource):
    """Sequential native cache storage with owned allocator/Session dependencies.

    Update input owners may be retained until close, including across Reset's
    storage reuse. Share inputs must remain immutable. Copy/Append retain native
    copying semantics; returned views own their independent lifetime dependencies.
    """

    def __init__(
        self,
        specs: Sequence[CacheSpec],
        *,
        provider: ProviderSelection = _CPU_PROVIDER,
        session: InferenceSession | None = None,
    ):
        definitions = _records(specs, CacheSpec, "specs")
        if not isinstance(provider, ProviderSelection):
            raise TypeError("provider must be ProviderSelection")
        original, owners = _native_session(session)
        super().__init__(_backend().Cache.create(definitions, original, provider), owners)

    def get_view(self, name: str) -> KVCacheView:
        """Return an immutable logical view without copying data."""
        _name(name, "name")
        with self._operation() as (native, owners):
            return KVCacheView(native.view(name), owners)

    def update(self, name: str, value: TensorInput, kind: CacheUpdate = CacheUpdate.SHARE) -> None:
        """Apply native Share/Copy/Append; Share requires allocator-owned, not borrowed NumPy storage."""
        _name(name, "name")
        _enum(kind, CacheUpdate, "kind")
        tensor, retained = _input(value)
        with self._operation() as (native, owners):
            # A failed advertised callback can retain storage; pin owners before advancement.
            self._owners = (*owners, *retained)
            native.update(name, tensor, int(kind))

    def reserve(self, name: str, capacity: int) -> None:
        """Reserve native storage capacity; prior views remain immutable."""
        _name(name, "name")
        _integer(capacity, "capacity")
        with self._operation() as (native, _):
            native.reserve(name, capacity)

    def reset(self) -> None:
        """Reset logical lengths using native capacity/storage-reuse rules."""
        with self._operation() as (native, _):
            native.reset()

    def requires_reset(self) -> bool:
        """Report native cache invalidation after a failed operation."""
        with self._operation() as (native, _):
            return native.requires_reset()

    def get_stats(self) -> CacheStats:
        """Return copied native allocation/copy/share counts."""
        with self._operation() as (native, _):
            return CacheStats(**native.stats())

    def get_dispatch_info(self, operation: ProviderOperation) -> DispatchInfo:
        """Return copied native cache dispatch diagnostics."""
        _enum(operation, ProviderOperation, "operation")
        with self._operation() as (native, _):
            return _dispatch_info(native.dispatch(int(operation)))


class Sampler(_Resource):
    """Native greedy sampler with explicit independent provider/device selection."""

    def __init__(
        self,
        config: SamplerConfig = _DEFAULT_SAMPLER,
        *,
        provider: ProviderSelection = _CPU_PROVIDER,
        session: InferenceSession | None = None,
    ):
        if not isinstance(config, SamplerConfig) or not isinstance(provider, ProviderSelection):
            raise TypeError("Sampler requires SamplerConfig and ProviderSelection")
        original, owners = _native_session(session)
        super().__init__(_backend().Sampler.create(config, original, provider), owners)

    def sample(self, logits: TensorInput) -> int:
        """Sample once using native dtype/NaN/tie/row semantics; failures are not retried."""
        tensor, retained = _input(logits)
        with self._operation() as (native, _):
            token = native.sample(tensor)
            del retained
            return token

    def get_dispatch_info(self) -> DispatchInfo:
        """Return copied sampler invocation/failure diagnostics."""
        with self._operation() as (native, _):
            return _dispatch_info(native.dispatch())


def query_provider_capabilities(session: InferenceSession, provider_name: str) -> ProviderCapabilities:
    """Query actual registered native capabilities without invoking operations."""
    if not isinstance(session, InferenceSession):
        raise TypeError("An InferenceSession is required")
    _name(provider_name, "provider_name")
    original, owners = _native_session(session)
    result = ProviderCapabilities(**_backend().capabilities(original, provider_name))
    del owners
    return result


def _session_options(options):
    if options is not None and not isinstance(options, _C.SessionOptions):
        raise TypeError("session_options must be SessionOptions or None")
    return options


class ModelPackageComponent(_Resource):
    """Read-only selected component; created Sessions retain native selection dependencies."""

    def get_info(self) -> SelectedComponentInfo:
        """Return copied selected variant name and resolved folder."""
        with self._operation() as (native, _):
            return SelectedComponentInfo(**native.info())

    def create_session(self, session_options: _C.SessionOptions | None = None) -> InferenceSession:
        """Adopt one initialized native Session without loading/initializing it again.

        Native package option precedence is preserved. Automatic Python fallback
        is disabled; explicit set_providers() reloads the same selected model file,
        not another package variant.
        """
        _session_options(session_options)
        with self._operation() as (native, _):
            session, path = native.session(session_options)
            return InferenceSession._from_native_session(session, path)


class ModelPackage(_Resource):
    """Read-only native package inspection and explicit component/chain loading."""

    def __init__(self, path: str | PathLike[str]):
        value = fspath(path)
        _name(value, "path")
        super().__init__(_backend().Package.open(value), ())

    def get_info(self) -> ModelPackageInfo:
        """Return copied schema-major/component/variant metadata, not borrowed views."""
        with self._operation() as (native, _):
            info = native.info()
            return ModelPackageInfo(
                info["schema_version_major"],
                tuple(
                    PackageComponentInfo(
                        component["name"],
                        tuple(PackageVariantInfo(**variant) for variant in component["variants"]),
                    )
                    for component in info["components"]
                ),
            )

    def resolve_string_ref(self, reference: str, *, base_dir: str = "", must_exist: bool = True) -> str:
        """Resolve through native package rules; no Python path reinterpretation is applied."""
        _name(reference, "reference")
        if not isinstance(base_dir, str) or "\0" in base_dir:
            raise TypeError("base_dir must be a string without NUL")
        if type(must_exist) is not bool:
            raise TypeError("must_exist must be bool")
        with self._operation() as (native, _):
            return native.resolve(reference, base_dir, must_exist)

    def select_component(self, name: str, session_options: _C.SessionOptions | None = None) -> ModelPackageComponent:
        """Select using a captured native options/provider snapshot."""
        _name(name, "name")
        _session_options(session_options)
        with self._operation() as (native, owners):
            return ModelPackageComponent(native.select(name, session_options), (_Lease(native, owners),))

    def create_chain(
        self,
        name: str,
        session_options: _C.SessionOptions | None = None,
        *,
        step_session_options: Mapping[str, _C.SessionOptions] | None = None,
    ) -> SessionChain:
        """Load an explicitly named owned chain; per-step options are complete replacements."""
        _name(name, "name")
        _session_options(session_options)
        if step_session_options is not None and not isinstance(step_session_options, Mapping):
            raise TypeError("step_session_options must be a mapping")
        steps = {}
        for step, options in () if step_session_options is None else step_session_options.items():
            _name(step, "step name")
            if not isinstance(options, _C.SessionOptions):
                raise TypeError("Each step_session_options value must be SessionOptions")
            steps[step] = options
        with self._operation() as (native, owners):
            chain = SessionChain.__new__(SessionChain)
            _Resource.__init__(chain, native.chain(name, session_options, steps), (_Lease(native, owners),))
            return chain


__all__ = [
    "CacheBinding",
    "CacheSpec",
    "CacheStats",
    "CacheUpdate",
    "CacheViewInfo",
    "ChainComponent",
    "ChainConfig",
    "ChainOutput",
    "DdimConfig",
    "DeviceType",
    "DispatchInfo",
    "DispatchOrigin",
    "ElementType",
    "ExitCondition",
    "ExitKind",
    "FeedbackBinding",
    "GenerationBootstrap",
    "GenerationConfig",
    "GenerationPhase",
    "GuidanceMode",
    "InputBinding",
    "KVCacheManager",
    "KVCacheView",
    "LoopConfig",
    "ModelPackage",
    "ModelPackageComponent",
    "ModelPackageInfo",
    "OutputGroup",
    "OutputKind",
    "PackageComponentInfo",
    "PackageVariantInfo",
    "ProviderCapabilities",
    "ProviderFeature",
    "ProviderMode",
    "ProviderOperation",
    "ProviderSelection",
    "ResultKind",
    "Sampler",
    "SamplerConfig",
    "SamplingAlgorithm",
    "SelectedComponentInfo",
    "SessionChain",
    "SessionLoop",
    "SourceKind",
    "StagingInfo",
    "StateBinding",
    "StatefulExecution",
    "StatefulValue",
    "StepInfo",
    "StepResult",
    "TensorInput",
    "TimestepFormat",
    "ValueInfo",
    "ValueSource",
    "query_provider_capabilities",
]
