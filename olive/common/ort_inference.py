# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Import onnxruntime lazily since it is not a required dependency for Olive.
# Import in TYPE_CHECKING block for type hinting is fine.
import collections
import logging
import platform
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from onnxruntime import InferenceSession, IOBinding

    from olive.hardware.accelerator import Device

logger = logging.getLogger(__name__)


class OrtSessionFallbackError(Exception):
    """Raised when the onnxruntime fallback happens."""


def ort_supports_ep_devices() -> bool:
    import onnxruntime as ort

    # ep registration and device discovery are not well defined on Linux
    # checking for api availability instead of ort version since Windows ML has this API in 1.22 and ORT in 1.23
    return platform.system() == "Windows" and hasattr(ort, "get_ep_devices")


def maybe_register_ep_libraries(ep_paths: dict[str, str]):
    """Register execution provider libraries if onnxruntime supports it."""
    try:
        import onnxruntime as ort
    except ImportError:
        logger.debug("Skipping EP registration since onnxruntime is not installed")
        return

    if not ort_supports_ep_devices():
        return

    # providers that ort was built with such as CUDA, QNN, VitisAI but need registration
    for provider in set(ort.get_available_providers()):
        if ep_paths.get(provider) is None:
            builtin_library_name = f"onnxruntime_providers_{provider.replace('ExecutionProvider', '').lower()}.dll"
            if (Path(ort.__file__).parent / "capi" / builtin_library_name).exists():
                ep_paths[provider] = builtin_library_name

    for ep_name, ep_path in ep_paths.items():
        if ep_path is None:
            continue

        if ep_name == "OpenVINOExecutionProvider":
            # importing openvino may be required for putting openvino.dll on path/loading it
            try:
                import openvino as _  # noqa: F401
            except ImportError:
                logger.info(
                    "Failed to import openvino. May see DLL not found error when registering OpenVINOExecutionProvider."
                )

        try:
            logger.debug("Registering EP %s with path %s", ep_name, ep_path)
            ort.register_execution_provider_library(ep_name, ep_path)
        except Exception as e:
            if "already registered" in str(e):
                logger.debug("Execution provider %s is already registered, skipping registration.", ep_name)
            else:
                raise


def get_ort_available_providers():
    """Get the available providers for ONNXRuntime."""
    import onnxruntime as ort

    if not ort_supports_ep_devices():
        # what ORT was built with. Session will be created directly with InferenceSession(model_path, providers=["ProviderName"])
        return ort.get_available_providers()

    # only return registered EPs since session with be created with SessionOptions.add_provider_for_devices()
    # this is ordered by priority
    all_providers = ort.get_all_providers()
    available_provider_set = {ep_device.ep_name for ep_device in ort.get_ep_devices()}
    return [ep_name for ep_name in all_providers if ep_name in available_provider_set]


def get_ort_hardware_device_type(device: Union["Device", str]):
    from onnxruntime import OrtHardwareDeviceType

    mapping = {
        "cpu": OrtHardwareDeviceType.CPU,
        "gpu": OrtHardwareDeviceType.GPU,
        "npu": OrtHardwareDeviceType.NPU,
    }
    return mapping.get(device.lower())


def get_ort_execution_provider_device_policy(policy: str):
    from onnxruntime import OrtExecutionProviderDevicePolicy

    mapping = {
        "default": OrtExecutionProviderDevicePolicy.DEFAULT,
        "prefer_cpu": OrtExecutionProviderDevicePolicy.PREFER_CPU,
        "prefer_npu": OrtExecutionProviderDevicePolicy.PREFER_NPU,
        "prefer_gpu": OrtExecutionProviderDevicePolicy.PREFER_GPU,
        "max_performance": OrtExecutionProviderDevicePolicy.MAX_PERFORMANCE,
        "max_efficiency": OrtExecutionProviderDevicePolicy.MAX_EFFICIENCY,
        "overall_power": OrtExecutionProviderDevicePolicy.MIN_OVERALL_POWER,
    }
    return mapping.get(policy.lower())


def initialize_inference_session_options(
    sess_options, device, providers, provider_options, provider_selection_policy=None
):
    import onnxruntime as ort

    providers = providers or []
    provider_options = provider_options or []
    provider_options_by_ep = dict(zip(providers, provider_options))
    ort_device_type = get_ort_hardware_device_type(device)

    # ort.get_ep_devices may return ep_devices with the same ep_name and device, for example when connecting remotely or when there are multiple graph cards.
    # However, in onnxruntime, each EP name can only be added once. See: https://github.com/microsoft/onnxruntime/blob/fb0f6c652be5db0a3182c424a995efecf792d41c/onnxruntime/core/framework/execution_providers.h#L75
    added_ep_names = set()
    for ep_device in ort.get_ep_devices():
        if (
            ep_device.device.type == ort_device_type
            and ep_device.ep_name in provider_options_by_ep
            and ep_device.ep_name not in added_ep_names
        ):
            added_ep_names.add(ep_device.ep_name)
            sess_options.add_provider_for_devices([ep_device], provider_options_by_ep.get(ep_device.ep_name) or {})

    if provider_selection_policy:
        provider_selection_policy = get_ort_execution_provider_device_policy(provider_selection_policy)
        sess_options.set_provider_selection_policy(provider_selection_policy)


# NOTE: `device_id` is only used internally for inference with Distributed ONNX models.
# For regular ONNX models, the recommended way to specify the device is to set the environment variable
# `CUDA_VISIBLE_DEVICES` before running a workflow.
def get_ort_inference_session(
    model_path: Union[Path, str],
    device: Union["Device", str],
    inference_settings: dict[str, Any],
    use_ort_extensions: bool = False,
    device_id: Optional[int] = None,
    external_initializers: Optional[dict[str, "NDArray"]] = None,
):
    """Get an ONNXRuntime inference session.

    :param model_path: Path to the ONNX model file.
    :param inference_settings: Inference settings for the session.
        session_options: dict, optional. Session options for the session.
        execution_provider: list. List of execution providers to use. Can be a list of provider names or a list of
            (provider name, provider options) tuples.
        provider_options: list, optional. List of provider options for the execution providers.
    :param use_ort_extensions: Whether to use onnxruntime-extensions. Default is False.
    :param device_id: Optional device id to use for CUDA or DML execution providers.
    :param external_initializers: Optional external initializers for the session. A dictionary of external initializer
        names and numpy arrays.
    """
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    if use_ort_extensions:
        # register custom ops for onnxruntime_extensions
        from onnxruntime_extensions import get_library_path

        sess_options.register_custom_ops_library(get_library_path())
    if external_initializers:
        from onnxruntime import OrtValue

        # convert external initializers to OrtValue
        initializer_names = []
        initializer_values = []
        for name, value in external_initializers.items():
            initializer_names.append(name)
            initializer_values.append(OrtValue.ortvalue_from_numpy(value))

        # add external initializers to the session
        sess_options.add_external_initializers(initializer_names, initializer_values)

    logger.debug("inference_settings: %s", inference_settings)

    # session options
    session_options = inference_settings.get("session_options", {})
    inter_op_num_threads = session_options.get("inter_op_num_threads")
    intra_op_num_threads = session_options.get("intra_op_num_threads")
    enable_profiling = session_options.get("enable_profiling", False)
    execution_mode = session_options.get("execution_mode")
    graph_optimization_level = session_options.get("graph_optimization_level")
    extra_session_config = session_options.get("extra_session_config")
    log_severity_level = session_options.get("log_severity_level")
    if enable_profiling:
        sess_options.enable_profiling = True
    if inter_op_num_threads:
        sess_options.inter_op_num_threads = inter_op_num_threads
    if intra_op_num_threads:
        sess_options.intra_op_num_threads = intra_op_num_threads
    if execution_mode is not None:
        if execution_mode == 0:
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        elif execution_mode == 1:
            sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    if graph_optimization_level is not None:
        # level can be 0, 1, 2, 3
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel(graph_optimization_level)
    if extra_session_config:
        for key, value in extra_session_config.items():
            sess_options.add_session_config_entry(key, value)
    if log_severity_level is not None:
        sess_options.log_severity_level = log_severity_level

    # execution providers and provider options
    providers, provider_options = check_and_normalize_provider_args(
        inference_settings.get("execution_provider"),
        inference_settings.get("provider_options"),
        get_ort_available_providers(),
    )
    for idx, provider in enumerate(providers):
        if provider in ["CUDAExecutionProvider", "DmlExecutionProvider"] and device_id is not None:
            provider_options[idx]["device_id"] = str(device_id)
        elif (
            provider == "QNNExecutionProvider"
            and "backend_path" not in provider_options[idx]
            and not ort_supports_ep_devices
        ):
            # add backend_path for QNNExecutionProvider
            # not required after 1.22.
            # Causes backend load failure for Windows ML where this dll is in a different location than the ort dlls
            provider_options[idx]["backend_path"] = "QnnHtp.dll"
    logger.debug("Normalized providers: %s, provider_options: %s", providers, provider_options)

    # dml specific settings
    if len(providers) >= 1 and providers[0] == "DmlExecutionProvider":
        sess_options.enable_mem_pattern = False

    sess_kwargs = {}
    if ort_supports_ep_devices():
        initialize_inference_session_options(
            sess_options, device, providers, provider_options, inference_settings.get("provider_selection_policy")
        )
    else:
        sess_kwargs.update({"providers": providers, "provider_options": provider_options})

    # create session
    session = ort.InferenceSession(str(model_path), sess_options=sess_options, **sess_kwargs)
    check_ort_fallback(session, providers)

    # set tuning results for tunable operators (currently only for ROCM EP)
    tuning_op_result = inference_settings.get("tuning_op_result")
    if tuning_op_result:
        assert isinstance(tuning_op_result, list)
        session.set_tuning_results(tuning_op_result)
    return session


def check_and_normalize_provider_args(
    providers: Sequence[Union[str, tuple[str, dict[Any, Any]]]],
    provider_options: Sequence[dict[Any, Any]],
    available_provider_names: Sequence[str],
):
    """Validate the 'providers' and 'provider_options' arguments and returns a normalized version.

    :param providers: Optional sequence of providers in order of decreasing
        precedence. Values can either be provider names or tuples of
        (provider name, options dict).
    :param provider_options: Optional sequence of options dicts corresponding
        to the providers listed in 'providers'.
    :param available_provider_names: The available provider names.

    :return: Tuple of (normalized 'providers' sequence, normalized
        'provider_options' sequence).

    'providers' can contain either names or names and options. When any options
        are given in 'providers', 'provider_options' should not be used.

    The normalized result is a tuple of:
    1. Sequence of provider names in the same order as 'providers'.
    2. Sequence of corresponding provider options dicts with string keys and
        values. Unspecified provider options yield empty dicts.
    """
    # This function is copied from the following file.
    #    https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/onnxruntime_inference_collection.py
    if providers is None:
        return [], []

    provider_name_to_options = collections.OrderedDict()

    def set_provider_options(name, options):
        if available_provider_names and name not in available_provider_names:
            logger.warning(
                "Specified provider '%s' is not in available provider names.Available providers: '%s'",
                name,
                ", ".join(available_provider_names),
            )

        if name in provider_name_to_options:
            logger.warning("Duplicate provider '%s' encountered, ignoring.", name)
            return

        normalized_options = {str(key): str(value) for key, value in options.items()}
        provider_name_to_options[name] = normalized_options

    if not isinstance(providers, collections.abc.Sequence):
        raise ValueError("'providers' should be a sequence.")

    if provider_options is not None:
        if not isinstance(provider_options, collections.abc.Sequence):
            raise ValueError("'provider_options' should be a sequence.")

        if len(providers) != len(provider_options):
            raise ValueError("'providers' and 'provider_options' should be the same length if both are given.")

        if not all(isinstance(provider, str) for provider in providers):
            raise ValueError("Only string values for 'providers' are supported if 'provider_options' is given.")

        if not all(isinstance(options_for_provider, dict) for options_for_provider in provider_options):
            raise ValueError("'provider_options' values must be dicts.")

        for name, options in zip(providers, provider_options):
            set_provider_options(name, options)
    else:
        for provider in providers:
            if isinstance(provider, str):
                set_provider_options(provider, {})
            elif (
                isinstance(provider, (tuple, list))
                and len(provider) == 2
                and isinstance(provider[0], str)
                and isinstance(provider[1], dict)
            ):
                set_provider_options(provider[0], provider[1])
            else:
                raise ValueError("'providers' values must be either strings or (string, dict) tuples.")

    return list(provider_name_to_options.keys()), list(provider_name_to_options.values())


def check_ort_fallback(session: "InferenceSession", execution_providers: Sequence[str]):
    """Check if the onnxruntime fallback happens and raise an error if it does."""
    # pylint: disable=protected-access
    session_providers = session.get_providers()
    for ep in execution_providers:
        if ep not in session_providers:
            raise OrtSessionFallbackError(
                f"The onnxruntime fallback happens. {ep} is not in the session providers {session_providers}."
                f" session._enable_fallback = {session._enable_fallback}"
            )
    session.disable_fallback()


class OrtInferenceSession:
    """ORT Inference Session with IO binding."""

    def __init__(
        self,
        session: "InferenceSession",
        io_bind: bool = False,
        device: str = "cpu",
        shared_kv_buffer: bool = False,
        use_fp16: bool = False,
        input_feed: Optional[dict[str, "NDArray"]] = None,
        constant_inputs: Optional[dict[str, "NDArray"]] = None,
    ):
        """Initialize self.

        :param session: ONNXRuntime InferenceSession
        :param io_bind: Whether to use IO binding. Default is False.
        :param device: Device to run inference on. Default is "cpu".
        :param shared_kv_buffer: Whether to share the key/value buffer across multiple runs.
            Default is False. Only valid if io_bind is True.
        :param use_fp16: Whether to use fp16. Default is False. Both shared_kv_buffer and use_fp16 must be True
            at the same time to use shared key/value buffer.
        :param input_feed: Optional input feed for the session. Required when shared_kv_buffer and use_fp16 are True.
        :param constant_inputs: Optional constant inputs for the session. These will be passed to the session every
            inference run.
        """
        # TODO(anyone): use_fp16 is redundant with shared_kv_buffer. Remove it.
        self.session = session
        self.io_bind = io_bind
        self.device = device
        self.shared_kv_buffer = shared_kv_buffer
        self.use_fp16 = use_fp16
        self.kv_cache_ortvalues = {} if (self.shared_kv_buffer and self.use_fp16) else None
        # TODO(jambayk): investigate if io binding can be run without having to bind constant
        # inputs every time.
        self.constant_inputs = constant_inputs or {}

        self.io_binding = None
        if self.io_bind:
            self.io_binding = self.session.io_binding()
            if self.shared_kv_buffer and self.use_fp16:
                assert input_feed is not None, "input_feed is required when shared_kv_buffer and use_fp16 are True"
                bind_input_data(
                    self.io_binding,
                    {**input_feed, **self.constant_inputs},
                    self.use_fp16,
                    self.device,
                    shared_kv_buffer=self.shared_kv_buffer,
                    kv_cache_ortvalues=self.kv_cache_ortvalues,
                )
            bind_output_data(
                self.io_binding,
                self.session.get_outputs(),
                self.use_fp16,
                self.device,
                shared_kv_buffer=self.shared_kv_buffer,
                kv_cache_ortvalues=self.kv_cache_ortvalues,
            )

    def get_full_input_feed(self, input_feed: dict[str, "NDArray"]) -> dict[str, "NDArray"]:
        """Get the full input feed including constant inputs."""
        return {**input_feed, **self.constant_inputs}

    def run(self, output_names, input_feed: dict[str, "NDArray"], run_options=None) -> Sequence["NDArray"]:
        """Run inference with the given input data."""
        input_feed = self.get_full_input_feed(input_feed)
        if self.io_bind and self.device == "gpu":
            bind_input_data(
                self.io_binding,
                input_feed,
                self.use_fp16,
                self.device,
                shared_kv_buffer=self.shared_kv_buffer,
                kv_cache_ortvalues=self.kv_cache_ortvalues,
            )
            self.io_binding.synchronize_inputs()
            self.session.run_with_iobinding(self.io_binding)
            self.io_binding.synchronize_outputs()
            res = [i.numpy() for i in self.io_binding.get_outputs()]
            self.io_binding.clear_binding_inputs()
        else:
            res = self.session.run(None, input_feed)
        return res

    def time_run(
        self, input_feed: dict[str, "NDArray"], num_runs: int, num_warmup: int = 0, sleep_time: int = 0
    ) -> Sequence[float]:
        """Time inference runs with the given input data."""
        input_feed = self.get_full_input_feed(input_feed)
        latencies = []
        if self.io_bind:
            bind_input_data(
                self.io_binding,
                input_feed,
                self.use_fp16,
                self.device,
                shared_kv_buffer=self.shared_kv_buffer,
                kv_cache_ortvalues=self.kv_cache_ortvalues,
            )

        for _ in range(num_warmup + num_runs):
            if self.io_bind:
                self.io_binding.synchronize_inputs()
                t = time.perf_counter()
                self.session.run_with_iobinding(self.io_binding)
                self.io_binding.synchronize_outputs()
                latencies.append(time.perf_counter() - t)
            else:
                t = time.perf_counter()
                self.session.run(None, input_feed)
                latencies.append(time.perf_counter() - t)
            time.sleep(sleep_time)

        if self.io_bind:
            self.io_binding.clear_binding_inputs()
        return latencies[num_warmup:]


def bind_input_data(
    io_bind_op: "IOBinding",
    input_data: dict[str, "NDArray"],
    use_fp16: bool,
    device: str,
    device_id: int = 0,
    shared_kv_buffer: bool = False,
    kv_cache_ortvalues: dict = None,
):
    from onnxruntime import OrtValue

    io_bind_device = "cuda" if device == "gpu" else "cpu"

    for k, v in input_data.items():
        # "cache": from microsoft llama model" https://github.com/microsoft/Llama-2-Onnx#before-you-start
        # "past_key_values": from huggingface llama2 https://huggingface.co/meta-llama/Llama-2-13b-hf
        if shared_kv_buffer and use_fp16 and ("cache" in k or "past_key_values" in k):
            if k not in kv_cache_ortvalues:
                kv_cache_ortvalues[k] = OrtValue.ortvalue_from_numpy(v, io_bind_device, device_id)
            else:
                kv_cache_ortvalues[k].update_inplace(v)
            ort_v = kv_cache_ortvalues[k]
        else:
            ort_v = OrtValue.ortvalue_from_numpy(v, io_bind_device, device_id)
        io_bind_op.bind_ortvalue_input(k, ort_v)


def bind_output_data(
    io_bind_op: "IOBinding",
    output_data,
    use_fp16: bool,
    device: str,
    shared_kv_buffer: bool = False,
    kv_cache_ortvalues: dict = None,
):
    io_bind_device = "cuda" if device == "gpu" else "cpu"

    for item in output_data:
        name = item.name
        # "out": from microsoft llama model" https://github.com/microsoft/Llama-2-Onnx#before-you-start
        # "present": from huggingface llama2 https://huggingface.co/meta-llama/Llama-2-13b-hf
        if shared_kv_buffer and use_fp16 and ("out" in name or "present" in name):
            # Bind present KV cache outputs to past KV cache inputs in order to use shared buffer
            output_name = name.replace("out", "cache").replace("present", "past_key_values")
            io_bind_op.bind_ortvalue_output(name, kv_cache_ortvalues[output_name])
        else:
            io_bind_op.bind_output(name, io_bind_device)


def prepare_io_bindings(
    session: "InferenceSession",
    input_data: dict[str, "NDArray"],
    device: str,
    device_id: int = 0,
    shared_kv_buffer: bool = False,
    kv_cache_ortvalues: dict = None,
) -> "IOBinding":
    """Convert input from numpy array to OrtValue.

    session: ONNXRuntime session
    input_data: dict of input data, value is numpy array
    device: olive device
    device_id: 0 by default. TODO(trajep): support user to specified device id
    shared_kv_buffer: whether to share the key/value buffer across multiple runs, it is False by default,
        and only used when we observe kv cache and fp16 is used.
        TODO(trajep): how shared_kv_buffer works with generation task
    kv_cache_ortvalues: dict of OrtValue for shared kv cache, it is None by default.
    """
    use_fp16 = any(v.dtype == np.float16 for v in input_data.values())
    io_bind_op = session.io_binding()

    if shared_kv_buffer:
        kv_cache_ortvalues = kv_cache_ortvalues or {}

    bind_input_data(io_bind_op, input_data, use_fp16, device, device_id, shared_kv_buffer, kv_cache_ortvalues)
    bind_output_data(io_bind_op, session.get_outputs(), use_fp16, device, shared_kv_buffer, kv_cache_ortvalues)
    return io_bind_op
