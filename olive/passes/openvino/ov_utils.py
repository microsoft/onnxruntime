# -------------------------------------------------------------------------
# Copyright (c) Intel Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import numbers
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any, Optional, Union

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

from olive.common.utils import StrEnumBase
from olive.passes.pass_config import BasePassConfig

logger = logging.getLogger(__name__)


class IgnoreScopeTypeEnum(StrEnumBase):
    NAMES = "names"
    TYPES = "types"
    PATTERNS = "patterns"


class OVOptimumLibrary(StrEnumBase):
    TRANSFORMERS = "transformers"
    DIFFUSERS = "diffusers"
    TIMM = "timm"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPEN_CLIP = "open_clip"


def _validate_enum_value(value, enum_class: type, param_name: str) -> tuple[bool, str]:
    """Validate that a value can be converted to an enum (case-insensitive).

    Args:
        value: The value to validate (None, string, or already enum).
        enum_class: The enum class to validate against.
        param_name: Name of the parameter for error messages.

    Returns:
        Tuple of (is_valid, error_message). error_message is empty if valid.

    """
    if value is None or isinstance(value, enum_class):
        return True, ""

    if not isinstance(value, str):
        return False, f"{param_name} '{value}' is not a valid string or {enum_class.__name__} enum."

    lookup_key = value.lower()

    # Try matching by enum.value first (case-insensitive)
    value_map = {m.value.lower(): m for m in enum_class}
    if lookup_key in value_map:
        return True, ""

    # Try matching by enum.name (case-insensitive)
    name_map = {m.name.lower(): m for m in enum_class}
    if lookup_key in name_map:
        return True, ""

    # Validation failed
    valid_values = sorted({m.value for m in enum_class} | {m.name for m in enum_class})
    return False, f"{param_name} '{value}' is not supported. Supported values are: {', '.join(valid_values)}."


def _convert_to_enum(value, enum_class: type, param_name: str):
    """Convert a value to an enum if needed (case-insensitive).

    Accepts:
    - None (returns None)
    - Enum instances of the correct type (returns as-is)
    - Strings matching enum.value (case-insensitive)
    - Strings matching enum.name (case-insensitive)

    Args:
        value: The value to convert (None, string, or already enum).
        enum_class: The enum class to convert to.
        param_name: Name of the parameter for error messages.

    Returns:
        The enum value, or None if input was None.

    Raises:
        ValueError: If conversion fails.

    """
    if value is None or isinstance(value, enum_class):
        return value

    if not isinstance(value, str):
        raise ValueError(f"{param_name} '{value}' is not a valid string or {enum_class.__name__} enum.")

    lookup_key = value.lower()

    # Try matching by enum.value first (case-insensitive)
    value_map = {m.value.lower(): m for m in enum_class}
    if lookup_key in value_map:
        return value_map[lookup_key]

    # Try matching by enum.name (case-insensitive)
    name_map = {m.name.lower(): m for m in enum_class}
    if lookup_key in name_map:
        return name_map[lookup_key]

    # Conversion failed
    valid_values = sorted({m.value for m in enum_class} | {m.name for m in enum_class})
    raise ValueError(f"{param_name} '{value}' is not supported. Supported values are: {', '.join(valid_values)}.")


def infer_library_name(
    model_name_or_path: str,
    subfolder: str = "",
    revision: Optional[str] = None,
    cache_dir: str = HUGGINGFACE_HUB_CACHE,
    token: Optional[Union[bool, str]] = None,
) -> str:
    """Infer the Optimum-Intel library name for a given model.

    Falls back to ``"transformers"`` when ``sentence_transformers`` is detected

    Args:
        model_name_or_path: The model identifier or path. str
        subfolder: The subfolder within the model repository. optional. str. default is "".
        revision: The specific model version to use. optional. str. default is None (latest version).
        cache_dir: The directory to use for caching. optional. str. default is HUGGINGFACE_HUB_CACHE.
        token: The huggingface token to use. optional. bool or str. default is None.

    Returns:
        The inferred library name. str

    Raises:
        ImportError: If the optimum.intel library cannot be imported.

    """
    try:
        from optimum.intel.utils.modeling_utils import _infer_library_from_model_name_or_path
    except ImportError as e:
        raise ImportError("Please install Intel® optimum[openvino] to use OpenVINO Optimum Conversion") from e
    library_name = _infer_library_from_model_name_or_path(
        model_name_or_path=model_name_or_path,
        subfolder=subfolder,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
    )
    if library_name == "sentence_transformers":
        logger.warning(
            "Library name is not specified. There are multiple possible variants: `sentence_transformers`, `transformers`."
            " `transformers` will be selected. If you want to load your model with the `sentence-transformers` library instead, please set --library sentence_transformers"
        )
        library_name = "transformers"
    return library_name


def _compatible_type(default_val: Any, new_val: Any) -> bool:
    """Loose type check: allow ints for floats, bool as bool, etc."""
    if default_val is None:
        return True
    if isinstance(default_val, bool):
        return isinstance(new_val, bool)
    if isinstance(default_val, numbers.Real) and not isinstance(default_val, bool):
        return isinstance(new_val, numbers.Real) and not isinstance(new_val, bool)
    if isinstance(default_val, str):
        return isinstance(new_val, str)
    if isinstance(default_val, (list, tuple)):
        return isinstance(new_val, (list, tuple))
    if isinstance(default_val, Mapping):
        return isinstance(new_val, Mapping)
    return True  # fall back to permissive


def apply_genai_overrides(
    defaults: MutableMapping[str, Any], overrides: Mapping[str, Any], *, path: str = ""
) -> MutableMapping[str, Any]:
    """Recursively merge *overrides* into *defaults*.

    Only keys that already exist in *defaults* are updated. Type mismatches
    are logged as warnings but still applied.

    Args:
        defaults: The original config to be updated (modified in-place). MutableMapping[str, Any].
        overrides: The config values to override. Mapping[str, Any].
        path: The current path within the config (used for recursive calls).

    Returns:
        The updated config with overrides applied. MutableMapping[str, Any].

    """
    for k, v in overrides.items():
        here = f"{path}.{k}" if path else k
        if k not in defaults:
            continue

        dv = defaults[k]

        # Recurse for dicts
        if isinstance(dv, Mapping) and isinstance(v, Mapping):
            apply_genai_overrides(dv, v, path=here)
            continue

        # Replace lists/tuples and scalars
        if not _compatible_type(dv, v):
            logger.warning("Type mismatch at %s", here)
        defaults[k] = v
    return defaults


def create_genai_config(model_name: str, output_path: str, config: BasePassConfig) -> None:
    """Generate ``genai_config.json`` from model config files.

    This is only for Generative AI models for which ``config.json`` and
    ``generation_config.json`` exist in *output_path*.

    Args:
        model_name: Name of the ONNX model file that was generated.
        output_path: Directory containing the model and config files.
        config: Pass configuration instance (must expose ``target_device``; may
            optionally expose ``genai_config_override``).

    Returns:
        None

    Raises:
        FileNotFoundError: If required config files are missing.

    """
    ip_conf_pth = Path(output_path) / "config.json"

    # do not create genai_config.json if config.json does not exist
    if not ip_conf_pth.exists():
        return

    ip_gen_pth = Path(output_path) / "generation_config.json"

    # do not create genai_config.json if generation_config.json does not exist
    if not ip_gen_pth.exists():
        return

    # Step 1: Create the data structure
    genai_config: dict[str, Any] = {
        "model": {
            "bos_token_id": -1,
            "context_length": -1,
            "decoder": {
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "graph_optimization_level": "ORT_DISABLE_ALL",
                    "provider_options": [
                        {"OpenVINO": {"device_type": config.target_device.upper(), "enable_causallm": "True"}}
                    ],
                },
                "filename": "openvino_model.onnx",
                "head_size": -1,
                "hidden_size": -1,
                "inputs": {},
                "outputs": {},
                "num_attention_heads": -1,
                "num_hidden_layers": -1,
                "num_key_value_heads": -1,
            },
            "eos_token_id": -1,
            "type": "",
            "vocab_size": -1,
        },
        "search": {
            "diversity_penalty": 0.0,
            "do_sample": False,
            "early_stopping": True,
            "length_penalty": 1.0,
            "max_length": -1,
            "min_length": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": 1,
            "num_return_sequences": 1,
            "past_present_share_buffer": False,
            "repetition_penalty": 1.0,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
        },
    }

    with open(ip_conf_pth) as f:
        src_config = json.load(f)

    with open(ip_gen_pth) as f:
        src_gen_config = json.load(f)

    try:
        import onnx
    except ImportError:
        raise ImportError(
            "Please install onnx to create genai_config.json for ONNX OpenVINO IR Encapsulated model"
        ) from None

    model_path = Path(output_path) / model_name
    model = onnx.load(model_path)

    # Get input and output tensor names
    inputs = [inp.name for inp in model.graph.input]
    outputs = [out.name for out in model.graph.output]

    genai_config["model"]["bos_token_id"] = src_config.get("bos_token_id", -1)
    genai_config["model"]["context_length"] = src_config.get("max_position_embeddings", -1)
    genai_config["model"]["decoder"]["filename"] = model_name

    # Safe head_size computation
    num_attention_heads = src_config.get("num_attention_heads", -1)
    hidden_size = src_config.get("hidden_size", -1)
    if (
        isinstance(num_attention_heads, int)
        and isinstance(hidden_size, int)
        and num_attention_heads > 0
        and hidden_size >= 0
    ):
        genai_config["model"]["decoder"]["head_size"] = hidden_size // num_attention_heads
    else:
        if not isinstance(num_attention_heads, int):
            logger.warning("num_attention_heads is not an int: %s found in src_config", num_attention_heads)
        elif num_attention_heads <= 0:
            logger.warning("Invalid num_attention_heads (<= 0) %s found in src_config", num_attention_heads)
        if not isinstance(hidden_size, int):
            logger.warning("hidden_size is not an int: %s found in src_config", hidden_size)
        elif hidden_size < 0:
            logger.warning("Invalid hidden_size (< 0) %s found in src_config", hidden_size)
        logger.warning("Setting genai_config['model']['decoder']['head_size'] to -1")
        genai_config["model"]["decoder"]["head_size"] = -1

    genai_config["model"]["decoder"]["hidden_size"] = src_config.get("hidden_size", -1)

    for name in inputs:
        if name != "beam_idx":
            genai_config["model"]["decoder"]["inputs"].update({name: name})

    for name in outputs:
        genai_config["model"]["decoder"]["outputs"].update({name: name})

    genai_config["model"]["decoder"]["num_attention_heads"] = src_config.get("num_attention_heads", -1)
    genai_config["model"]["decoder"]["num_hidden_layers"] = src_config.get("num_hidden_layers", -1)
    genai_config["model"]["decoder"]["num_key_value_heads"] = src_config.get("num_key_value_heads", -1)

    eos_token_id = src_gen_config.get("eos_token_id", -1)
    genai_config["model"]["eos_token_id"] = eos_token_id
    pad_token_id = src_gen_config.get("pad_token_id", None)
    if pad_token_id is not None:
        genai_config["model"]["pad_token_id"] = pad_token_id
    elif eos_token_id != -1:
        genai_config["model"]["pad_token_id"] = (
            eos_token_id[0] if isinstance(eos_token_id, list) and len(eos_token_id) > 0 else eos_token_id
        )
    else:
        genai_config["model"]["pad_token_id"] = -1

    genai_config["model"]["type"] = src_config.get("model_type", "")
    genai_config["model"]["vocab_size"] = src_config.get("vocab_size", -1)

    genai_config["search"]["max_length"] = src_config.get("max_position_embeddings", -1)

    # Apply genai_config_override if the pass config exposes it
    genai_config_override = getattr(config, "genai_config_override", None)
    if isinstance(genai_config_override, dict):
        apply_genai_overrides(genai_config, genai_config_override)

    # Step 2: Write to JSON file
    output_genai_config = Path(output_path) / "genai_config.json"
    with open(output_genai_config, "w") as f:
        json.dump(genai_config, f, indent=4)
