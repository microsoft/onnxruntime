# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import re
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import ClassVar, Optional

from olive.common.constants import DEFAULT_HF_TASK
from olive.common.user_module_loader import UserModuleLoader
from olive.common.utils import hf_repo_exists, set_nested_dict_value, unescaped_str
from olive.constants import DiffusersModelVariant
from olive.hardware.accelerator import AcceleratorSpec
from olive.hardware.constants import DEVICE_TO_EXECUTION_PROVIDERS
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS


class BaseOliveCLICommand(ABC):
    allow_unknown_args: ClassVar[bool] = False

    def __init__(self, parser: ArgumentParser, args: Namespace, unknown_args: Optional[list] = None):
        self.args = args
        self.unknown_args = unknown_args

        if unknown_args and not self.allow_unknown_args:
            parser.error(f"Unknown arguments: {unknown_args}")

    def _run_workflow(self):
        import tempfile

        from olive.workflows import run as olive_run

        Path(self.args.output_path).mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="olive-cli-tmp-", dir=self.args.output_path) as tempdir:
            run_config = self._get_run_config(tempdir)
            if self.args.save_config_file or self.args.dry_run:
                self._save_config_file(run_config)
            if self.args.dry_run:
                print("Dry run mode enabled. Configuration file is generated but no optimization is performed.")
                return None
            workflow_output = olive_run(run_config)
            if not workflow_output.has_output_model():
                print("No output model produced. Please check the log for details.")
            else:
                print(f"Model is saved at {self.args.output_path}")
            return workflow_output

    @staticmethod
    def _parse_extra_options(kv_items):
        from onnxruntime_genai import __version__ as OrtGenaiVersion
        from packaging import version

        if version.parse(OrtGenaiVersion) <= version.parse("0.9.0"):
            raise ValueError(
                "onnxruntime-genai version <= 0.9.0 is not supported for extra_options in CLI. "
                "Please either upgrade to onnxruntime-genai version > 0.9.0 or use the model builder pass directly in the config file."
            )

        from onnxruntime_genai.models.builder import parse_extra_options

        return parse_extra_options(kv_items)

    @staticmethod
    def _save_config_file(config: dict):
        """Save the config file."""
        config_file_path = Path(config["output_dir"]) / "config.json"
        with open(config_file_path, "w") as f:
            json.dump(config, f, indent=4)
        print(f"Config file saved at {config_file_path}")

    @staticmethod
    @abstractmethod
    def register_subcommand(parser: ArgumentParser):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError


def _get_hf_input_model(args: Namespace, model_path: OLIVE_RESOURCE_ANNOTATIONS) -> dict:
    """Get the input model config for HuggingFace model.

    args.task is optional.
    args.adapter_path might not be present.
    """
    print(f"Loading HuggingFace model from {model_path}")
    input_model = {
        "type": "HfModel",
        "model_path": model_path,
        "load_kwargs": {
            "attn_implementation": "eager",
        },
    }
    # use getattr to avoid AttributeError in case hf model or adapter_path is not supported
    # will let the command fail if hf model is returned even though it is not supported
    if getattr(args, "task", None):
        # conditional is needed since task=None is not handled in the model handler
        input_model["task"] = args.task
    if getattr(args, "adapter_path", None):
        input_model["adapter_path"] = args.adapter_path
    if getattr(args, "trust_remote_code", None) is not None:
        input_model["load_kwargs"]["trust_remote_code"] = args.trust_remote_code
    return input_model


def _get_onnx_input_model(args: Namespace, model_path: str) -> dict:
    """Get the input model config for ONNX model.

    Only supports local ONNX model file path.
    """
    print(f"Loading ONNX model from {model_path}")
    model_config = {
        "type": "OnnxModel",
        "model_path": model_path,
    }

    # additional processing for the model folder
    model_path = Path(model_path).resolve()
    if model_path.is_dir():
        onnx_files = list(model_path.glob("*.onnx"))
        if len(onnx_files) > 1:
            raise ValueError("Found multiple .onnx model files in the model folder. Please specify one.")
        onnx_file_path = onnx_files[0]
        model_config["onnx_file_name"] = onnx_file_path.name

        # all files other than the .onnx file and .onnx.data file considered as additional files
        additional_files = sorted(
            set({str(fp) for fp in model_path.iterdir()} - {str(onnx_file_path), str(onnx_file_path) + ".data"})
        )
        if additional_files:
            model_config["model_attributes"] = {"additional_files": additional_files}

    return model_config


def _get_pt_input_model(args: Namespace, model_path: OLIVE_RESOURCE_ANNOTATIONS) -> dict:
    """Get the input model config for PyTorch model.

    args.model_script is required.
    model_path is optional.
    """
    if not args.model_script:
        raise ValueError("model_script is not provided. Either model_name_or_path or model_script is required.")

    user_module_loader = UserModuleLoader(args.model_script, args.script_dir)

    if not model_path and not user_module_loader.has_function("_model_loader"):
        raise ValueError("Either _model_loader or model_name_or_path is required for PyTorch model.")

    input_model_config = {
        "type": "PyTorchModel",
        "model_script": args.model_script,
    }

    if args.script_dir:
        input_model_config["script_dir"] = args.script_dir

    if model_path:
        print("Loading PyTorch model from", model_path)
        input_model_config["model_path"] = model_path

    if user_module_loader.has_function("_model_loader"):
        print("Loading PyTorch model from function: _model_loader.")
        input_model_config["model_loader"] = "_model_loader"

    model_funcs = [
        ("io_config", "_io_config"),
        ("dummy_inputs_func", "_dummy_inputs"),
        ("model_file_format", "_model_file_format"),
    ]
    input_model_config.update(
        {config_key: func_name for config_key, func_name in model_funcs if user_module_loader.has_function(func_name)}
    )

    if "io_config" not in input_model_config and "dummy_inputs_func" not in input_model_config:
        raise ValueError("_io_config or _dummy_inputs is required in the script for PyTorch model.")
    return input_model_config


def get_diffusers_input_model(args: Namespace, model_path: OLIVE_RESOURCE_ANNOTATIONS) -> dict:
    """Get the input model config for Diffusers model.

    args.adapter_path might not be present.
    args.model_variant might not be present.
    """
    print(f"Loading Diffusers model from {model_path}")
    input_model = {
        "type": "DiffusersModel",
        "model_path": model_path,
    }
    if getattr(args, "adapter_path", None):
        input_model["adapter_path"] = args.adapter_path
    if getattr(args, "model_variant", None):
        input_model["model_variant"] = args.model_variant
    return input_model


def get_input_model_config(args: Namespace, required: bool = True) -> Optional[dict]:
    """Parse the model_name_or_path and return the input model config.

    Check model_name_or_path formats in order:
    1. Local PyTorch model with model loader but no model path
    2. Output of a previous command
    3. Load PyTorch model with model_script
    4. azureml://registries/<registry_name>/models/<model_name>/versions/<version> (only for HF model)
    5. https://huggingface.co/<model_name> (only for HF model)
    6. HF model name string
    7. local file path
      a. local onnx model file path (either a user-provided model or a model produced by the Olive CLI)
      b. local HF model file path (either a user-provided model or a model produced by the Olive CLI)
    """
    model_name_or_path = args.model_name_or_path

    if model_name_or_path is None:
        if hasattr(args, "model_script"):
            if args.model_script:
                # pytorch model with model_script, model_path is optional
                print("model_name_or_path is not provided. Using model_script to load the model.")
                return _get_pt_input_model(args, None)
            elif required:
                raise ValueError(
                    "model_name_or_path is required. Either model_name_or_path or model_script is required."
                )
        if not required:
            # optional model_name_or_path, return empty config
            return None
        raise ValueError("model_name_or_path is required.")

    model_path = Path(model_name_or_path)
    # check if is the output of a previous command
    if model_path.is_dir() and (model_path / "model_config.json").exists():
        with open(model_path / "model_config.json") as f:
            model_config = json.load(f)

        if adapter_path := getattr(args, "adapter_path", None):
            assert model_config["type"].lower() == "hfmodel", "Only HfModel supports adapter_path."
            model_config["config"]["adapter_path"] = adapter_path

        print(f"Loaded previous command output of type {model_config['type']} from {model_name_or_path}")
        return model_config

    if getattr(args, "model_script", None):
        return _get_pt_input_model(args, model_name_or_path)

    # Check AzureML Registry model
    pattern = (
        r"^azureml://registries/(?P<registry_name>[^/]+)/models/(?P<model_name>[^/]+)/versions/(?P<version>[^/]+)$"
    )
    match = re.match(pattern, model_name_or_path)
    if match:
        return _get_hf_input_model(
            args,
            {
                "type": "azureml_registry_model",
                "registry_name": match.group("registry_name"),
                "name": match.group("model_name"),
                "version": match.group("version"),
            },
        )

    # Check HuggingFace url
    pattern = r"https://huggingface\.co/([^/]+/[^/]+)(?:/.*)?"
    match = re.search(pattern, model_name_or_path)
    if match:
        return _get_hf_input_model(args, match.group(1))

    # Check HF model name string
    if not model_path.exists():
        if not hf_repo_exists(model_name_or_path):
            raise ValueError(f"{model_name_or_path} is not a valid Huggingface model name.")
        return _get_hf_input_model(args, model_name_or_path)

    # Check local onnx file/folder (user-provided model)
    if (model_path.is_file() and model_path.suffix == ".onnx") or any(model_path.glob("*.onnx")):
        return _get_onnx_input_model(args, model_name_or_path)

    # Check local HF model file (user-provided model)
    return _get_hf_input_model(args, model_name_or_path)


def update_input_model_options(args, config):
    config["input_model"] = get_input_model_config(args)


def add_logging_options(sub_parser: ArgumentParser, default: int = 3):
    """Add logging options to the sub_parser."""
    sub_parser.add_argument(
        "--log_level",
        type=int,
        default=default,
        help=f"Logging level. Default is {default}. level 0: DEBUG, 1: INFO, 2: WARNING, 3: ERROR, 4: CRITICAL",
    )
    return sub_parser


def add_save_config_file_options(sub_parser: ArgumentParser):
    """Add save config file options to the sub_parser."""
    sub_parser.add_argument(
        "--save_config_file",
        action="store_true",
        help="Generate and save the config file for the command.",
    )

    sub_parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Enable dry run mode. This will not perform any actual optimization but will validate the configuration.",
    )

    return sub_parser


def add_input_model_options(
    sub_parser: ArgumentParser,
    enable_hf: bool = False,
    enable_hf_adapter: bool = False,
    enable_pt: bool = False,
    enable_onnx: bool = False,
    enable_diffusers: bool = False,
    default_output_path: Optional[str] = None,
    directory_output: bool = True,
    required: bool = True,
):
    """Add model options to the sub_parser.

    Use enable_hf, enable_hf_adapter, enable_pt, enable_onnx, enable_diffusers
    to enable the corresponding model options.
    If default_output_path is None, it is required to provide the output_path.
    If directory_output is True, the output_path is a directory and will be created if it doesn't exist.
    """
    assert any([enable_hf, enable_hf_adapter, enable_pt, enable_onnx, enable_diffusers]), (
        "At least one model option should be enabled."
    )

    model_group = sub_parser

    model_group.add_argument(
        "-m",
        "--model_name_or_path",
        type=str,
        # only pytorch model doesn't require model_name_or_path
        required=required and not enable_pt,
        help=(
            "Path to the input model. "
            "See https://microsoft.github.io/Olive/reference/cli.html#providing-input-models "
            "for more information."
        ),
    )
    if enable_hf:
        model_group.add_argument(
            "-t",
            "--task",
            type=str,
            help=f"Task for which the huggingface model is used. Default task is {DEFAULT_HF_TASK}.",
        )
        model_group.add_argument(
            "--trust_remote_code", action="store_true", help="Trust remote code when loading a huggingface model."
        )

    if enable_hf_adapter:
        assert enable_hf, "enable_hf must be True when enable_hf_adapter is True."
        model_group.add_argument(
            "-a",
            "--adapter_path",
            type=str,
            help="Path to the adapters weights saved after peft fine-tuning. Local folder or huggingface id.",
        )
    if enable_diffusers:
        model_group.add_argument(
            "--model_variant",
            type=DiffusersModelVariant,
            choices=[
                DiffusersModelVariant.AUTO,
                DiffusersModelVariant.SD,
                DiffusersModelVariant.SDXL,
                DiffusersModelVariant.SD3,
                DiffusersModelVariant.FLUX,
                DiffusersModelVariant.SANA,
            ],
            default=DiffusersModelVariant.AUTO,
            help="Model variant: 'sd', 'sdxl', 'sd3', 'flux', 'sana', or 'auto' for auto-detection.",
        )
        if not enable_hf_adapter:
            # Add adapter_path for diffusers if not already added by enable_hf_adapter
            model_group.add_argument(
                "-a",
                "--adapter_path",
                type=str,
                help="Path to the LoRA adapter weights. Local folder or huggingface id.",
            )
    if enable_pt:
        model_group.add_argument(
            "--model_script",
            type=str,
            help="The script file containing the model definition. Required for the local PyTorch model.",
        )
        model_group.add_argument(
            "--script_dir",
            type=str,
            help=(
                "The directory containing the local PyTorch model script file."
                " See https://microsoft.github.io/Olive/reference/cli.html#model-script-file-information "
                "for more information."
            ),
        )
    model_group.add_argument(
        "-o",
        "--output_path",
        type=output_path_type if directory_output else str,
        required=required and default_output_path is None,
        default=default_output_path,
        help="Path to save the command output.",
    )
    return model_group


def output_path_type(path: str) -> str:
    """Resolve the output path and mkdir if it doesn't exist."""
    path = Path(path).resolve()

    if path.exists():
        assert path.is_dir(), f"{path} is not a directory."

    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def add_dataset_options(sub_parser, required=True, include_train=True, include_eval=True):
    dataset_group = sub_parser
    dataset_group.add_argument(
        "-d",
        "--data_name",
        type=str,
        required=required,
        help="The dataset name.",
    )

    if include_train:
        dataset_group.add_argument("--train_subset", type=str, help="The subset to use for training.")
        dataset_group.add_argument("--train_split", type=str, default="train", help="The split to use for training.")

    if include_eval:
        dataset_group.add_argument("--eval_subset", type=str, help="The subset to use for evaluation.")
        dataset_group.add_argument("--eval_split", default="", help="The dataset split to evaluate on.")

    if not (include_train and include_eval):
        dataset_group.add_argument("--subset", type=str, help="The subset of the dataset to use.")
        dataset_group.add_argument("--split", type=str, help="The dataset split to use.")

    # TODO(jambayk): currently only supports single file or list of files, support mapping
    dataset_group.add_argument(
        "--data_files", type=str, help="The dataset files. If multiple files, separate by comma."
    )

    text_group = dataset_group.add_mutually_exclusive_group(required=False)
    text_group.add_argument(
        "--text_field",
        type=str,
        help="The text field to use for fine-tuning.",
    )
    text_group.add_argument(
        "--text_template",
        # using special string type to allow for escaped characters like \n
        type=unescaped_str,
        help=r"Template to generate text field from. E.g. '### Question: {prompt} \n### Answer: {response}'",
    )
    text_group.add_argument("--use_chat_template", action="store_true", help="Use chat template for text field.")
    dataset_group.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="Maximum sequence length for the data.",
    )
    dataset_group.add_argument(
        "--add_special_tokens",
        type=bool,
        default=False,
        help="Whether to add special tokens during preprocessing.",
    )
    dataset_group.add_argument(
        "--max_samples",
        type=int,
        default=256,
        help="Maximum samples to select from the dataset.",
    )
    dataset_group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size.",
    )
    dataset_group.add_argument(
        "--input_cols",
        type=str,
        nargs="+",
        help=(
            "List of input column names. Provide one or more names separated by space. Example: --input_cols sentence1"
            " sentence2"
        ),
    )

    return dataset_group, text_group


def update_dataset_options(args, config):
    load_key = ("data_configs", 0, "load_dataset_config")
    preprocess_key = ("data_configs", 0, "pre_process_data_config")
    dataloader_key = ("data_configs", 0, "dataloader_config")
    split = args.train_split if hasattr(args, "train_split") else args.split
    subset = args.train_subset if hasattr(args, "train_subset") else args.subset

    to_replace = [
        ((*load_key, "data_name"), args.data_name),
        ((*load_key, "split"), split),
        ((*load_key, "subset"), subset),
        (
            (*load_key, "data_files"),
            args.data_files.split(",") if args.data_files else None,
        ),
        ((*preprocess_key, "text_cols"), args.text_field),
        ((*preprocess_key, "text_template"), args.text_template),
        ((*preprocess_key, "chat_template"), args.use_chat_template),
        ((*preprocess_key, "max_seq_len"), args.max_seq_len),
        ((*preprocess_key, "add_special_tokens"), args.add_special_tokens),
        ((*preprocess_key, "input_cols"), args.input_cols),
        ((*preprocess_key, "max_samples"), args.max_samples),
        ((*dataloader_key, "batch_size"), args.batch_size),
    ]
    for keys, value in to_replace:
        if value is not None:
            set_nested_dict_value(config, keys, value)


def add_shared_cache_options(sub_parser: ArgumentParser):
    shared_cache_group = sub_parser
    shared_cache_group.add_argument(
        "--account_name",
        type=str,
        help="Azure storage account name for shared cache.",
    )
    shared_cache_group.add_argument(
        "--container_name",
        type=str,
        help="Azure storage container name for shared cache.",
    )


def update_shared_cache_options(config, args):
    config["cache_config"] = {
        "account_name": args.account_name,
        "container_name": args.container_name,
    }


def add_accelerator_options(sub_parser, single_provider: bool = True):
    accelerator_group = sub_parser

    accelerator_group.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["gpu", "cpu", "npu"],
        help="Target device to run the model. Default is cpu.",
    )

    execution_providers = sorted(
        {provider for provider_list in DEVICE_TO_EXECUTION_PROVIDERS.values() for provider in provider_list}
    )

    if single_provider:
        accelerator_group.add_argument(
            "--provider",
            type=str,
            default="CPUExecutionProvider",
            choices=execution_providers,
            help="Execution provider to use for ONNX model. Default is CPUExecutionProvider.",
        )
    else:
        accelerator_group.add_argument(
            "--providers_list",
            type=str,
            nargs="*",
            choices=execution_providers,
            help=(
                "List of execution providers to use for ONNX model. They are case sensitive. "
                "If not provided, all available providers will be used."
            ),
        )
    accelerator_group.add_argument(
        "--memory",
        type=AcceleratorSpec.str_to_int_memory,
        default=None,
        help="Memory limit for the accelerator in bytes. Default is None.",
    )

    return accelerator_group


def update_accelerator_options(args, config, single_provider: bool = True):
    execution_providers = [args.provider] if single_provider else args.providers_list
    to_replace = [
        (("systems", "local_system", "accelerators", 0, "device"), args.device),
        (("systems", "local_system", "accelerators", 0, "execution_providers"), execution_providers),
        (("systems", "local_system", "accelerators", 0, "memory"), args.memory),
    ]
    for k, v in to_replace:
        if v is not None:
            set_nested_dict_value(config, k, v)


def add_search_options(sub_parser: ArgumentParser):
    search_strategy_group = sub_parser
    search_strategy_group.add_argument(
        "--enable_search",
        type=str,
        default=None,
        const="sequential",
        nargs="?",
        choices=["random", "sequential", "tpe"],
        help=(
            "Enable search to produce optimal model for the given criteria. "
            "Optionally provide sampler from available choices. "
            "By default, uses sequential sampler."
        ),
    )
    search_strategy_group.add_argument("--seed", type=int, default=0, help="Random seed for search sampler")


def add_telemetry_options(sub_parser: ArgumentParser):
    """Add telemetry options to the sub_parser."""
    sub_parser.add_argument("--disable_telemetry", action="store_true", help="Disable telemetry for this command.")
    return sub_parser


def update_search_options(args, config):
    to_replace = []
    to_replace.extend(
        [
            (
                "search_strategy",
                {
                    "execution_order": "joint",
                    "sampler": args.enable_search,
                    "seed": args.seed,
                },
            ),
        ]
    )

    for keys, value in to_replace:
        if value is None:
            continue
        set_nested_dict_value(config, keys, value)
