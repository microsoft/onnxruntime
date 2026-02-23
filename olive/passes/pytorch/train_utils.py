# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import dataclasses
import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Union

import torch
import transformers
from packaging import version
from pydantic import Field, field_validator
from transformers import __version__ as transformers_version

from olive.common.config_utils import NestedConfig, validate_config
from olive.common.utils import cleanup_memory
from olive.data.config import DataConfig
from olive.data.template import huggingface_data_config_template
from olive.model import HfModelHandler, PyTorchModelHandler
from olive.model.config.hf_config import HfLoadKwargs

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from transformers import PreTrainedModel


# creating a Config class since transformers.TrainingArguments is a dataclass
# pydantic handles dataclasses differently and causes issues with validation
# this also allows us to handle and validate extra_args better
class BaseHFTrainingArguments(NestedConfig):
    """Training arguments for transformers.Trainer."""

    _nested_field_name: ClassVar[str] = "extra_args"

    gradient_checkpointing: bool = Field(True, description="Use gradient checkpointing. Recommended.")
    report_to: Union[str, list[str]] = Field(
        "none", description="The list of integrations to report the results and logs to."
    )
    output_dir: Optional[str] = Field(
        None, description="The output dir for logs and checkpoints. If None, will use a temp dir."
    )
    deepspeed: Optional[Union[bool, str, dict]] = Field(
        None,
        description=(
            "Use [Deepspeed](https://github.com/microsoft/deepspeed). If True, will use default deepspeed config. Else,"
            " it is a path to a deepspeed config file or a dict with deepspeed config."
        ),
    )
    extra_args: Optional[dict[str, Any]] = Field(
        None,
        description=(
            "Extra arguments to pass to the trainer. Values can be provided directly to this field as a dict or as"
            " keyword arguments to the config. See transformers.TrainingArguments for more details on the available"
            " arguments."
        ),
        validate_default=True,
    )

    @field_validator("extra_args", mode="before")
    @classmethod
    def validate_extra_args(cls, v):
        if v is None:
            v = {}
        # make sure extra args are fields of transformers.Trainer
        training_args_fields = {f.name for f in dataclasses.fields(transformers.TrainingArguments) if f.init}
        for k in list(v):  # need a copy of the keys since we are mutating the dict
            if k not in training_args_fields:
                logger.warning("Extra arg %s is not a field of transformers.TrainingArguments. Ignoring.", k)
                del v[k]
        return v

    def create_training_args(self) -> transformers.TrainingArguments:
        args = self.model_dump()
        if not args["output_dir"]:
            raise ValueError("output_dir must be provided.")
        if args["deepspeed"] is True:
            args["deepspeed"] = deepcopy(DEFAULT_DEEPSPEED_CONFIG)
        elif args["deepspeed"] is False:
            del args["deepspeed"]
        if version.parse(transformers_version) < version.parse("4.41") and "eval_strategy" in args:
            args["evaluation_strategy"] = args.pop("eval_strategy")
        extra_args = args.pop("extra_args")
        # Filter out fields that are not valid TrainingArguments parameters (e.g. overwrite_output_dir
        # was removed in transformers 5.0 but is still used by Olive's own logic) and None values
        # so that transformers uses its own defaults
        training_args_fields = {f.name for f in dataclasses.fields(transformers.TrainingArguments) if f.init}
        args = {k: v for k, v in args.items() if k in training_args_fields and v is not None}
        return transformers.TrainingArguments(**args, **extra_args)


def load_hf_base_model(
    model_handler: HfModelHandler,
    torch_dtype: Optional["torch.dtype"] = None,
    device_map: Optional[Union[int, str, dict]] = None,
    **kwargs,
) -> "PreTrainedModel":
    """Load a base PyTorch model.

    :param model_handler: The input model handler.
    :param torch_dtype: The torch dtype to load the model with.
        If None, will use the dtype from model_handler's load_kwargs if set, else will use "auto".
    :param device_map: The device map to load the model with.
    :param kwargs: Additional arguments to update load_kwargs with.
    :return: The new loaded pytorch model
    """
    # model cannot have it's own adapter
    if model_handler.adapter_path:
        raise ValueError("Model already has an adapter. Please provide a model without an adapter.")

    # don't want the original loaded model
    # also frees gpu memory if original model is on gpu
    model_handler.model = None
    cleanup_memory()

    # create copy of the input model, will modify this model
    # also resets adapter_path
    new_model_handler = deepcopy(model_handler)

    # load model, reset load_kwargs and adapter_path
    load_kwargs = new_model_handler.load_kwargs.model_dump() if new_model_handler.load_kwargs else {}
    load_kwargs.update(
        {
            # use "auto" as default to use the dtype from the model config
            # with new transformers versions, None doesn't use the model config dtype and instead uses float32
            "torch_dtype": torch_dtype or new_model_handler.get_load_kwargs().get("torch_dtype") or "auto"
        }
    )
    # Not all models support device_map. The default value of device_map is "auto".
    # User needs to set device_map to None if their model does not support device_map.
    if device_map:
        load_kwargs.update({"device_map": device_map})
    # overwrite load_kwargs with kwargs
    load_kwargs.update(kwargs)
    new_model_handler.load_kwargs = HfLoadKwargs(**load_kwargs)

    return new_model_handler.load_model(cache_model=False)


def prepare_model_for_finetuning(model: "PreTrainedModel", training_args: BaseHFTrainingArguments):
    """Prepare the model for fine-tuning.

    Freeze base model's layers and prepare model for gradient checkpointing if necessary.
    Similar to peft.prepare_model_for_kbit_training but no casting to fp32 and gradient checkpointing is
    also supported for non-quantized models.

    :param model: The Hugging Face PyTorch model to prepare for fine-tuning.
    :param training_args: The training arguments for the model.
    """
    for param in model.parameters():
        # freeze base model's layers
        param.requires_grad = False

    if training_args.gradient_checkpointing and not model.supports_gradient_checkpointing:
        logger.warning(
            "gradient_checkpointing is True, but model does not support gradient checkpointing! Setting"
            " gradient_checkpoing to False"
        )
        training_args.gradient_checkpointing = False
    elif training_args.gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module_, input_, output_):
                output_.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    logger.debug("The number of trainable parameters in the original model: %s", count_trainable_parameters(model))


def count_trainable_parameters(model) -> str:
    """Count and return the number of trainable parameters in a model."""
    trainable_params = 0
    all_param = 0
    for param in model.parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return (
        f"trainable params: {trainable_params} || all params: {all_param} "
        f"|| trainable%: {100 * trainable_params / all_param:.2f}"
    )


def get_training_dataset(data_config: DataConfig):
    """Get the training dataset from the data config."""
    from datasets import Dataset

    def data_generator(dataset):
        # not iterating over dataset directly since we only require loaded dataset to have __len__ and __getitem__
        for idx in range(len(dataset)):  # pylint: disable=consider-using-enumerate
            example = dataset[idx]
            if isinstance(example, tuple):
                # if example = {**example[0], "labels": example[1]}, the attention_mask is not the same
                # for some reason, so yield a new dict
                yield {**example[0], "labels": example[1]}
            else:
                yield example

    # each sample is an (input_dict, target) tuple
    data_container = data_config.to_data_container()
    dataset = data_container.pre_process(data_container.load_dataset())
    dataset = Dataset.from_generator(data_generator, gen_kwargs={"dataset": dataset})
    dataset.set_format("torch")

    return dataset


DEFAULT_DEEPSPEED_CONFIG = {
    "zero_optimization": {
        "stage": 3,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": "auto",
        "contiguous_gradients": True,
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "sub_group_size": 1e9,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": "auto",
        "offload_param": {
            "device": "cpu",
        },
        "offload_optimizer": {
            "device": "cpu",
        },
    },
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    "bf16": {"enabled": "auto"},
    "train_micro_batch_size_per_gpu": "auto",
    "train_batch_size": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
}


def get_calibration_dataset(
    model: HfModelHandler | PyTorchModelHandler,
    data_config: DataConfig | dict | None = None,
    split: str = "train[:1000]",
    batch_size: int = 1,
    max_seq_len: int = 2048,
    max_samples: int = 128,
) -> list[dict[str, Any]]:
    """Get the dataset for quantization calibration.

    Args:
        model: The HuggingFace or PyTorch model to get dataset for.
        data_config: Configuration object or dictionary containing data settings.
        split: The dataset split to use for default data config. Default is 'train[:1000]'.
        batch_size: The batch size to use for default data config. Default is 1.
        max_seq_len: Maximum sequence length for default data config. Default is 2048.
        max_samples: Maximum number of samples for default data config. Default is 128.

    Returns:
        List of tokenized data dictionaries for calibration.

    Raises:
        ValueError: If the dataset format is invalid.

    """
    if not data_config and isinstance(model, HfModelHandler):
        data_config = get_calibration_data_config(
            model.model_name_or_path,
            trust_remote_code=model.get_load_kwargs().get("trust_remote_code", None),
            split=split,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            max_samples=max_samples,
        )
    elif not data_config:
        raise ValueError("Data config is required for PyTorch model.")
    data_config = validate_config(data_config, DataConfig)
    dataloader = data_config.to_data_container().create_dataloader()
    # each batch consists of (input_data, labels)
    dataset = [data[0] for data in dataloader]

    if (
        not dataset
        or not isinstance(dataset, list)
        or not isinstance(dataset[0], dict)
        or ("input_ids" not in dataset[0] or "attention_mask" not in dataset[0])
    ):
        raise ValueError(
            "Provided dataset is invalid. The returned datasets is a list of tokenized data "
            "(e.g. [{ 'input_ids': [[ 1, 100, 15, ... ]],'attention_mask': [[ 1, 1, 1, ... ]]},...])"
        )

    return dataset


def get_calibration_data_config(
    model_name_or_path: str,
    trust_remote_code: bool | None = None,
    split: str = "train[:1000]",
    batch_size: int = 1,
    max_seq_len: int = 2048,
    max_samples: int = 128,
) -> DataConfig:
    """Get default calibration data configuration for GPTQ quantization.

    Args:
        model_name_or_path: Name or path of the model.
        trust_remote_code: Whether to trust remote code when loading data.
        split: The dataset split to use. Default is 'train[:1000]'.
        batch_size: The batch size to use. Default is 1.
        max_seq_len: Maximum sequence length. Default is 2048.
        max_samples: Maximum number of samples. Default is 128.

    Returns:
        DataConfig object for calibration data.

    """
    return huggingface_data_config_template(
        model_name=model_name_or_path,
        task="text-generation",
        load_dataset_config={
            "data_name": "Salesforce/wikitext",
            "subset": "wikitext-2-raw-v1",
            "split": split,
            "trust_remote_code": trust_remote_code,
        },
        pre_process_data_config={
            # should we randomize the data?
            "add_special_tokens": False,
            "max_seq_len": max_seq_len,
            "max_samples": max_samples,
            "trust_remote_code": trust_remote_code,
        },
        dataloader_config={"batch_size": batch_size},
    )


def kl_div_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Compute the KL divergence loss between student and teacher logits."""
    student_log_probs = torch.nn.functional.log_softmax(student_logits, dim=-1)
    teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=-1)
    kl = torch.nn.functional.kl_div(student_log_probs, teacher_probs, reduction="none")
    return kl.sum(-1)
