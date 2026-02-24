# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn as nn

from olive.common.config_utils import validate_config
from olive.constants import Framework
from olive.data.config import DataComponentConfig, DataConfig
from olive.data.registry import Registry
from olive.data.template import huggingface_data_config_template
from olive.evaluator.metric import AccuracySubType, LatencySubType, Metric, MetricType
from olive.evaluator.metric_config import MetricGoal
from olive.model import HfModelHandler, ModelConfig, ONNXModelHandler, PyTorchModelHandler
from olive.passes.olive_pass import Pass, create_pass_from_dict

ONNX_MODEL_PATH = Path(__file__).absolute().parent / "dummy_model.onnx"


class DummyModel(nn.Module):
    def __init__(self, batch_size=1):
        super().__init__()
        self.fc1 = nn.Linear(batch_size, 10)

    def forward(self, x):
        return torch.sigmoid(self.fc1(x))


def pytorch_model_loader(model_path):
    return DummyModel().eval()


def get_pytorch_model_io_config(batch_size=1):
    return {
        "input_names": ["input"],
        "output_names": ["output"],
        "input_shapes": [(batch_size, 1)],
    }


def get_pytorch_model_dynamic_shapes():
    """Get dynamic_shapes dict for dynamo export to preserve batch dimension."""
    from torch.export import Dim

    batch_size = Dim("batch_size")
    # DummyModel.forward(x) takes input 'x' with shape (batch_size, 1)
    # Mark first dim as dynamic to preserve batch dimension
    return {"x": {0: batch_size}}


def get_pytorch_model_dummy_input(model=None, batch_size=1):
    return torch.randn(batch_size, 1)


def get_pytorch_model_config(batch_size=1):
    config = {
        "type": "PyTorchModel",
        "config": {
            "model_loader": pytorch_model_loader,
            "io_config": get_pytorch_model_io_config(batch_size),
        },
    }
    return ModelConfig.model_validate(config)


def get_pytorch_model(batch_size=1):
    return PyTorchModelHandler(
        model_loader=pytorch_model_loader,
        model_path=None,
        io_config=get_pytorch_model_io_config(batch_size),
    )


def get_hf_model(model_path="hf-internal-testing/tiny-random-LlamaForCausalLM"):
    return HfModelHandler(model_path=model_path, task="text-generation")


def get_hf_model_config():
    return ModelConfig.model_validate(get_hf_model().to_json())


def create_onnx_model_file():
    pytorch_model = pytorch_model_loader(model_path=None)
    dummy_input = get_pytorch_model_dummy_input(pytorch_model)
    io_config = get_pytorch_model_io_config()
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        ONNX_MODEL_PATH,
        input_names=io_config["input_names"],
        output_names=io_config["output_names"],
        external_data=False,
        dynamo=True,
    )


def create_onnx_model_with_dynamic_axis(onnx_model_path, batch_size=1):
    pytorch_model = pytorch_model_loader(model_path=None)
    dummy_input = get_pytorch_model_dummy_input(pytorch_model, batch_size)
    io_config = get_pytorch_model_io_config()
    dynamic_shapes = get_pytorch_model_dynamic_shapes()
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        onnx_model_path,
        input_names=io_config["input_names"],
        output_names=io_config["output_names"],
        external_data=False,
        dynamo=True,
        dynamic_shapes=dynamic_shapes,
    )


def get_onnx_model_config(model_path=None):
    return ModelConfig.model_validate(
        {"type": "ONNXModel", "config": {"model_path": str(model_path or ONNX_MODEL_PATH)}}
    )


def get_composite_onnx_model_config(model_path=None):
    onnx_model_config = get_onnx_model_config(model_path).model_dump()
    return ModelConfig.model_validate(
        {
            "type": "CompositeModel",
            "config": {
                "model_components": [onnx_model_config, onnx_model_config],
                "model_component_names": "test_component_name",
            },
        }
    )


def get_onnx_model(model_attributes=None):
    return ONNXModelHandler(model_path=str(ONNX_MODEL_PATH), model_attributes=model_attributes)


def delete_onnx_model_files():
    if os.path.exists(ONNX_MODEL_PATH):
        os.remove(ONNX_MODEL_PATH)


def get_mock_openvino_model():
    olive_model = MagicMock()
    olive_model.framework = Framework.OPENVINO
    return olive_model


def _get_dummy_data_config(name, input_shapes, max_samples=1):
    data_config = DataConfig(
        name=name,
        type="DummyDataContainer",
        load_dataset_config=DataComponentConfig(
            params={
                "input_shapes": input_shapes,
                "max_samples": max_samples,
            }
        ),
        post_process_data_config=DataComponentConfig(type="text_classification_post_process"),
    )
    return validate_config(data_config, DataConfig)


def get_accuracy_metric(
    *acc_subtype,
    user_config=None,
    backend="torch_metrics",
    goal_type="threshold",
    goal_value=0.99,
):
    accuracy_score_metric_config = {"task": "multiclass", "num_classes": 10}
    sub_types = [
        {
            "name": sub,
            "metric_config": accuracy_score_metric_config if sub == "accuracy_score" else {},
            "goal": MetricGoal(type=goal_type, value=goal_value),
        }
        for sub in acc_subtype
    ]
    sub_types[0]["priority"] = 1
    return Metric(
        name="accuracy",
        type=MetricType.ACCURACY,
        sub_types=sub_types,
        user_config=user_config,
        backend=backend,
        data_config=_get_dummy_data_config("accuracy_metric_data_config", [[1, 1]]),
    )


def get_glue_accuracy_metric():
    return Metric(
        name="accuracy",
        type=MetricType.ACCURACY,
        sub_types=[{"name": AccuracySubType.ACCURACY_SCORE}],
        data_config=get_glue_huggingface_data_config(),
    )


def get_glue_latency_metric():
    return Metric(
        name="latency",
        type=MetricType.LATENCY,
        sub_types=[{"name": LatencySubType.AVG}],
        data_config=get_glue_huggingface_data_config(),
    )


def get_custom_metric(user_config=None):
    user_script_path = str(Path(__file__).absolute().parent / "assets" / "user_script.py")
    return Metric(
        name="custom",
        type=MetricType.CUSTOM,
        sub_types=[{"name": "custom"}],
        user_config=user_config or {"evaluate_func": "eval_func", "user_script": user_script_path},
    )


def get_custom_metric_no_eval():
    custom_metric = get_custom_metric()
    custom_metric.user_config.evaluate_func = None
    return custom_metric


def get_latency_metric(*lat_subtype, user_config=None):
    sub_types = [{"name": sub} for sub in lat_subtype]
    return Metric(
        name="latency",
        type=MetricType.LATENCY,
        sub_types=sub_types,
        user_config=user_config,
        data_config=_get_dummy_data_config("latency_metric_data_config", [[1, 1]]),
    )


def get_throughput_metric(*lat_subtype, user_config=None):
    sub_types = [{"name": sub} for sub in lat_subtype]
    return Metric(
        name="throughput",
        type=MetricType.THROUGHPUT,
        sub_types=sub_types,
        user_config=user_config,
        data_config=_get_dummy_data_config("throughput_metric_data_config", [[1, 1]]),
    )


def get_onnxconversion_pass(target_opset=None) -> type[Pass]:
    from olive.passes.onnx.conversion import OnnxConversion

    onnx_conversion_config = {"use_dynamo_exporter": True}
    if target_opset is not None:
        onnx_conversion_config["target_opset"] = target_opset
    return create_pass_from_dict(OnnxConversion, onnx_conversion_config)


def get_onnx_dynamic_quantization_pass(disable_search=False):
    from olive.passes.onnx.quantization import OnnxDynamicQuantization

    return create_pass_from_dict(OnnxDynamicQuantization, disable_search=disable_search)


def get_data_config():
    @Registry.register_dataset("test_dataset")
    def _test_dataset(data_dir, test_value): ...

    @Registry.register_dataloader()
    def _test_dataloader(dataset, test_value): ...

    @Registry.register_pre_process()
    def _pre_process(dataset, test_value): ...

    @Registry.register_post_process()
    def _post_process(output, test_value): ...

    return DataConfig(
        name="test_data_config",
        load_dataset_config=DataComponentConfig(
            type="test_dataset",  # renamed by Registry.register_dataset
            params={"test_value": "test_value"},
        ),
        dataloader_config=DataComponentConfig(
            type="_test_dataloader",  # This is the key to get dataloader
            params={"test_value": "test_value"},
        ),
    )


def get_glue_huggingface_data_config():
    return DataConfig(
        name="glue_huggingface_data_config",
        type="HuggingfaceContainer",
        load_dataset_config=DataComponentConfig(
            params={
                "data_name": "nyu-mll/glue",
                "subset": "mrpc",
                "split": "validation",
                "batch_size": 1,
            }
        ),
        pre_process_data_config=DataComponentConfig(
            params={
                "model_name": "Intel/bert-base-uncased-mrpc",
                "task": "text-classification",
                "input_cols": ["sentence1", "sentence2"],
                "label_col": "label",
            }
        ),
        post_process_data_config=DataComponentConfig(
            params={
                "model_name": "Intel/bert-base-uncased-mrpc",
                "task": "text-classification",
            }
        ),
    )


def get_transformer_dummy_input_data_config():
    return DataConfig(
        name="transformer_token_dummy_data",
        type="TransformersTokenDummyDataContainer",
        load_dataset_config=DataComponentConfig(
            params={
                "model_name": "Intel/bert-base-uncased-mrpc",
                "use_step": True,
            }
        ),
        dataloader_config=DataComponentConfig(
            params={
                "batch_size": 2,
            }
        ),
    )


def create_raw_data(raw_data_dir, input_names, input_shapes, input_types=None, num_samples=1):
    data_dir = Path(raw_data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    input_types = input_types or ["float32"] * len(input_names)

    num_samples_digits = len(str(num_samples))

    data = {}
    for input_name, input_shape, input_type in zip(input_names, input_shapes, input_types):
        data[input_name] = []
        input_dir = data_dir / input_name
        input_dir.mkdir(parents=True, exist_ok=True)
        for i in range(num_samples):
            data_i = np.random.rand(*input_shape).astype(input_type)
            data_i.tofile(input_dir / f"{i}.bin".zfill(num_samples_digits + 4))
            data[input_name].append(data_i)

    return data


def make_local_tiny_llama(save_path, model_type="hf"):
    # this checkpoint has an invalid generation config that cannot be saved
    input_model = HfModelHandler(
        model_path="hf-internal-testing/tiny-random-LlamaForCausalLM", load_kwargs={"pad_token_id": 1}
    )
    loaded_model = input_model.load_model()

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    if model_type == "hf":
        loaded_model.save_pretrained(save_path)
    else:
        onnx_file_path = save_path / "model.onnx"
        onnx_file_path.write_text("dummy onnx file")
        loaded_model.config.save_pretrained(save_path)
        loaded_model.generation_config.save_pretrained(save_path)
    input_model.get_hf_tokenizer().save_pretrained(save_path)

    return (
        HfModelHandler(model_path=save_path)
        if model_type == "hf"
        else ONNXModelHandler(model_path=save_path, onnx_file_name="model.onnx")
    )


def get_tiny_phi3():
    return HfModelHandler(
        model_path="katuni4ka/tiny-random-phi3", load_kwargs={"revision": "585361abfee667f3c63f8b2dc4ad58405c4e34e2"}
    )


def get_wikitext_data_config(
    model_name_or_path, max_seq_len=1024, max_samples=1, strategy="join-random", pad_to_max_len=True
):
    return huggingface_data_config_template(
        model_name=model_name_or_path,
        task="text-generation",
        load_dataset_config={
            "data_name": "Salesforce/wikitext",
            "subset": "wikitext-2-raw-v1",
            "split": "train[:1000]",
        },
        pre_process_data_config={
            "strategy": strategy,
            "max_seq_len": max_seq_len,
            "max_samples": max_samples,
            "pad_to_max_len": pad_to_max_len,
            "random_seed": 42,
        },
    )


def package_version_at_least(package_name: str, min_ver: str) -> bool:
    """Return True if *package_name* is installed and its version is >= *min_ver*, False otherwise.

    Intended for use in ``pytest.mark.skipif`` conditions where the check
    must never raise during test collection.
    """
    try:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as pkg_version

        from packaging.version import InvalidVersion
        from packaging.version import parse as parse_version
    except ImportError:
        return False

    try:
        return parse_version(pkg_version(package_name)) >= parse_version(min_ver)
    except (PackageNotFoundError, InvalidVersion):
        return False
