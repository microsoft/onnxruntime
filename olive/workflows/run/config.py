# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import shutil
from pathlib import Path
from typing import Any, ClassVar, Optional, Union

from pydantic import Field, field_validator, model_validator

from olive.cache import CacheConfig
from olive.common.config_utils import NestedConfig, validate_config
from olive.common.constants import DEFAULT_CACHE_DIR, DEFAULT_HF_TASK, DEFAULT_WORKFLOW_ID
from olive.data.config import DataComponentConfig, DataConfig
from olive.data.container.dummy_data_container import TRANSFORMER_DUMMY_DATA_CONTAINER
from olive.data.container.huggingface_container import HuggingfaceContainer
from olive.engine import Engine
from olive.engine.config import EngineConfig, RunPassConfig
from olive.engine.packaging.packaging_config import PackagingConfig
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.model import ModelConfig
from olive.passes.pass_config import PassParamDefault
from olive.systems.common import SystemType
from olive.systems.system_config import SystemConfig


class RunEngineConfig(EngineConfig):
    evaluate_input_model: bool = Field(
        True,
        description="When true, input model will be evaluated using the engine's evaluator. Set this to false to skip.",
    )
    output_dir: Optional[Union[Path, str]] = Field(
        None,
        description="Path where final output get saved.",
        validate_default=True,
    )
    packaging_config: Optional[Union[PackagingConfig, list[PackagingConfig]]] = Field(
        None, description="Artifacts packaging configuration."
    )
    cache_config: Optional[Union[CacheConfig, dict[str, Any]]] = Field(
        None, description="Cache configuration to speed up workflow runs."
    )
    cache_dir: Union[str, Path, list[str]] = Field(
        DEFAULT_CACHE_DIR,
        description=(
            "Path where intermediate results get saved.  Default is .olive-cache in the current working directory."
        ),
    )
    clean_cache: bool = Field(False, description="Set this to true to force clean the cache on next run.")
    clean_evaluation_cache: bool = Field(
        False, description="Set this to true to limit cache clean to generated evaluation results only."
    )
    enable_shared_cache: bool = Field(False, description="Set this to true to enable shared cache.")
    log_severity_level: int = Field(
        1,
        description=(
            "Logging level. Default is 3. Available options: 0: DEBUG, 1: INFO, 2: WARNING, 3: ERROR, 4: CRITICAL"
        ),
    )
    ort_log_severity_level: int = Field(
        3,
        description=(
            "Log severity level for ONNX Runtime C++ logs. "
            "Default is 3. Available options: 0: VERBOSE, 1: INFO, 2: WARNING, 3: ERROR, 4: FATAL"
        ),
    )
    ort_py_log_severity_level: int = Field(
        3,
        description=(
            "Log severity level for ONNX Runtime Python logs. "
            "Available options: 0: VERBOSE, 1: INFO, 2: WARNING, 3: ERROR, 4: FATAL"
        ),
    )
    log_to_file: bool = Field(
        False,
        description=(
            "Set this to true to reroute logs to a file instead of the console. "
            "If `true`, the log will be stored in a olive-<timestamp>.log file under the current working directory."
        ),
    )

    @field_validator("output_dir", mode="before")
    @classmethod
    def validate_output_dir(cls, v):
        if v is None:
            v = Path.cwd().resolve()
        else:
            v = Path(v).resolve()
        return v

    def create_engine(self, olive_config, workflow_id):
        config = self.model_dump(include=EngineConfig.model_fields.keys())
        if self.cache_config:
            cache_config = validate_config(self.cache_config, CacheConfig)
        else:
            cache_config = CacheConfig(
                cache_dir=self.cache_dir,
                clean_cache=self.clean_cache,
                clean_evaluation_cache=self.clean_evaluation_cache,
                enable_shared_cache=self.enable_shared_cache,
            )
        return Engine(
            **config,
            olive_config=olive_config,
            cache_config=cache_config,
            workflow_id=workflow_id,
        )


class RunConfig(NestedConfig):
    """Run configuration for Olive workflow.

    This is the top-level configuration. It includes configurations for input model, systems, data,
    evaluators, engine, passes, and auto optimizer.
    """

    _nested_field_name: ClassVar[str] = "engine"

    workflow_id: str = Field(
        DEFAULT_WORKFLOW_ID, description="Workflow ID. If not provided, use the default ID 'default_workflow'."
    )
    input_model: ModelConfig = Field(description="Input model configuration.")
    systems: Optional[dict[str, SystemConfig]] = Field(
        None,
        description="System configurations. Other fields such as engine and passes can refer to these systems by name.",
    )
    data_configs: list[DataConfig] = Field(
        default_factory=list,
        description=(
            "Data configurations. Each data config must have a unique name. Other fields such as engine, passes and"
            " evaluators can refer to these data configs by name. In auto-optimizer mode, only one data config is"
            " allowed."
        ),
    )
    evaluators: Optional[dict[str, OliveEvaluatorConfig]] = Field(
        None,
        description=(
            "Evaluator configurations. Other fields such as engine and passes can refer to these evaluators by name."
        ),
    )
    engine: RunEngineConfig = Field(
        default_factory=RunEngineConfig,
        description=(
            "Engine configuration. If not provided, the workflow uses the default engine configuration which runs in"
            " no-search or auto-optimizer mode based on whether passes field is provided."
        ),
    )
    passes: dict[str, list[RunPassConfig]] = Field(default_factory=dict, description="Pass configurations.")

    @model_validator(mode="before")
    @classmethod
    def patch_evaluators(cls, values):
        # In pydantic v2, values can be None when no arguments are provided
        if values is None:
            values = {}

        if "evaluators" in values:
            for name, evaluator_config in values["evaluators"].items():
                evaluator_config["name"] = name
        return values

    @model_validator(mode="before")
    @classmethod
    def patch_passes(cls, values):
        # In pydantic v2, values can be None when no arguments are provided
        if values is None:
            values = {}

        if "passes" in values:
            for name, passes_config in values["passes"].items():
                if isinstance(passes_config, dict):
                    values["passes"][name] = [passes_config]
        return values

    @model_validator(mode="after")
    def validate_python_environment_paths(self):  # noqa: N804  # model_validator mode="after" uses self
        # Check if we need to validate python environment path
        engine = self.engine
        if engine:
            engine_host = engine.host
            if not engine_host or engine_host.type != SystemType.Docker:
                systems = self.systems
                if systems:
                    _validate_python_environment_path(systems)
        return self

    @field_validator("data_configs", mode="before")
    @classmethod
    def validate_data_config_names(cls, v):
        if not v:
            return v

        # validate data config name is unique
        data_name_set = set()
        for data_config in v:
            data_config_obj = validate_config(data_config, DataConfig)
            if data_config_obj.name in data_name_set:
                raise ValueError(f"Data config name {data_config_obj.name} is duplicated. Please use another name.")
            data_name_set.add(data_config_obj.name)
        return v

    @field_validator("data_configs", mode="before")
    @classmethod
    def validate_data_configs_with_hf_model(cls, v, info):
        if "input_model" not in info.data:
            raise ValueError("Invalid input model")

        result = []
        for item in v:
            input_model_config = info.data["input_model"].model_dump()
            if input_model_config["type"].lower() not in ["hfmodel", "onnxmodel"]:
                result.append(item)
                continue

            if isinstance(item, DataConfig):
                item = item.model_dump()  # noqa: PLW2901  # Intentional reassignment for use throughout function

            # all model related info used for auto filling
            task = None
            model_name = None
            if input_model_config["type"].lower() == "onnxmodel" and "model_attributes" in input_model_config["config"]:
                # Only valid for onnx model converted from Olive Conversion pass
                task = input_model_config["config"]["model_attributes"].get("hf_task", DEFAULT_HF_TASK)
                model_name = input_model_config["config"]["model_attributes"].get("_name_or_path")
            else:
                task = input_model_config["config"].get("task", DEFAULT_HF_TASK)
                model_name = input_model_config["config"]["model_path"]

            model_info = {
                "model_name": model_name,
                "task": task,
                "trust_remote_code": input_model_config["config"].get("load_kwargs", {}).get("trust_remote_code"),
            }
            kv_cache = input_model_config.get("io_config", {}).get("kv_cache")
            if isinstance(kv_cache, dict):
                for key in ["ort_past_key_name", "ort_past_value_name", "batch_size"]:
                    model_info[key] = kv_cache.get(key)

            # TODO(anyone): Will this container ever be used with non-HF models?
            if item.get("type"):
                if item["type"] in TRANSFORMER_DUMMY_DATA_CONTAINER:
                    _auto_fill_data_config(
                        item,
                        model_info,
                        ["load_dataset_config"],
                        ["model_name", "ort_past_key_name", "ort_past_value_name", "batch_size"],
                    )
                elif item["type"] == HuggingfaceContainer.__name__:
                    # auto insert model_name and task from input model hf config if not present
                    # both are required for huggingface container
                    _auto_fill_data_config(
                        item,
                        model_info,
                        ["pre_process_data_config", "post_process_data_config"],
                        ["model_name", "task"],
                    )

                    # auto insert trust_remote_code from input model hf config
                    # won't override if value was set to False explicitly
                    _auto_fill_data_config(
                        item,
                        model_info,
                        ["pre_process_data_config", "load_dataset_config"],
                        ["trust_remote_code"],
                        only_none=True,
                    )

            result.append(validate_config(item, DataConfig))
        return result

    @field_validator("evaluators", mode="before")
    @classmethod
    def validate_evaluators(cls, v, info):
        if not v:
            return {}

        for name, config in v.items():
            for idx, metric in enumerate(config.get("metrics", [])):
                v[name]["metrics"][idx] = _resolve_data_config(metric, info.data, "data_config")
        return v

    @field_validator("engine", mode="before")
    @classmethod
    def validate_engine(cls, v, info):
        v = _resolve_system(v, info.data, "host")
        v = _resolve_system(v, info.data, "target")
        if v.get("search_strategy") and not v.get("evaluator"):
            raise ValueError(
                "Can't search without a valid evaluator config. "
                "Either provider a valid evaluator config or disable search."
            )

        return _resolve_evaluator(v, info.data)

    @field_validator("passes", mode="before")
    @classmethod
    def validate_pass_host_evaluator(cls, v, info):
        # passes is a dict[str, list[RunPassConfig]]
        for pass_configs in v.values():  # Only need values, not keys
            for i, _ in enumerate(pass_configs):
                pass_configs[i] = _resolve_system(pass_configs[i], info.data, "host")
                pass_configs[i] = _resolve_evaluator(pass_configs[i], info.data)
        return v

    @field_validator("passes", mode="before")
    @classmethod
    def validate_pass_search(cls, v, info):
        if "engine" not in info.data:
            raise ValueError("Invalid engine")

        # passes is a dict[str, list[RunPassConfig]]
        for pass_configs in v.values():  # Only need values, not keys
            for i, _ in enumerate(pass_configs):
                # validate first to gather config params
                pass_configs[i] = iv = validate_config(pass_configs[i], RunPassConfig).model_dump()

                if iv.get("config"):
                    _resolve_all_data_configs(iv["config"], info.data)

                    searchable_configs = set()
                    for param_name in iv["config"]:
                        if iv["config"][param_name] == PassParamDefault.SEARCHABLE_VALUES:
                            searchable_configs.add(param_name)

                    if not info.data["engine"].search_strategy and searchable_configs:
                        raise ValueError(
                            f"You cannot disable search for {iv['type']} and"
                            f" set {searchable_configs} to SEARCHABLE_VALUES at the same time."
                            " Please remove SEARCHABLE_VALUES or enable search (needs search strategy configs)."
                        )
        return v


def _validate_python_environment_path(systems):
    for system_config in systems.values():
        if system_config.type != SystemType.PythonEnvironment:
            continue

        python_environment_path = system_config.config.python_environment_path
        if python_environment_path is None:
            raise ValueError("python_environment_path is required for PythonEnvironmentSystem native mode")

        # check if the path exists
        if not Path(python_environment_path).exists():
            raise ValueError(f"Python path {python_environment_path} does not exist")

        # check if python exists in the path
        python_path = shutil.which("python", path=python_environment_path)
        if not python_path:
            raise ValueError(f"Python executable not found in the path {python_environment_path}")


def _resolve_all_data_configs(config, values):
    """Recursively traverse the config dictionary to resolve all 'data_config' keys."""
    if isinstance(config, dict):
        for param_name, param_value in config.items():
            if param_name.endswith("data_config"):
                _resolve_data_config(config, values, param_name)
            else:
                _resolve_all_data_configs(param_value, values)
    elif isinstance(config, list):
        for element in config:
            _resolve_all_data_configs(element, values)


def _resolve_config_str(v, values, alias, component_name):
    """Resolve string value for alias in v to corresponding component config in values.

    values: {
        ...
        component_name: {
            ...
            component_name_1: component_config_1,
            ...
        }
        ...
    }

    v: {
        ...
        alias: component_name_1
        ...
    }
    -> {
        ...
        alias: component_config_1
        ...
    }
    """
    if not isinstance(v, dict):
        # if not a dict, return the original value
        return v

    # get name of sub component
    sub_component = v.get(alias)
    if not isinstance(sub_component, str):
        return v

    component_config = _resolve_config(values, sub_component, component_name)

    v[alias] = component_config
    return v


def _resolve_config(values, sub_component, component_name="systems"):
    # resolve component name to component configs
    if component_name not in values:
        raise ValueError(f"Invalid {component_name}")

    components = values[component_name] or {}
    # resolve sub component name to component config
    if sub_component not in components:
        raise ValueError(f"{sub_component} not found in {components}")
    return components[sub_component]


def _resolve_system(v, values, system_alias):
    return _resolve_config_str(v, values, system_alias, component_name="systems")


def _resolve_data_config(v, values, data_config_alias):
    if isinstance(data_config_alias, (dict, DataConfig)):
        raise ValueError(
            "Inline data configs are not supported. Define the config under 'data_configs' and use its name here."
        )

    return _resolve_config_str(
        v,
        {"data_configs_dict": {dc.name: dc for dc in values.get("data_configs") or []}},
        data_config_alias,
        component_name="data_configs_dict",
    )


def _resolve_evaluator(v, values):
    if not isinstance(v, dict):
        return v

    evaluator = v.get("evaluator")
    if isinstance(evaluator, dict):
        for idx, metric in enumerate(evaluator.get("metrics", [])):
            evaluator["metrics"][idx] = _resolve_data_config(metric, values, "data_config")
        return v

    return _resolve_config_str(v, values, "evaluator", component_name="evaluators")


def _auto_fill_data_config(config, info, config_names, param_names, only_none=False):
    """Auto fill data config with model info.

    :param config: data config
    :param info: model info
    :param config_names: list of config names to fill
    :param param_names: list of param names to fill in each config
    :param only_none: only fill if the value is None, otherwise fill if the value is falsy
    """
    for component_config_name in config_names:
        # validate the component config first to gather the config params
        config[component_config_name] = component_config = validate_config(
            config.get(component_config_name) or {}, DataComponentConfig
        ).model_dump()
        component_config["params"] = component_config_params = component_config.get("params") or {}

        for key in param_names:
            if info.get(key) is None:
                continue

            if (only_none and component_config_params.get(key) is None) or (
                not only_none and not component_config_params.get(key)
            ):
                component_config_params[key] = info[key]
