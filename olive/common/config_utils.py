# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import inspect
import json
import logging
from functools import partial
from pathlib import Path
from types import FunctionType, MethodType
from typing import Any, Callable, ClassVar, Optional, TypeVar, Union

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    RootModel,
    create_model,
    field_validator,
    model_validator,
)

from olive.common.utils import StrEnumBase, hash_function, hash_object

logger = logging.getLogger(__name__)


def serialize_function(function: Union[FunctionType, MethodType]) -> dict:
    """Serialize a function into a dictionary."""
    return {
        "olive_parameter_type": "Function",
        "name": function.__name__,
        "signature": str(inspect.signature(function)),
        "sourcecode_hash": hash_function(function),
    }


def serialize_object(obj: Any) -> dict:
    """Serialize an object into a dictionary."""
    return {
        "olive_parameter_type": "Object",
        "type": type(obj).__name__,
        "hash": hash_object(obj),
    }


def _expanded_default(custom_default: Callable[[Any], Any], make_absolute: bool, obj: Any) -> Any:
    if custom_default is not None:
        try:
            return custom_default(obj)
        except TypeError:
            pass
    if isinstance(obj, (FunctionType, MethodType)):
        return serialize_function(obj)
    if isinstance(obj, Path):
        return str(obj.resolve()) if make_absolute else str(obj)
    if hasattr(obj, "to_json"):
        return obj.to_json()
    return serialize_object(obj)


def config_json_dumps(
    obj: Any, default: Optional[Callable[[Any], Any]] = None, make_absolute: bool = True, **kwargs
) -> str:
    """Serialize a Python object into a JSON string. Also serializes functions and objects."""
    default = partial(_expanded_default, default, make_absolute)
    return json.dumps(obj, default=default, **kwargs)


def _expanded_object_hook(custom_object_hook: Callable[[dict], Any], obj: dict) -> Any:
    if obj.get("olive_parameter_type") in ["Function", "Object"]:
        param_type = obj.get("type", obj.get("olive_parameter_type"))
        raise ValueError(
            f"Cannot load a {param_type} from JSON. Replace {param_type} with user_script and name string."
        )
    if custom_object_hook is None:
        return obj
    return custom_object_hook(obj)


def config_json_loads(
    s: Union[str, bytes, bytearray], *, object_hook: Optional[Callable[[dict], Any]] = None, **kwargs
) -> Any:
    """Deserialize a JSON string into a Python object."""
    object_hook = partial(_expanded_object_hook, object_hook)
    return json.loads(s, object_hook=object_hook, **kwargs)


def serialize_to_json(obj: Any, check_object: bool = False, make_absolute: bool = True) -> dict:
    """Serialize a Python object into a JSON dict. Also serializes functions and objects."""
    if isinstance(obj, BaseModel):
        # In pydantic v2, use model_dump() and then json.dumps with custom encoder
        raw_json = config_json_dumps(obj.model_dump(), make_absolute=make_absolute)
    else:
        raw_json = config_json_dumps(obj, make_absolute=make_absolute)
    if check_object:
        try:
            config_json_loads(raw_json)
        except ValueError as e:
            e = str(e)
            if "user_script" in e:
                e = e.replace("Cannot load", "Cannot serialize")
                e = e.replace("from JSON", "to JSON")
            raise ValueError(e) from None
    return json.loads(raw_json)


def load_config_file(file_path: Union[str, Path]) -> dict:
    """Load a file into a dictionary."""
    file_path = Path(file_path)
    if file_path.suffix in {".yaml", ".yml"}:
        with open(file_path) as f:
            return yaml.safe_load(f)
    elif file_path.suffix == ".json":
        with open(file_path) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")


class ConfigBase(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def to_json(self, check_object: bool = False, make_absolute: bool = True) -> dict:
        return serialize_to_json(self, check_object, make_absolute)

    @classmethod
    def from_json(cls, json_dict: dict) -> "ConfigBase":
        return cls.model_validate_json(json.dumps(json_dict))

    @classmethod
    def parse_file_or_obj(cls, file_or_obj: Union[str, Path, dict]) -> "ConfigBase":
        """Parse a file or a dictionary object into a ConfigBase object.

        :param file_or_obj: File path or dictionary object.
            File can be a YAML file with .yaml or .yml extension or a JSON file with .json extension.
        :return: ConfigBase object.
        """
        if isinstance(file_or_obj, dict):
            obj = file_or_obj
        else:
            obj = load_config_file(file_or_obj)

        return cls.model_validate(obj)


class ConfigListBase(RootModel):
    root: list[Any]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]

    def __len__(self):
        return len(self.root)

    def to_json(self, check_object: bool = False, make_absolute: bool = True) -> dict:
        return serialize_to_json(self, check_object, make_absolute)

    @classmethod
    def from_json(cls, json_dict: dict) -> "ConfigListBase":
        return cls.model_validate_json(json.dumps(json_dict))


class ConfigDictBase(RootModel):
    root: dict[str, Any]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __iter__(self):
        return iter(self.root)

    def keys(self):
        return self.root.keys()

    def values(self):
        return self.root.values()

    def items(self):
        return self.root.items()

    def __getitem__(self, item):
        return self.root[item]

    def __len__(self):
        return len(self.root) if self.root else 0

    def to_json(self, check_object: bool = False, make_absolute: bool = True) -> dict:
        return serialize_to_json(self, check_object, make_absolute)

    @classmethod
    def from_json(cls, json_dict: dict) -> "ConfigDictBase":
        return cls.model_validate_json(json.dumps(json_dict))


class NestedConfig(ConfigBase):
    """Config class that automatically gathers all values.

    The values are defined in the class fields and inserted into a dict Field called described by `_nested_field_name`.
    Generally used for configs that have a nested structure like:
        {
            "type": "object",
            "config": {
                "key1": "value1",
                "key2": "value2"
            }
        }
    Must ensure that there are no fields inside the `_nested_field_name` dict/class that are also defined as fields in
    this class. The fields of this class take precedence over the fields in the nested class.
    """

    _nested_field_name: ClassVar[str] = "config"

    @model_validator(mode="before")
    @classmethod
    def gather_nested_field(cls, values):
        if values is None:
            # In pydantic v2, values can be None when no arguments are provided
            values = {}

        if not isinstance(values, dict):
            # Accept the value as is if 'values' is non-dict type (like bool, int, etc.).
            return values

        all_fields = set(cls.model_fields.keys())
        for field in cls.model_fields.values():
            if field.alias:
                all_fields.add(field.alias)
        if cls._nested_field_name not in all_fields:
            logger.debug(
                "TypedConfig is used but is missing the nested field name '%s'. Ignoring root validator",
                cls._nested_field_name,
            )
            return values

        other_fields = all_fields - {cls._nested_field_name}

        nested_field = values.pop(cls._nested_field_name, {}) or {}
        if isinstance(nested_field, ConfigBase):
            nested_field = nested_field.model_dump()

        for name in list(values):
            if name in other_fields:
                continue
            if name in nested_field:
                logger.warning("field '%s' is already defined in '%s'. Ignoring.", name, cls._nested_field_name)
            else:
                nested_field[name] = values.pop(name)

        field_info = cls.model_fields.get(cls._nested_field_name)
        if nested_field or (field_info and field_info.is_required()):
            values[cls._nested_field_name] = nested_field
        return values


class CaseInsensitiveEnum(StrEnumBase):
    """StrEnum class that is insensitive to the case of the input string.

    Note: Only insensitive when creating the enum object like `CaseInsensitiveEnum("value")`.
    The values of the enum are still case-sensitive.
    """

    @classmethod
    def _missing_(cls, value):
        value = value.lower()
        for member in cls:
            if member.lower() == value:
                return member
        return None


# TODO(jambayk): remove ParamCategory once validate object is removed or updated
class ParamCategory(CaseInsensitiveEnum):
    NONE = "none"
    OBJECT = "object"
    PATH = "path"
    DATA = "data"


class ConfigParam(ConfigBase):
    """Dataclass for pass configuration parameters."""

    type_: Any
    required: bool = False
    default_value: Any = None
    category: ParamCategory = ParamCategory.NONE
    description: Optional[str] = None

    def __repr__(self):
        repr_list = []
        booleans = ["required"]
        for k, v in self.__dict__.items():
            if k in booleans:
                if v:
                    repr_list.append(f"{k}={v}")
            elif v is not None:
                repr_list.append(f"{k}={v}")
        return f"({', '.join(repr_list)})"


# validator for enum params
def validate_enum(enum_class: type, value: str):
    try:
        value = enum_class(value)
    except ValueError:
        raise ValueError(f"Invalid value '{value}'. Valid values are {[e.value for e in enum_class]}") from None
    return value


# validator for object params. This ensures user_script is not None if value v is string
def validate_object(cls, v, info):
    if "user_script" not in info.data:
        raise ValueError("Invalid user_script")
    if isinstance(v, str) and info.data["user_script"] is None:
        raise ValueError(f"user_script must be provided if {info.field_name} is a name string")
    return v


# validator that always converts string to lowercase
def validate_lowercase(v):
    if isinstance(v, str):
        return v.lower()
    return v


def create_config_class(
    class_name: str,
    default_config: dict[str, ConfigParam],
    base: type = ConfigBase,
    validators: Optional[dict[str, Callable]] = None,
) -> type[ConfigBase]:
    """Create a Pydantic model class from a configuration dictionary."""
    config = {}
    validators = validators.copy() if validators else {}

    for param, param_config in default_config.items():
        if param_config.category == ParamCategory.OBJECT:
            validator_name = f"validate_{param}_object"
            count = 0
            while validator_name in validators:
                validator_name = f"{validator_name}_{count}"
                count += 1

            validators[validator_name] = field_validator(param, mode="before")(validate_object)

        type_ = param_config.type_
        if param_config.required:
            config[param] = (type_, ...)
            continue

        config[param] = (Optional[type_], param_config.default_value)

    return create_model(class_name, **config, __base__=base, __validators__=validators)


T = TypeVar("T", bound=ConfigBase)


def validate_config(
    config: Union[dict[str, Any], T, None],
    instance_class: type[T],
    warn_unused_keys: bool = True,
) -> T:
    """Validate a config dictionary or object against a base class and instance class.

    instance class is a subclass of base class.
    """
    config = config or {}

    if isinstance(config, dict):
        user_keys = set(config.keys())
        config = instance_class(**config)
        config_dict = config.model_dump()
        config_keys = set(config_dict.keys())
        unused_keys = user_keys - config_keys
        if isinstance(config, NestedConfig):
            unused_keys -= set((config_dict.get(config._nested_field_name) or {}).keys())  # pylint: disable=W0212
        if unused_keys and warn_unused_keys:
            logger.warning("Keys %s are not part of %s. Ignoring them.", unused_keys, instance_class.__name__)
    # for dynamically created class by Pydantic create_model, the classes are different even if the class names are same
    elif (
        isinstance(config, ConfigBase)
        and config.__class__.__module__ == instance_class.__module__
        and config.__class__.__name__ == instance_class.__name__
    ):
        pass
    else:
        raise ValueError(
            f"Invalid config class. Expected {instance_class.__name__} but got {config.__class__.__name__}"
        )
    return config


def convert_configs_to_dicts(config: Any) -> Any:
    """Convert all ConfigBase objects to dictionaries."""
    if isinstance(config, (ConfigBase, RootModel)):
        return config.model_dump()
    if isinstance(config, dict):
        return {k: convert_configs_to_dicts(v) for k, v in config.items()}
    if isinstance(config, list):
        return [convert_configs_to_dicts(v) for v in config]
    return config


def get_the_flattened_and_tree_spec(
    dynamic_shapes: Union[dict[str, Any], list[Any]], leaf_is_str: bool = False
) -> tuple[list[Any], Any]:
    """Flattens a pytree into a list of values and a TreeSpec that can be used to reconstruct the pytree."""
    # More info: https://github.com/pytorch/pytorch/blob/48203bec636692e1a9140fe7f23ba1323b19550d/torch/utils/_pytree.py#L985
    from torch.utils import _pytree

    def is_axes_with_str_key(x) -> bool:
        # axes can be either a dict or a list/tuple
        # dict: {str: str}
        # list/tuple: [str]
        return (
            isinstance(x, dict)
            and all(isinstance(k, str) and (v is None or isinstance(v, (str, int))) for k, v in x.items())
        ) or (isinstance(x, (list, tuple)) and all(v is None or isinstance(v, (str, int)) for v in x))

    def is_axes_with_int_key(x) -> bool:
        # axes can be either a dict or a list/tuple
        # dict: {int: str}
        # list/tuple: [str]
        return (
            isinstance(x, dict)
            and all(isinstance(k, int) and (v is None or isinstance(v, (str, int))) for k, v in x.items())
        ) or (isinstance(x, (list, tuple)) and all(v is None or isinstance(v, (str, int)) for v in x))

    return _pytree.tree_flatten(dynamic_shapes, is_leaf=is_axes_with_str_key if leaf_is_str else is_axes_with_int_key)
