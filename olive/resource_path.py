# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import shutil
import tempfile
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, ClassVar, Optional, Union

from pydantic import Field, field_validator

from olive.common.auto_config import AutoConfigClass
from olive.common.config_utils import (
    CaseInsensitiveEnum,
    ConfigBase,
    ConfigParam,
    NestedConfig,
    serialize_to_json,
    validate_config,
)
from olive.common.utils import copy_dir, get_credentials, retry_func

logger = logging.getLogger(__name__)


class ResourceType(CaseInsensitiveEnum):
    LocalFile = "file"
    LocalFolder = "folder"
    StringName = "string_name"
    AzureMLRegistryModel = "azureml_registry_model"


LOCAL_RESOURCE_TYPES = (ResourceType.LocalFile, ResourceType.LocalFolder)


class ResourcePath(AutoConfigClass):
    registry: ClassVar[dict[str, type["ResourcePath"]]] = {}
    name: Optional[ResourceType] = None

    def __repr__(self) -> str:
        return self.get_path()

    @property
    def type(self) -> Optional[ResourceType]:
        return self.name

    @abstractmethod
    def get_path(self) -> str:
        """Return the resource path as a string."""
        raise NotImplementedError

    @abstractmethod
    def save_to_dir(self, dir_path: Union[Path, str], name: Optional[str] = None, overwrite: bool = False) -> str:
        """Save the resource to a directory."""
        raise NotImplementedError

    def is_local_resource(self) -> bool:
        """Return True if the resource is a local resource."""
        return self.type in LOCAL_RESOURCE_TYPES

    def is_azureml_resource(self) -> bool:
        """Return True if the resource is an AzureML resource."""
        return self.type == ResourceType.AzureMLRegistryModel

    def is_string_name(self) -> bool:
        """Return True if the resource is a string name."""
        return self.type == ResourceType.StringName

    def is_local_resource_or_string_name(self) -> bool:
        """Return True if the resource is a local resource or a string name."""
        return self.is_local_resource() or self.is_string_name()

    def to_json(self):
        json_data = {"type": self.type, "config": self.config.to_json()}
        return serialize_to_json(json_data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResourcePath):
            return False
        return self.type == other.type and self.config.to_json() == other.config.to_json()

    def __hash__(self) -> int:
        return hash((self.config.to_json(), self.type))


class ResourcePathConfig(NestedConfig):
    type: ResourceType = Field(..., description="Type of the resource.")
    config: ConfigBase = Field(..., description="Config of the resource.")

    @field_validator("config", mode="before")
    @classmethod
    def validate_config(cls, v, info):
        if "type" not in info.data:
            raise ValueError("Invalid type.")

        config_class = ResourcePath.registry[info.data["type"]].get_config_class()
        return validate_config(v, config_class)

    def create_resource_path(self) -> ResourcePath:
        return ResourcePath.registry[self.type](self.config)


VALID_RESOURCE_CONFIGS = (str, Path, dict, ResourcePathConfig, ResourcePath)
OLIVE_RESOURCE_ANNOTATIONS = Optional[Union[str, Path, dict, ResourcePathConfig, ResourcePath]]


def create_resource_path(
    resource_path: OLIVE_RESOURCE_ANNOTATIONS,
) -> Optional[ResourcePath]:
    """Create a resource path from a string or a dict.

    If a string or Path is provided, it is inferred to be a file, folder, or string name.
    If a dict is provided, it must have "type" and "config" fields. The "type" field must be one of the
    values in the ResourceType enum. The "config" field must be a dict that can be used to create a resource
    config of the specified type.

    :param resource_path:
    :return: A resource path.
    """
    if resource_path is None:
        return None
    if isinstance(resource_path, ResourcePath):
        return resource_path
    if isinstance(resource_path, (ResourcePathConfig, dict)):
        resource_path_config = validate_config(resource_path, ResourcePathConfig)
        return resource_path_config.create_resource_path()

    # check if the resource path is a file, folder, azureml datastore, or a string name
    resource_type: Optional[ResourceType] = None
    config_key = None
    if Path(resource_path).is_file():
        resource_type = ResourceType.LocalFile
        config_key = "path"
    elif Path(resource_path).is_dir():
        resource_type = ResourceType.LocalFolder
        config_key = "path"
    else:
        resource_type = ResourceType.StringName
        config_key = "name"

    return ResourcePathConfig(type=resource_type, config={config_key: resource_path}).create_resource_path()


def validate_resource_path(cls, v, info):
    try:
        v = create_resource_path(v)
        if v and v.is_local_resource_or_string_name():
            # might expect a string or Path when using this resource locally
            v = v.get_path()
    except ValueError as e:
        raise ValueError(f"Invalid resource path '{v}': {e}") from None
    return v


def find_all_resources(config, ignore_keys: Optional[list[str]] = None) -> dict[str, ResourcePath]:
    """Find all resources in a config.

    :param config: The config to search for resources.
    :param ignore_keys: A list of keys to ignore when searching for resources.
    :return: A dictionary of all resources found in the config.
        keys are tuples representing the path to the resource in the config and the values are
        the resource paths.
    """
    if isinstance(config, VALID_RESOURCE_CONFIGS):
        try:
            # don't want to accidentally modify the original config
            resource_path = create_resource_path(deepcopy(config))
            if resource_path.is_string_name():
                return {}
            return {(): resource_path}
        except ValueError:
            pass

    resources = {}
    if isinstance(config, (dict, list)):
        for k, v in config.items() if isinstance(config, dict) else enumerate(config):
            if ignore_keys and k in ignore_keys:
                continue
            resources.update({(k, *k2): v2 for k2, v2 in find_all_resources(v, ignore_keys=ignore_keys).items()})

    return resources


def _overwrite_helper(new_path: Union[Path, str], overwrite: bool):
    new_path = Path(new_path).resolve()

    # check if the resource already exists
    if new_path.exists():
        if not overwrite:
            # raise an error if the file/folder with same name already exists and overwrite is set to False
            # Olive doesn't know if the existing file/folder is the same as the one being saved
            # or if the user wants to overwrite the existing file/folder
            raise FileExistsError(
                f"Trying to save resource to {new_path} but a file/folder with the same name already exists and"
                " overwrite is set to False. If you want to overwrite the existing file/folder, set overwrite to True."
            )
        else:
            # delete the resource if it already exists
            if new_path.is_file():
                new_path.unlink()
            else:
                shutil.rmtree(new_path)


def _validate_path(v):
    if not Path(v).exists():
        raise ValueError(f"Path {v} does not exist.")
    return Path(v).resolve()


class LocalResourcePath(ResourcePath):
    """Base class for a local resource path."""

    @classmethod
    def _default_config(cls) -> dict[str, Any]:
        return {
            "path": ConfigParam(type_=Union[Path, str], required=True, description="Path to the resource."),
        }

    @classmethod
    def _validators(cls) -> dict[str, Callable]:
        return {"validate_path": field_validator("path", mode="before")(_validate_path)}

    def get_path(self) -> str:
        return str(self.config.path)

    def save_to_dir(
        self, dir_path: Union[Path, str], name: Optional[str] = None, overwrite: bool = False, flatten: bool = False
    ) -> str:
        # directory to save the resource to
        dir_path = Path(dir_path).resolve()
        dir_path.mkdir(parents=True, exist_ok=True)

        # path to save the resource to
        if name:
            new_path_name = Path(name).with_suffix(self.config.path.suffix).name
        else:
            new_path_name = self.config.path.name
        new_path = dir_path / new_path_name

        # is the resource a file or a folder
        is_file = Path(self.config.path).is_file()

        # Handle overwrite logic based on actual destination
        if is_file or not flatten:
            # Actual destination is new_path
            _overwrite_helper(new_path, overwrite)
        else:
            # flatten=True and is folder: destination is dir_path
            # Check for conflicts with individual files in source
            if overwrite:
                for item in Path(self.config.path).iterdir():
                    target = dir_path / item.name
                    if target.exists():
                        if target.is_file():
                            target.unlink()
                        else:
                            shutil.rmtree(target)

        # copy the resource to the new path
        if is_file:
            shutil.copy(self.config.path, new_path)
        else:
            if flatten:
                copy_dir(self.config.path, dir_path, dirs_exist_ok=True)
            else:
                copy_dir(self.config.path, new_path)

        return str(new_path)


def _validate_file_path(v):
    path = Path(v)
    if not path.is_file():
        raise ValueError(f"Path {path} is not a file.")
    return path


class LocalFile(LocalResourcePath):
    """Local file resource path."""

    name = ResourceType.LocalFile

    @classmethod
    def _validators(cls) -> dict[str, Callable[..., Any]]:
        validators = super()._validators()
        validators.update({"validate_file_path": field_validator("path", mode="before")(_validate_file_path)})
        return validators


def _validate_folder_path(v):
    path = Path(v)
    if not path.is_dir():
        raise ValueError(f"Path {path} is not a folder.")
    return path


class LocalFolder(LocalResourcePath):
    """Local folder resource path."""

    name = ResourceType.LocalFolder

    @classmethod
    def _validators(cls) -> dict[str, Callable[..., Any]]:
        validators = super()._validators()
        validators.update({"validate_folder_path": field_validator("path", mode="before")(_validate_folder_path)})
        return validators


class StringName(ResourcePath):
    """String name resource path."""

    name = ResourceType.StringName

    @classmethod
    def _default_config(cls) -> dict[str, Any]:
        return {
            "name": ConfigParam(type_=str, required=True, description="Name of the resource."),
        }

    def get_path(self) -> str:
        return self.config.name

    def save_to_dir(self, dir_path: Union[Path, str], name: Optional[str] = None, overwrite: bool = False) -> str:
        logger.debug("Resource is a string name. No need to save to directory.")
        return self.config.name


class AzureMLRegistryModel(ResourcePath):
    """AzureML Model resource path."""

    name = ResourceType.AzureMLRegistryModel

    @classmethod
    def _default_config(cls) -> dict[str, Any]:
        return {
            "registry_name": ConfigParam(
                type_=str,
                required=True,
                description=(
                    "Name of the registry. Basically, the value is parent directory name of given model in azureml"
                ),
            ),
            "name": ConfigParam(type_=str, required=True, description="Name of the model."),
            "version": ConfigParam(type_=Union[int, str], required=True, description="Version of the model."),
            "max_operation_retries": ConfigParam(
                type_=int,
                default_value=3,
                description="Max number of retries for AzureML operations like resource creation or download.",
            ),
            "operation_retry_interval": ConfigParam(
                type_=int,
                default_value=5,
                description=(
                    "Initial interval in seconds between retries for AzureML operations like resource creation or"
                    " download. The interval doubles after each retry."
                ),
            ),
        }

    def get_path(self) -> str:
        return (
            f"azureml://registries/{self.config.registry_name}/models/{self.config.name}/versions/{self.config.version}"
        )

    def save_to_dir(self, dir_path: Union[Path, str], name: Optional[str] = None, overwrite: bool = False) -> str:
        from azure.ai.ml import MLClient

        self.set_azure_logging_if_noset()

        ml_client = MLClient(credential=get_credentials(), registry_name=self.config.registry_name)

        # directory to save the resource to
        dir_path = Path(dir_path).resolve()
        dir_path.mkdir(parents=True, exist_ok=True)

        # azureml model
        model = ml_client.models.get(self.config.name, version=self.config.version)
        model_path = Path(model.path)

        # path to save the resource to
        if name:
            new_path_name = Path(name).with_suffix(model_path.suffix).name
        else:
            new_path_name = model_path.name
        new_path = dir_path / new_path_name
        _overwrite_helper(new_path, overwrite)

        # download the resource to the new path
        logger.debug("Downloading model %s version %s to %s.", self.config.name, self.config.version, new_path)
        from azure.core.exceptions import ServiceResponseError

        with tempfile.TemporaryDirectory(dir=dir_path, prefix="olive_tmp") as tempdir:
            temp_dir = Path(tempdir)
            retry_func(
                ml_client.models.download,
                [self.config.name],
                {"version": self.config.version, "download_path": temp_dir},
                max_tries=self.config.max_operation_retries,
                delay=self.config.operation_retry_interval,
                exceptions=ServiceResponseError,
            )
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(temp_dir / self.config.name / model_path.name, new_path)
        return str(new_path)

    def set_azure_logging_if_noset(self):
        # set logger level to error to avoid too many logs from azure sdk
        azure_ml_logger = logging.getLogger("azure.ai.ml")
        # only set the level if it is not set, to avoid changing the level set by the user
        if not azure_ml_logger.level:
            azure_ml_logger.setLevel(logging.ERROR)
        azure_identity_logger = logging.getLogger("azure.identity")
        # only set the level if it is not set, to avoid changing the level set by the user
        if not azure_identity_logger.level:
            azure_identity_logger.setLevel(logging.ERROR)
