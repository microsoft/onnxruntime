# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Any, ClassVar, Optional, Union

from pydantic import Field, field_serializer, field_validator, model_validator

from olive.common.config_utils import ConfigBase, NestedConfig, validate_config
from olive.common.utils import StrEnumBase
from olive.data.config import DataConfig
from olive.evaluator.accuracy import AccuracyBase
from olive.evaluator.metric_config import (
    LatencyMetricConfig,
    MetricGoal,
    SizeOnDiskMetricConfig,
    ThroughputMetricConfig,
    get_user_config_class,
)

logger = logging.getLogger(__name__)


class MetricType(StrEnumBase):
    # TODO(trajep): support throughput
    ACCURACY = "accuracy"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    SIZE_ON_DISK = "size_on_disk"
    CUSTOM = "custom"


class AccuracySubType(StrEnumBase):
    ACCURACY_SCORE = "accuracy_score"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"
    AUROC = "auroc"
    PERPLEXITY = "perplexity"


class LatencySubType(StrEnumBase):
    # unit: millisecond
    AVG = "avg"
    MAX = "max"
    MIN = "min"
    P50 = "p50"
    P75 = "p75"
    P90 = "p90"
    P95 = "p95"
    P99 = "p99"
    P999 = "p999"


class ThroughputSubType(StrEnumBase):
    # unit: token per second, tps
    AVG = "avg"
    MAX = "max"
    MIN = "min"
    P50 = "p50"
    P75 = "p75"
    P90 = "p90"
    P95 = "p95"
    P99 = "p99"
    P999 = "p999"


class SizeOnDiskSubType(StrEnumBase):
    BYTES = "bytes"


class SubMetric(ConfigBase):
    name: Union[AccuracySubType, LatencyMetricConfig, str]
    metric_config: Optional[ConfigBase] = None
    # -1 means no priority which will be evaluated only
    priority: int = -1
    higher_is_better: bool = False
    goal: Optional[MetricGoal] = None

    @field_validator("goal")
    @classmethod
    def validate_goal(cls, v, info):
        if v is None:
            return v
        if v.type not in ["percent-min-improvement", "percent-max-degradation"]:
            return v

        if "higher_is_better" not in info.data:
            raise ValueError("Invalid higher_is_better")
        higher_is_better = info.data["higher_is_better"]

        ranges = {
            ("percent-min-improvement", True): (0, float("inf")),
            ("percent-min-improvement", False): (0, 100),
            ("percent-max-degradation", True): (0, 100),
            ("percent-max-degradation", False): (0, float("inf")),
        }
        valid_range = ranges[(v.type, higher_is_better)]
        if not valid_range[0] < v.value < valid_range[1]:
            raise ValueError(
                f"Invalid goal value {v.value} for {v.type} and higher_is_better={higher_is_better}. Valid range is"
                f" {valid_range}"
            )
        return v

    @field_serializer("metric_config")
    def serialize_metric_config(self, metric_config):
        return metric_config.model_dump()


class Metric(NestedConfig):
    _nested_field_name: ClassVar[str] = "user_config"

    name: str
    type: MetricType
    backend: Optional[str] = Field(default="torch_metrics", validate_default=True)
    sub_types: list[SubMetric] = Field(default=[], validate_default=True)
    user_config: Optional[ConfigBase] = Field(default=None, validate_default=True)
    data_config: Optional[DataConfig] = Field(default=None, validate_default=True)

    def get_inference_settings(self, framework):
        if self.user_config is None:
            return None
        if self.user_config.inference_settings:
            return self.user_config.inference_settings.get(framework)
        else:
            return None

    def get_run_kwargs(self) -> dict[str, Any]:
        return self.user_config.run_kwargs if (self.user_config and self.user_config.run_kwargs) else {}

    def get_sub_type_info(self, info_name, no_priority_filter=True, callback=lambda x: x):
        sub_type_info = {}
        for sub_type in self.sub_types:
            if no_priority_filter and sub_type.priority <= 0:
                continue
            sub_type_info[sub_type.name] = callback(getattr(sub_type, info_name))
        return sub_type_info

    @model_validator(mode="before")
    @classmethod
    def validate_object(cls, values):
        if "type" not in values:
            raise ValueError("Invalid type")

        v = values.get("user_config") or {}
        user_config_class = get_user_config_class(values["type"])
        values["user_config"] = validate_config(v, user_config_class)
        return values

    @field_validator("backend", mode="before")
    @classmethod
    def validate_backend(cls, v, info):
        if info.data["type"] == MetricType.CUSTOM:
            return None
        from olive.evaluator.metric_backend import MetricBackend

        assert v in MetricBackend.registry, f"Backend {v} is not in {list(MetricBackend.registry.keys())}"
        assert MetricBackend.registry[v]() is not None, f"Backend {v} is not available"
        return v

    @field_validator("sub_types", mode="before")
    @classmethod
    def validate_sub_types(cls, v, info):
        if "type" not in info.data:
            raise ValueError("Invalid type")

        result = []
        for item in v:
            if info.data["type"] == MetricType.CUSTOM:
                if item.get("priority", -1) != -1 and item.get("higher_is_better", None) is None:
                    raise ValueError(f"higher_is_better must be specified for ranked custom metric: {item['name']}")
                result.append(item)
                continue

            # backend joint checking
            if info.data["backend"] == "huggingface_metrics":
                import evaluate

                try:
                    evaluate.load(item["name"])
                except FileNotFoundError as e:
                    raise ValueError(f"could not load metric {item['name']} from huggingface/evaluate") from e
            elif info.data["backend"] == "torch_metrics":
                sub_metric_type_cls = None
                if info.data["type"] == MetricType.ACCURACY:
                    sub_metric_type_cls = AccuracySubType
                elif info.data["type"] == MetricType.LATENCY:
                    sub_metric_type_cls = LatencySubType
                elif info.data["type"] == MetricType.THROUGHPUT:
                    sub_metric_type_cls = ThroughputSubType
                elif info.data["type"] == MetricType.SIZE_ON_DISK:
                    sub_metric_type_cls = SizeOnDiskSubType
                if not sub_metric_type_cls:
                    raise ValueError(f"Unrecognized metric type: {info.data['type']}") from None
                try:
                    # if not exist, will raise ValueError
                    item["name"] = sub_metric_type_cls(item["name"])
                except ValueError:
                    raise ValueError(
                        f"sub_type {item['name']} is not in {list(sub_metric_type_cls.__members__.keys())}"
                        f" for {info.data['type']} metric"
                    ) from None

            # metric_config
            metric_config_cls = None
            if info.data["type"] == MetricType.ACCURACY:
                item["higher_is_better"] = item.get("higher_is_better", True)
                if info.data["backend"] == "torch_metrics":
                    metric_config_cls = AccuracyBase.registry[item["name"]].get_config_class()
                elif info.data["backend"] == "huggingface_metrics":
                    from olive.evaluator.metric_backend import HuggingfaceMetrics

                    metric_config_cls = HuggingfaceMetrics.get_config_class()
            elif info.data["type"] == MetricType.LATENCY:
                item["higher_is_better"] = item.get("higher_is_better", False)
                metric_config_cls = LatencyMetricConfig
            elif info.data["type"] == MetricType.THROUGHPUT:
                item["higher_is_better"] = item.get("higher_is_better", True)
                metric_config_cls = ThroughputMetricConfig
            elif info.data["type"] == MetricType.SIZE_ON_DISK:
                item["higher_is_better"] = False
                metric_config_cls = SizeOnDiskMetricConfig
            if not metric_config_cls:
                raise ValueError("Unable to identify configuration class for metric.")
            item["metric_config"] = validate_config(item.get("metric_config", {}), metric_config_cls)
            result.append(item)

        return result

    @field_validator("user_config", mode="before")
    @classmethod
    def validate_user_config(cls, v, info):
        if "type" not in info.data:
            raise ValueError("Invalid type")

        user_config_class = get_user_config_class(info.data["type"])
        return validate_config(v, user_config_class)

    @field_serializer("backend")
    def serialize_backend(self, backend):
        return backend if isinstance(backend, str) else backend.model_dump()


def get_latency_config_from_metric(metric: Metric):
    warmup_num, repeat_test_num, sleep_num = None, None, None
    for sub_type in metric.sub_types:
        if sub_type.metric_config:
            warmup_num = sub_type.metric_config.warmup_num
            repeat_test_num = sub_type.metric_config.repeat_test_num
            sleep_num = sub_type.metric_config.sleep_num
            break
    return warmup_num, repeat_test_num, sleep_num
