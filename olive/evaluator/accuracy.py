# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from abc import abstractmethod
from inspect import isfunction, signature
from typing import Any, Callable, ClassVar, Optional, Union

import torch
import torchmetrics

from olive.common.auto_config import AutoConfigClass, ConfigBase
from olive.common.config_utils import ConfigParam
from olive.data.constants import IGNORE_INDEX

logger = logging.getLogger(__name__)


class AccuracyBase(AutoConfigClass):
    registry: ClassVar[dict[str, type["AccuracyBase"]]] = {}
    metric_cls_map: ClassVar[dict[str, Union[torchmetrics.Metric, Callable]]] = {
        "accuracy_score": torchmetrics.Accuracy,
        "f1_score": torchmetrics.F1Score,
        "precision": torchmetrics.Precision,
        "recall": torchmetrics.Recall,
        "auroc": torchmetrics.AUROC,
        "perplexity": torchmetrics.text.perplexity.Perplexity,
    }

    def __init__(self, config: Optional[Union[ConfigBase, dict[str, Any]]] = None) -> None:
        super().__init__(config or {})
        self.resolve_kwargs()

    def resolve_kwargs(self):
        config_dict = self.config.model_dump()
        kwargs = config_dict.pop("kwargs", {})
        config_dict.update(kwargs or {})
        self.config_dict = config_dict

    @classmethod
    def _metric_config_from_torch_metrics(cls):
        metric_module = cls.metric_cls_map[cls.name]
        params = signature(metric_module).parameters
        # if the metrics is calculated by torchmetrics.functional, we should filter the label data out
        ignore_idx = 0
        if isfunction(metric_module):
            ignore_idx = 2
            logger.debug("Will ignore the first two params of torchmetrics.functional.")
        metric_config = {}
        for param, info in params.items():
            if ignore_idx > 0:
                ignore_idx -= 1
                continue
            annotation = info.annotation if info.annotation != info.empty else None
            default_value, required = (info.default, False) if info.default != info.empty else (None, True)
            if info.kind in (info.VAR_KEYWORD, info.VAR_POSITIONAL):
                required = False
            metric_config[param] = ConfigParam(type_=annotation, required=required, default_value=default_value)
        if "task" in metric_config:
            metric_config["task"].default_value = "binary"
            metric_config["task"].required = False
        return metric_config

    @classmethod
    def _default_config(cls) -> dict[str, ConfigParam]:
        return cls._metric_config_from_torch_metrics()

    @staticmethod
    def prepare_tensors(preds, target, dtypes: Union[torch.dtype, list[torch.dtype], tuple[torch.dtype]] = torch.int):
        dtypes = dtypes if isinstance(dtypes, (list, tuple)) else [dtypes, dtypes]
        assert len(dtypes) == 2, "dtypes should be a list or tuple with two elements."
        preds = torch.tensor(preds, dtype=dtypes[0]) if not isinstance(preds, torch.Tensor) else preds.to(dtypes[0])
        target = torch.tensor(target, dtype=dtypes[1]) if not isinstance(target, torch.Tensor) else target.to(dtypes[1])
        return preds, target

    @abstractmethod
    def measure(self, model_output, target):
        raise NotImplementedError


class AccuracyScore(AccuracyBase):
    name: Optional[str] = "accuracy_score"

    def measure(self, model_output, target):
        preds_tensor, target_tensor = self.prepare_tensors(model_output.preds, target)
        accuracy = torchmetrics.Accuracy(**self.config_dict)
        result = accuracy(preds_tensor, target_tensor)
        return result.item()


class F1Score(AccuracyBase):
    name: Optional[str] = "f1_score"

    def measure(self, model_output, target):
        preds_tensor, target_tensor = self.prepare_tensors(model_output.preds, target)
        f1 = torchmetrics.F1Score(**self.config_dict)
        result = f1(preds_tensor, target_tensor)
        return result.item()


class Precision(AccuracyBase):
    name: Optional[str] = "precision"

    def measure(self, model_output, target):
        preds_tensor, target_tensor = self.prepare_tensors(model_output.preds, target)
        precision = torchmetrics.Precision(**self.config_dict)
        result = precision(preds_tensor, target_tensor)
        return result.item()


class Recall(AccuracyBase):
    name: Optional[str] = "recall"

    def measure(self, model_output, target):
        preds_tensor, target_tensor = self.prepare_tensors(model_output.preds, target)
        recall = torchmetrics.Recall(**self.config_dict)
        result = recall(preds_tensor, target_tensor)
        return result.item()


class AUROC(AccuracyBase):
    name: Optional[str] = "auroc"

    def measure(self, model_output, target):
        logits_tensor, target_tensor = self.prepare_tensors(model_output.logits, target, [torch.float, torch.int32])
        if self.config_dict.get("task") == "binary" and len(logits_tensor.shape) > 1 and logits_tensor.shape[-1] == 2:
            logits_tensor = torch.softmax(logits_tensor, dim=-1)[:, 1]
        auroc = torchmetrics.AUROC(**self.config_dict)
        target_tensor = target_tensor.flatten()
        result = auroc(logits_tensor, target_tensor)
        return result.item()


class Perplexity(AccuracyBase):
    name: Optional[str] = "perplexity"

    def measure(self, model_output, target):
        # update ignore_index if not set
        config = self.config_dict
        if config["ignore_index"] is None:
            config["ignore_index"] = IGNORE_INDEX

        # create perplexity metric
        perplexity = torchmetrics.text.perplexity.Perplexity(**config)

        # loop through samples
        # the logits are large matrix, so converting all to tensors at once is slow
        num_samples = len(model_output.preds)
        for i in range(num_samples):
            logits, targets = self.prepare_tensors(model_output.preds[i], target[i], dtypes=[torch.float, torch.long])
            logits = logits.unsqueeze(0)
            targets = targets.unsqueeze(0)
            # shift targets to the right by one, and drop the last token of logits
            logits = logits[..., :-1, :]
            targets = targets[..., 1:]
            perplexity.update(logits, targets)
        result = perplexity.compute()
        return result.item()
