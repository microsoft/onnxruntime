# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import warnings
from typing import ClassVar, Dict, Optional

from ._apex_amp_modifier import ApexAMPModifier
from ._ds_modifier import DeepSpeedZeROModifier
from ._megatron_modifier import LegacyMegatronLMModifier
from ._modifier import FP16OptimizerModifier


class _AccelerateDeepSpeedZeROModifier(DeepSpeedZeROModifier):
    """
    Modifier for wrapper of DeepSpeed Optimizer in accelerator.
    https://github.com/huggingface/accelerate/blob/7843286f2e1c50735d259fbc0084a7f1c85e00e3/src/accelerate/utils/deepspeed.py#L182C19-L182C19
    """

    def __init__(self, accelerator_optimizer, **kwargs) -> None:
        super().__init__(accelerator_optimizer.optimizer)


def get_full_qualified_type_name(o):
    klass = o.__class__
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__
    return module + "." + klass.__qualname__


class OptimizerModifierTypeRegistry:
    _MAP: ClassVar[Dict[str, FP16OptimizerModifier]] = {
        "megatron.fp16.fp16.FP16_Optimizer": LegacyMegatronLMModifier,
        "deepspeed.runtime.zero.stage2.FP16_DeepSpeedZeroOptimizer": DeepSpeedZeROModifier,
        "deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer": DeepSpeedZeROModifier,
        "apex.amp.optimizer.unique_name_as_id": ApexAMPModifier,
    }

    @staticmethod
    def create_modifier(optimizer_full_qualified_name: str, optimizer, **kwargs) -> Optional[FP16OptimizerModifier]:
        """ Create modifier for optimizer."""
        if optimizer_full_qualified_name in OptimizerModifierTypeRegistry._MAP:
            return OptimizerModifierTypeRegistry._MAP[optimizer_full_qualified_name](optimizer, **kwargs)

        if optimizer_full_qualified_name == "accelerate.utils.deepspeed.DeepSpeedOptimizerWrapper":
            if (
                hasattr(optimizer, "optimizer")
                and get_full_qualified_type_name(optimizer.optimizer) in OptimizerModifierTypeRegistry._MAP
            ):
                return _AccelerateDeepSpeedZeROModifier(optimizer, **kwargs)

        warnings.warn(
            "Skip modifying optimizer because of optimizer name not found in the registry: "
            f"{optimizer_full_qualified_name}",
            UserWarning,
        )
        return None
