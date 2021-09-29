# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# __init__.py

from .megatron_modifier import MegatronLMModifier
from .ds_modifier import DeepSpeedZeROModifier

LEAGCY_MEGATRON_LM_OPTIMIZER_NAME = "megatron.fp16.fp16.FP16_Optimizer"
DEEPSPEED_ZERO1_AND_ZERO2_OPTIMIZER_NAME = "deepspeed.runtime.zero.stage2.FP16_DeepSpeedZeroOptimizer"

OptimizerModifierTypeRegistry = {
    LEAGCY_MEGATRON_LM_OPTIMIZER_NAME: MegatronLMModifier,
    DEEPSPEED_ZERO1_AND_ZERO2_OPTIMIZER_NAME : DeepSpeedZeROModifier,
}

def get_full_qualified_type_name(o):
    klass = o.__class__
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__
    return module + '.' + klass.__qualname__

def FP16_Optimizer(optimizer, **kwargs):
    """
    Simple wrapper to replace inefficient FP16_Optimizer function calls implemented by library for example
        Apex, DeepSpeed, Megatron-LM.

    Args:
        optimizer: the FP16_Optimizer instance

    Returns:
        The modified FP16_Optimizer instance

    """
    optimizer_full_qualified_name = get_full_qualified_type_name(optimizer)
    if optimizer_full_qualified_name not in OptimizerModifierTypeRegistry:
        return optimizer

    modifier = OptimizerModifierTypeRegistry[optimizer_full_qualified_name](optimizer, **kwargs)
    modifier.apply()

    return optimizer
