# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from ._apex_amp_modifier import ApexAMPModifier
from ._ds_modifier import DeepSpeedZeROModifier
from ._megatron_modifier import LegacyMegatronLMModifier

OptimizerModifierTypeRegistry = {
    "megatron.fp16.fp16.FP16_Optimizer": LegacyMegatronLMModifier,
    "deepspeed.runtime.zero.stage2.FP16_DeepSpeedZeroOptimizer": DeepSpeedZeROModifier,
    "deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer": DeepSpeedZeROModifier,
    "apex.amp.optimizer.unique_name_as_id": ApexAMPModifier,
}
