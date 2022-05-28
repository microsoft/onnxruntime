# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from ._ds_modifier import DeepSpeedZeROModifier
from ._megatron_modifier import LegacyMegatronLMModifier
from ._apex_amp_modifier import ApexAMPModifier

LEAGCY_MEGATRON_LM_OPTIMIZER_NAME = "megatron.fp16.fp16.FP16_Optimizer"
DEEPSPEED_ZERO1_AND_ZERO2_OPTIMIZER_NAME = "deepspeed.runtime.zero.stage2.FP16_DeepSpeedZeroOptimizer"
APEX_AMP_OPTIMIZER_NAME = "apex.amp.optimizer.unique_name_as_id"

OptimizerModifierTypeRegistry = {
    LEAGCY_MEGATRON_LM_OPTIMIZER_NAME: LegacyMegatronLMModifier,
    DEEPSPEED_ZERO1_AND_ZERO2_OPTIMIZER_NAME: DeepSpeedZeROModifier,
    APEX_AMP_OPTIMIZER_NAME: ApexAMPModifier,
}
