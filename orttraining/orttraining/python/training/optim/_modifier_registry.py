# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from ._ds_modifier import DeepSpeedZeROModifier
from ._megatron_modifier import LegacyMegatronLMModifier

LEAGCY_MEGATRON_LM_OPTIMIZER_NAME = "megatron.fp16.fp16.FP16_Optimizer"
DEEPSPEED_ZERO1_AND_ZERO2_OPTIMIZER_NAME = "deepspeed.runtime.zero.stage2.FP16_DeepSpeedZeroOptimizer"

OptimizerModifierTypeRegistry = {
    LEAGCY_MEGATRON_LM_OPTIMIZER_NAME: LegacyMegatronLMModifier,
    DEEPSPEED_ZERO1_AND_ZERO2_OPTIMIZER_NAME : DeepSpeedZeROModifier,
}
