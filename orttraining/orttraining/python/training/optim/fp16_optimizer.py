# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from ._modifier_registry import OptimizerModifierTypeRegistry

def FP16_Optimizer(optimizer, **kwargs):
    """
    Simple wrapper to replace inefficient FP16_Optimizer function calls implemented by libraries for example
        Apex, DeepSpeed, Megatron-LM.

    Usage:
        1. DeepSpeed ZeRO Optimizer Override：

        >>> from onnxruntime.training.optim.fp16_optimizer import FP16_Optimizer
        >>> optimizer = Adam(param_groups,
        >>>                  lr=args.lr,
        >>>                  weight_decay=args.weight_decay,
        >>>                  betas=(args.adam_beta1, args.adam_beta2),
        >>>                  eps=args.adam_eps)

        >>> model, optimizer, _, lr_scheduler = deepspeed.initialize(
        >>>     model=model,
        >>>     optimizer=optimizer,
        >>>     args=args,
        >>>     lr_scheduler=lr_scheduler,
        >>>     mpu=mpu,
        >>>     dist_init_required=False)
        >>> if args.fp16:
        >>>     optimizer = FP16_Optimizer(optimizer)

        2. Megatron-LM-v1.1.5 Optimizer Override:

        >>> from onnxruntime.training.optim.fp16_optimizer import FP16_Optimizer as ORT_FP16_Optimizer
        >>> optimizer = Adam(param_groups,
        >>>                  lr=args.lr,
        >>>                  weight_decay=args.weight_decay,
        >>>                  betas=(args.adam_beta1, args.adam_beta2),
        >>>                  eps=args.adam_eps)

        >>> # Wrap into fp16 optimizer.
        >>> if args.fp16:
        >>>     optimizer = FP16_Optimizer(optimizer,
        >>>                                static_loss_scale=args.loss_scale,
        >>>                                dynamic_loss_scale=args.dynamic_loss_scale,
        >>>                                dynamic_loss_args={
        >>>                                     'scale_window': args.loss_scale_window,
        >>>                                     'min_scale': args.min_scale,
        >>>                                     'delayed_shift': args.hysteresis},
        >>>                                verbose=True)
        >>>     optimizer = ORT_FP16_Optimizer(optimizer,
        >>>                                    get_tensor_model_parallel_rank=mpu.get_model_parallel_rank,
        >>>                                    get_tensor_model_parallel_group=mpu.get_model_parallel_group)

    Args:
        optimizer: the FP16_Optimizer instance

    Returns:
        The modified FP16_Optimizer instance

    """
    def get_full_qualified_type_name(o):
        klass = o.__class__
        module = klass.__module__
        if module == 'builtins':
            return klass.__qualname__
        return module + '.' + klass.__qualname__

    optimizer_full_qualified_name = get_full_qualified_type_name(optimizer)
    if optimizer_full_qualified_name not in OptimizerModifierTypeRegistry:
        return optimizer

    modifier = OptimizerModifierTypeRegistry[optimizer_full_qualified_name](optimizer, **kwargs)
    modifier.apply()

    return optimizer
