# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""This file is used to generate test data for MemoryOptimizer tests in
    onnxruntime/test/optimizer/memory_optimizer_test.cc.

    The libs used to generate 3 layer bloom model.

    optimum: f6adbef5c4a6bd16a17e3b22712028ed5ae3709b
    huggingface: 4.34.1
    deepspeed: 0.11.1
    PyTorch: 2.1.0.dev20230803+cu118

    Change below line in optimum/onnxruntime/trainer.py
    "model = ORTModule(self.model)"
    to
    "model = ORTModule(self.model, DebugOptions(save_onnx=True, log_level=LogLevel.WARNING, onnx_prefix="3layer_bloom"))"

    Add below in examples/onnxruntime/training/language-modeling/run_clm.py before the config is used to load the model.
    "config.num_hidden_layers = 3"

    Run below command to generate the model, there will be 3layer_bloom_optimized_training.onnx generated.
    #!/bin/bash
    ds_config=`mktemp --suffix ".json"`
    echo the deepspeed config is put at $ds_config
    cat << EOF > $ds_config
    {
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "zero_optimization": {
        "stage": 1,
        "allgather_partitions": true,
        "allgather_bucket_size": 200000000,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 200000000,
        "contiguous_gradients": false,
        "cpu_offload": false,
        "memory_efficient_linear": true
    },
    "zero_allow_untested_optimizer": true,
    "optimizer": {
        "type": "AdamW",
        "params": {
        "lr": "auto",
        "betas": "auto",
        "eps": "auto",
        "weight_decay": "auto"
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
        "warmup_min_lr": "auto",
        "warmup_max_lr": "auto",
        "warmup_num_steps": "auto"
        }
    },
    "steps_per_print": 2000,
    "train_micro_batch_size_per_gpu": "auto"
    }
    EOF

    num_gpus=1
    export ORTMODULE_ENABLE_CUSTOM_AUTOGRAD=0 # GELU PythonOp will be used if this is set to 1
    torchrun --nproc_per_node $num_gpus \
    examples/onnxruntime/training/language-modeling/run_clm.py \
        --model_name_or_path bigscience/bloom-560m \
        --dataset_name wikitext \
        --dataset_config_name wikitext-2-raw-v1 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 1 \
        --do_train \
        --output_dir /tmp/test-clm --overwrite_output_dir \
        --fp16 \
        --report_to none \
        --max_steps 10000 --logging_steps 1 --use_module_with_loss \
        --deepspeed $ds_config
   """
