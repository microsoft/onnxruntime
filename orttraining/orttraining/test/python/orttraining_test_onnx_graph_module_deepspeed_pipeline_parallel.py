"""
Usage:
deepspeed --num_gpus 2 orttraining_test_onnx_graph_module_deepspeed_pipeline_parallel.py \
    --deepspeed_config orttraining_test_onnx_graph_module_deepspeed_pipeline_parallel_config.json \
    --pipeline-parallel-size 2 --steps 1000
"""

import argparse
import time
from typing import Iterable
import numpy as np
import onnx
import torch
from torch import nn
import torch.distributed as dist
import deepspeed
from deepspeed.pipe import PipelineModule

from torchvision import datasets, transforms

from onnxruntime.training.ortmodule import ORTModule
from onnxruntime.training.ortmodule._onnx_graph_module import OnnxGraphModule


def wrap_model_as_onnx_graph_module(model, d_in, save_prefix=""):
    print("Saving inference / training onnx models")
    model = ORTModule(model)
    sample_input = torch.randn(10, d_in)  # 10 is an arbitrary batch size
    sample_input.requires_grad = True
    output = model(sample_input)
    loss = torch.sum(output)
    loss.backward()
    manager = model._torch_module._execution_manager._training_manager
    inference_model, training_model = manager._onnx_model, manager._optimized_onnx_model
    if len(save_prefix) > 0:
        onnx.save_model(inference_model, f"{save_prefix}_inference_model.onnx")
        onnx.save_model(training_model, f"{save_prefix}_training_model.onnx")
    return OnnxGraphModule(
        inference_model,
        training_model,
        user_input_names=["input"],  # this is an assumption that may be false
        require_grad_names=["input"],
        named_parameters=list(model.named_parameters()),
        initializer_names_to_train=[name for name, _ in model.named_parameters()],
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )
    parser.add_argument(
        "-s", "--steps", type=int, default=100, help="quit after this many steps"
    )
    parser.add_argument(
        "-p",
        "--pipeline-parallel-size",
        type=int,
        default=2,
        help="pipeline parallelism",
    )
    parser.add_argument(
        "--backend", type=str, default="nccl", help="distributed backend"
    )
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    parser.add_argument("--fp16", type=bool, default=False, help="fp16 run")
    parser.add_argument(
        "--run_without_ort", type=bool, default=False, help="onlydeepspeed run"
    )

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def group_every_n(iterable: Iterable, n):
    """ Groups every n elements from a iterable and yields iterators over them.
    """
    group = []
    for batch in iterable:
        group.append(batch)
        if len(group) == n:
            yield iter(group)
            group.clear()


def get_mnist_dataset_with_download(**kwargs):
    if dist.get_rank() == 0:
        # This may take a while
        return datasets.MNIST("./mnist", download=True, **kwargs)
    while True:
        try:
            return datasets.MNIST("./mnist", **kwargs)
        except RuntimeError:
            time.sleep(3)


def main():
    d_in = 784
    d_hidden = 512
    d_out = 10
    args = get_args()
    torch.cuda.set_device(args.local_rank)

    dist.init_process_group(backend=args.backend)
    torch.manual_seed(args.seed)

    if args.run_without_ort:
        print("Running deepspeed pipeline parallel module without ORTModule")
        model = nn.Sequential(
            nn.Linear(d_in, d_hidden),  # Stage 1
            nn.ReLU(),  # Stage 1
            nn.Linear(d_hidden, d_hidden),  # Stage 1
            nn.ReLU(),  # Stage 1
            nn.Linear(d_hidden, d_hidden),  # Stage 2
            nn.ReLU(),  # Stage 2
            nn.Linear(d_hidden, d_out),  # Stage 2
        )

    else:
        print("Running deepspeed pipeline parallel module with OnnxGraphModule")
        model = [
            wrap_model_as_onnx_graph_module(
                nn.Sequential(
                    nn.Linear(d_in, d_hidden),  # Stage 1
                    nn.ReLU(),  # Stage 1
                    nn.Linear(d_hidden, d_hidden),  # Stage 1
                    nn.ReLU(),  # Stage 1
                ),
                d_in,
                save_prefix="stage_1",
            ),
            wrap_model_as_onnx_graph_module(
                nn.Sequential(
                    nn.Linear(d_hidden, d_hidden),  # Stage 2
                    nn.ReLU(),  # Stage 2
                    nn.Linear(d_hidden, d_out),  # Stage 2
                ),
                d_hidden,
                save_prefix="stage_2",
            ),
        ]

    model = PipelineModule(
        layers=model,
        loss_fn=torch.nn.CrossEntropyLoss(),
        num_stages=args.pipeline_parallel_size,
        partition_method="uniform",  #'parameters',
        activation_checkpoint_interval=0,
    )

    params = [p for p in model.parameters() if p.requires_grad]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda t: t.view(-1)),
        ]
    )
    dataset = get_mnist_dataset_with_download(train=True, transform=transform)

    print("Initialize deepspeed")
    model_engine, _, _, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=params, training_data=dataset
    )

    for _ in range(args.steps):
        model_engine.train_batch()

    # inference
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("./mnist", train=False, transform=transform),
        batch_size=64,
        shuffle=False,
    )

    """
    Currently inference following training doesn't work. Need help from DeepSpeed people.
    test_losses = []
    for batch_iter in group_every_n(test_loader, n=2):
        test_loss = model_engine.eval_batch(batch_iter)
        print(test_loss)
        test_losses.append(test_loss.cpu().detach().numpy())
    print(f"Test loss: {np.mean(test_losses)}")
    """


if __name__ == "__main__":
    main()
