# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path

from olive.cli.base import BaseOliveCLICommand, add_input_model_options, add_telemetry_options, get_input_model_config
from olive.model import ModelConfig
from olive.telemetry import action

logger = logging.getLogger(__name__)


class GenerateCostModelCommand(BaseOliveCLICommand):
    @staticmethod
    def register_subcommand(parser):
        sub_parser = parser.add_parser(
            "generate-cost-model",
            help=(
                "Generate a cost model for a given model and save it as a csv file. This cost model is consumed by the"
                " CaptureSplitInfo pass. Only supports HfModel."
            ),
        )

        add_input_model_options(
            sub_parser, enable_hf=True, default_output_path="cost-model.csv", directory_output=False
        )

        sub_parser.add_argument(
            "-p",
            "--weight_precision",
            type=str,
            default="fp16",
            choices=PRECISON_TO_BYTES.keys(),
            help="Weight precision",
        )
        add_telemetry_options(sub_parser)
        sub_parser.set_defaults(func=GenerateCostModelCommand)

    @action
    def run(self):
        import torch

        model_handler = ModelConfig.model_validate(get_input_model_config(self.args)).create_model()

        # model costs
        costs = {}
        hidden_dim = 0
        for name, module in model_handler.load_model().named_modules():
            # pylint: disable=protected-access
            if module._modules:
                # has children
                continue

            num_params = sum(p.numel() for p in module.parameters())
            num_bytes = num_params * PRECISON_TO_BYTES[self.args.weight_precision]
            if isinstance(module, torch.nn.Linear):
                hidden_dim = module.out_features
                num_flops = 2 * num_params
            elif isinstance(module, torch.nn.Embedding):
                hidden_dim = module.embedding_dim
                num_flops = 1
            elif isinstance(module, torch.nn.LayerNorm) or module.__class__.__name__.endswith("RMSNorm"):
                num_flops = 4 * hidden_dim
            elif isinstance(module, torch.nn.SiLU):
                # activation functions are constant * hidden_dim
                # TODO(jambayk): add other activation functions if needed. Calculate the cost based on the function.
                num_flops = 6 * hidden_dim
            else:
                # unknown cost
                num_flops = 0

            costs[name] = (num_params, num_bytes, num_flops)

        # write to csv
        if not self.args.output_path.endswith(".csv"):
            # do this instead of using Path.with_suffix because it will remove periods in the path
            # but model names can have periods
            self.args.output_path += ".csv"
        output_path = Path(self.args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("module,num_params,num_bytes,num_flops\n")
            for module, (num_params, num_bytes, num_flops) in costs.items():
                f.write(f"{module},{num_params},{num_bytes},{num_flops}\n")

        print(f"Cost model written to {output_path}")
        print(
            f"Total cost: {sum(cost[1] for cost in costs.values())} bytes for {self.args.weight_precision} precision."
        )


PRECISON_TO_BYTES = {
    "fp32": 4,
    "fp16": 2,
    "fp8": 1,
    "int32": 4,
    "uint32": 4,
    "int16": 2,
    "uint16": 2,
    "int8": 1,
    "uint8": 1,
    "int4": 0.5,
    "uint4": 0.5,
    "nf4": 0.5,
    "fp4": 0.5,
}
