# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# model_generation.py

import argparse
import os

import onnx

import onnxruntime.training.onnxblock as onnxblock


class MyTrainingModel(onnxblock.TrainingModel):
    def __init__(self):
        super().__init__()
        self._loss = {
            "MSELoss": onnxblock.loss.MSELoss(),
            "CrossEntropyLoss": onnxblock.loss.CrossEntropyLoss(),
            "BCEWithLogitsLoss": onnxblock.loss.BCEWithLogitsLoss(),
        }
        self._weights_for_averaging = [onnxblock.building_blocks.Constant(0.5), onnxblock.building_blocks.Constant(0.5)]
        self._add = onnxblock.building_blocks.Add()
        self._mul = onnxblock.building_blocks.Mul()

    def build(self, loss_input_names):
        if len(loss_input_names) == 1:
            return self._loss[args.loss](loss_input_names[0])

        return self._add(
            self._mul(self._weights_for_averaging[0](), self._loss[args.loss](loss_input_names[0], "target1")),
            self._mul(self._weights_for_averaging[1](), self._loss[args.loss](loss_input_names[1], "target2")),
        )


parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Path to the model file.")
parser.add_argument("--loss", help="Loss to build", type=str, default="MSELoss")
args = parser.parse_args()

model_dir = os.path.dirname(os.path.realpath(args.model))
model_path = args.model
output_model_path = os.path.join(model_dir, f"training_model_with_{args.loss}.onnx")
output_eval_model_path = os.path.join(model_dir, f"eval_model_with_{args.loss}.onnx")
output_optimizer_model_path = os.path.join(model_dir, "adamw_optimizer.onnx")
output_checkpoint_path = os.path.join(model_dir, "checkpoint")

model = onnx.load(model_path)
loss_inputs = [output.name for output in model.graph.output]

my_model = MyTrainingModel()
with onnxblock.onnx_model(model) as accessor:
    loss_output_name = my_model(loss_inputs)
    eval_model = accessor.eval_model

optimizer = onnxblock.optim.AdamW()
with onnxblock.onnx_model() as accessor:
    optimizer_outputs = optimizer(my_model.parameters())
    optimizer_model = accessor.model

# save all the models
onnx.save(model, output_model_path)
onnx.save(eval_model, output_eval_model_path)
onnx.save(optimizer_model, output_optimizer_model_path)

# save checkpoint
onnxblock.save_checkpoint(my_model.parameters(), output_checkpoint_path)
