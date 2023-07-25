import os
import unittest

import torch
import torch.nn as nn
from orttraining_test_bert_postprocess import postprocess_model
from orttraining_test_data_loader import create_ort_test_dataloader
from orttraining_test_transformers import BertForPreTraining, BertModelTest
from orttraining_test_utils import map_optimizer_attributes

import onnxruntime
from onnxruntime.capi.ort_trainer import (  # noqa: F401
    IODescription,
    LossScaler,
    ModelDescription,
    ORTTrainer,
    generate_sample,
)

torch.manual_seed(1)
onnxruntime.set_seed(1)


class Test_PostPasses(unittest.TestCase):  # noqa: N801
    def get_onnx_model(
        self, model, model_desc, inputs, device, _enable_internal_postprocess=True, _extra_postprocess=None
    ):
        lr_desc = IODescription(
            "Learning_Rate",
            [
                1,
            ],
            torch.float32,
        )
        model = ORTTrainer(
            model,
            None,
            model_desc,
            "LambOptimizer",
            map_optimizer_attributes,
            lr_desc,
            device,
            world_rank=0,
            world_size=1,
            _opset_version=14,
            _enable_internal_postprocess=_enable_internal_postprocess,
            _extra_postprocess=_extra_postprocess,
        )

        model.train_step(*inputs)
        return model.onnx_model_

    def count_all_nodes(self, model):
        return len(model.graph.node)

    def count_nodes(self, model, node_type):
        count = 0
        for node in model.graph.node:
            if node.op_type == node_type:
                count += 1
        return count

    def find_nodes(self, model, node_type):
        nodes = []
        for node in model.graph.node:
            if node.op_type == node_type:
                nodes.append(node)  # noqa: PERF401
        return nodes

    def get_name(self, name):
        if os.path.exists(name):
            return name
        rel = os.path.join("testdata", name)
        if os.path.exists(rel):
            return rel
        this = os.path.dirname(__file__)
        data = os.path.join(this, "..", "..", "..", "..", "onnxruntime", "test", "testdata")
        res = os.path.join(data, name)
        if os.path.exists(res):
            return res
        raise FileNotFoundError(f"Unable to find '{name}' or '{rel}' or '{res}'")

    def test_layer_norm(self):
        class LayerNormNet(nn.Module):
            def __init__(self, target):
                super().__init__()
                self.ln_1 = nn.LayerNorm(10)
                self.loss = nn.CrossEntropyLoss()
                self.target = target

            def forward(self, x):
                output1 = self.ln_1(x)
                loss = self.loss(output1, self.target)
                return loss, output1

        device = torch.device("cpu")
        target = torch.ones(20, 10, 10, dtype=torch.int64).to(device)
        model = LayerNormNet(target)
        input = torch.randn(20, 5, 10, 10, dtype=torch.float32).to(device)

        input_desc = IODescription("input", [], "float32")
        output0_desc = IODescription("output0", [], "float32")
        output1_desc = IODescription("output1", [20, 5, 10, 10], "float32")
        model_desc = ModelDescription([input_desc], [output0_desc, output1_desc])

        learning_rate = torch.tensor([1.0000000e00]).to(device)
        input_args = [input, learning_rate]

        onnx_model = self.get_onnx_model(model, model_desc, input_args, device)

        count_layer_norm = self.count_nodes(onnx_model, "LayerNormalization")
        count_nodes = self.count_all_nodes(onnx_model)

        assert count_layer_norm == 0
        assert count_nodes == 3

    def test_expand(self):
        class ExpandNet(nn.Module):
            def __init__(self, target):
                super().__init__()
                self.loss = nn.CrossEntropyLoss()
                self.target = target
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x, x1):
                output = x.expand_as(x1)
                output = self.linear(output)
                output = output + output
                loss = self.loss(output, self.target)
                return loss, output

        device = torch.device("cpu")
        target = torch.ones(5, 5, 2, dtype=torch.int64).to(device)
        model = ExpandNet(target).to(device)

        x = torch.randn(5, 3, 1, 2, dtype=torch.float32).to(device)
        x1 = torch.randn(5, 3, 5, 2, dtype=torch.float32).to(device)

        input0_desc = IODescription("x", [5, 3, 1, 2], "float32")
        input1_desc = IODescription("x1", [5, 3, 5, 2], "float32")
        output0_desc = IODescription("output0", [], "float32")
        output1_desc = IODescription("output1", [5, 3, 5, 2], "float32")
        model_desc = ModelDescription([input0_desc, input1_desc], [output0_desc, output1_desc])

        learning_rate = torch.tensor([1.0000000e00]).to(device)
        input_args = [x, x1, learning_rate]

        onnx_model = self.get_onnx_model(model, model_desc, input_args, device)

        # check that expand output has shape
        expand_nodes = self.find_nodes(onnx_model, "Expand")
        assert len(expand_nodes) == 1

        model_info = onnx_model.graph.value_info
        assert model_info[0].name == expand_nodes[0].output[0]
        assert model_info[0].type == onnx_model.graph.input[1].type

    def test_bert(self):
        device = torch.device("cpu")

        model_tester = BertModelTest.BertModelTester(self)
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = model_tester.prepare_config_and_inputs()

        model = BertForPreTraining(config=config)
        model.eval()

        loss, prediction_scores, seq_relationship_score = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            masked_lm_labels=token_labels,
            next_sentence_label=sequence_labels,
        )

        model_desc = ModelDescription(
            [
                model_tester.input_ids_desc,
                model_tester.attention_mask_desc,
                model_tester.token_type_ids_desc,
                model_tester.masked_lm_labels_desc,
                model_tester.next_sentence_label_desc,
            ],
            [model_tester.loss_desc, model_tester.prediction_scores_desc, model_tester.seq_relationship_scores_desc],
        )

        from collections import namedtuple

        MyArgs = namedtuple(
            "MyArgs", "local_rank world_size max_steps learning_rate warmup_proportion batch_size seq_len"
        )
        args = MyArgs(
            local_rank=0,
            world_size=1,
            max_steps=100,
            learning_rate=0.00001,
            warmup_proportion=0.01,
            batch_size=13,
            seq_len=7,
        )

        dataset_len = 100
        dataloader = create_ort_test_dataloader(model_desc.inputs_, args.batch_size, args.seq_len, dataset_len, device)
        learning_rate = torch.tensor(1.0e0, dtype=torch.float32).to(device)
        for b in dataloader:
            batch = b
            break
        learning_rate = torch.tensor([1.00e00]).to(device)
        inputs = [*batch, learning_rate]

        onnx_model = self.get_onnx_model(model, model_desc, inputs, device, _extra_postprocess=postprocess_model)

        self._bert_helper(onnx_model)

    def _bert_helper(self, onnx_model):
        # count layer_norm
        count_layer_norm = self.count_nodes(onnx_model, "LayerNormalization")
        assert count_layer_norm == 0

        # get expand node and check output shape
        expand_nodes = self.find_nodes(onnx_model, "Expand")
        assert len(expand_nodes) == 1

        model_info = onnx_model.graph.value_info
        assert model_info[0].name == expand_nodes[0].output[0]
        assert model_info[0].type == onnx_model.graph.input[0].type

    def test_extra_postpass(self):
        def postpass_replace_first_add_with_sub(model):
            # this post pass replaces the first Add node with Sub in the model.
            # Previous graph
            #   (subgraph 1)        (subgraph 2)
            #        |                   |
            #        |                   |
            #        |________   ________|
            #                 | |
            #                 Add
            #                  |
            #             (subgraph 3)
            #
            # Post graph
            #   (subgraph 1)        (subgraph 2)
            #        |                   |
            #        |                   |
            #        |________   ________|
            #                 | |
            #                 Sub
            #                  |
            #             (subgraph 3)
            add_nodes = [n for n in model.graph.node if n.op_type == "Add"]
            add_nodes[0].op_type = "Sub"

        class MultiAdd(nn.Module):
            def __init__(self, target):
                super().__init__()
                self.loss = nn.CrossEntropyLoss()
                self.target = target
                self.linear = torch.nn.Linear(2, 2, bias=False)

            def forward(self, x, x1):
                output = x + x1
                output = output + x
                output = output + x1
                output = self.linear(output)
                loss = self.loss(output, self.target)
                return loss, output

        device = torch.device("cpu")
        target = torch.ones(5, 2, dtype=torch.int64).to(device)
        model = MultiAdd(target).to(device)

        x = torch.randn(5, 5, 2, dtype=torch.float32).to(device)
        x1 = torch.randn(5, 5, 2, dtype=torch.float32).to(device)

        input0_desc = IODescription("x", [5, 5, 2], "float32")
        input1_desc = IODescription("x1", [5, 5, 2], "float32")
        output0_desc = IODescription("output0", [], "float32")
        output1_desc = IODescription("output1", [5, 5, 2], "float32")
        model_desc = ModelDescription([input0_desc, input1_desc], [output0_desc, output1_desc])

        learning_rate = torch.tensor([1.0000000e00]).to(device)
        input_args = [x, x1, learning_rate]

        onnx_model = self.get_onnx_model(
            model, model_desc, input_args, device, _extra_postprocess=postpass_replace_first_add_with_sub
        )

        # check that extra postpass is called, and called only once.
        add_nodes = self.find_nodes(onnx_model, "Add")
        sub_nodes = self.find_nodes(onnx_model, "Sub")
        assert len(add_nodes) == 2
        assert len(sub_nodes) == 1

        unprocessed_onnx_model = self.get_onnx_model(
            model, model_desc, input_args, device, _extra_postprocess=None, _enable_internal_postprocess=False
        )
        # check that the model is unchanged.
        add_nodes = self.find_nodes(unprocessed_onnx_model, "Add")
        sub_nodes = self.find_nodes(unprocessed_onnx_model, "Sub")
        assert len(add_nodes) == 3
        assert len(sub_nodes) == 0

        processed_onnx_model = self.get_onnx_model(
            unprocessed_onnx_model,
            model_desc,
            input_args,
            device,
            _extra_postprocess=postpass_replace_first_add_with_sub,
        )
        # check that extra postpass is called, and called only once.
        add_nodes = self.find_nodes(processed_onnx_model, "Add")
        sub_nodes = self.find_nodes(processed_onnx_model, "Sub")
        assert len(add_nodes) == 2
        assert len(sub_nodes) == 1


if __name__ == "__main__":
    unittest.main(module=__name__, buffer=True)
