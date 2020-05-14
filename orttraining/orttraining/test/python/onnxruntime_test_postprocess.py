import unittest
import pytest
import sys
import os
import copy
from numpy.testing import assert_allclose, assert_array_equal

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

from orttraining_test_utils import map_optimizer_attributes
from orttraining_test_transformers import BertModelTest, BertForPreTraining
from orttraining_test_data_loader import create_ort_test_dataloader
import onnxruntime

from onnxruntime.capi.ort_trainer import ORTTrainer, IODescription, ModelDescription, LossScaler, generate_sample

torch.manual_seed(1)
onnxruntime.set_seed(1)

class Test_PostPasses(unittest.TestCase):
    def get_onnx_model(self, model, model_desc, inputs, device):
        lr_desc = IODescription('Learning_Rate', [1,], torch.float32)
        model = ORTTrainer(model,
                           None,
                           model_desc,
                           "LambOptimizer",
                           map_optimizer_attributes,
                           lr_desc,
                           device,
                           world_rank=0,
                           world_size=1,
                           _opset_version=12)

        train_output = model.train_step(*inputs)
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
                nodes.append(node)
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
        raise FileNotFoundError("Unable to find '{0}' or '{1}' or '{2}'".format(name, rel, res))

    def test_layer_norm(self):
        class LayerNormNet(nn.Module):
            def __init__(self, target):
                super(LayerNormNet, self).__init__()
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

        input_desc = IODescription('input', [], "float32")
        output0_desc = IODescription('output0', [], "float32")
        output1_desc = IODescription('output1', [20, 5, 10, 10], "float32")
        output2_desc = IODescription('output2', [20, 5, 10, 10], "float32")
        model_desc = ModelDescription([input_desc], [output0_desc, output1_desc])

        learning_rate = torch.tensor([1.0000000e+00]).to(device)
        input_args=[input, learning_rate]

        onnx_model = self.get_onnx_model(model, model_desc, input_args, device)

        count_layer_norm = self.count_nodes(onnx_model, "LayerNormalization")
        count_nodes = self.count_all_nodes(onnx_model)
       
        assert count_layer_norm == 1
        assert count_nodes == 3

    def test_expand(self):
        class ExpandNet(nn.Module):
            def __init__(self, target):
                super(ExpandNet, self).__init__()
                self.loss = nn.CrossEntropyLoss()
                self.target = target

            def forward(self, x, x1):
                output = x.expand_as(x1)
                output = output + output
                loss = self.loss(output, self.target)
                return loss, output

        device = torch.device("cpu")
        target = torch.ones(5, 5, 2, dtype=torch.int64).to(device)
        model = ExpandNet(target).to(device)

        x = torch.randn(5, 3, 1, 2, dtype=torch.float32).to(device)
        x1 = torch.randn(5, 3, 5, 2, dtype=torch.float32).to(device)

        input0_desc = IODescription('input0', [5, 3, 1, 2], "float32")
        input1_desc = IODescription('input1', [5, 3, 5, 2], "float32")
        output0_desc = IODescription('output0', [], "float32")
        output1_desc = IODescription('output1', [5, 3, 5, 2], "float32")
        model_desc = ModelDescription([input0_desc, input1_desc], [output0_desc, output1_desc])

        learning_rate = torch.tensor([1.0000000e+00]).to(device)
        input_args = [x, x1, learning_rate]

        onnx_model = self.get_onnx_model(model, model_desc, input_args, device)

        # check that expand output has shape 
        expand_nodes = self.find_expand_nodes(onnx_model, "Expand")
        assert len(expand_nodes) == 1

        model_info = onnx_model.grah.value_info
        assert model_info[0].name == expand_nodes[0].output[0]
        assert model_info[0].type == onnx_model.graph.input[1].type

    def test_bert(self):
        device = torch.device("cpu")

        model_tester = BertModelTest.BertModelTester(self)
        config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels = model_tester.prepare_config_and_inputs()
        
        model = BertForPreTraining(config=config)
        model.eval()

        loss, prediction_scores, seq_relationship_score = model(input_ids,
                                                                attention_mask=input_mask,
                                                                token_type_ids=token_type_ids,
                                                                masked_lm_labels=token_labels,
                                                                next_sentence_label=sequence_labels)

        model_desc = ModelDescription([model_tester.input_ids_desc,
                                       model_tester.attention_mask_desc,
                                       model_tester.token_type_ids_desc,
                                       model_tester.masked_lm_labels_desc,
                                       model_tester.next_sentence_label_desc],
                                      [model_tester.loss_desc,
                                       model_tester.prediction_scores_desc,
                                       model_tester.seq_relationship_scores_desc])

        from collections import namedtuple
        MyArgs = namedtuple("MyArgs",
                            "local_rank world_size max_steps learning_rate warmup_proportion batch_size seq_len")
        args = MyArgs(local_rank=0,
                      world_size=1,
                      max_steps=100,
                      learning_rate=0.00001,
                      warmup_proportion=0.01,
                      batch_size=13,
                      seq_len=7)

        dataloader = create_ort_test_dataloader(model_desc.inputs_,
                                                args.batch_size,
                                                args.seq_len,
                                                device)
        learning_rate = torch.tensor(1.0e+0, dtype=torch.float32).to(device)
        for b in dataloader:
            batch = b
            break
        learning_rate = torch.tensor([1.00e+00]).to(device)
        inputs = batch + [learning_rate,]

        onnx_model = self.get_onnx_model(model, model_desc, inputs, device)

        self._bert_helper(onnx_model)

    def test_bert_from_ONNX(self):
        device = torch.device("cpu")
        onnx_model = onnx.load(self.get_name("bert_no_postprocessing.onnx"))

        vocab_size = 30528
        input_ids_desc = IODescription('input_ids', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes=vocab_size)
        segment_ids_desc = IODescription('segment_ids', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes=2)
        input_mask_desc = IODescription('input_mask', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes=2)
        masked_lm_labels_desc = IODescription('masked_lm_labels', ['batch', 'max_seq_len_in_batch'], torch.int64, num_classes=vocab_size)
        next_sentence_labels_desc = IODescription('next_sentence_labels', ['batch', ], torch.int64, num_classes=2)
        loss_desc = IODescription('loss', [], torch.float32)

        model_desc = ModelDescription([input_ids_desc, segment_ids_desc, input_mask_desc, masked_lm_labels_desc, next_sentence_labels_desc], [loss_desc])


        map_opimizer_attributes = None

        # verify that the initial model is not processed
        count_layer_norm = self.count_nodes(onnx_model, "LayerNormalization")
        assert count_layer_norm == 0

        model_info = onnx_model.graph.value_info
        assert len(model_info) == 0

        model = ORTTrainer(onnx_model, None, model_desc, "LambOptimizer",
                           map_optimizer_attributes, IODescription('Learning_Rate', [1,], torch.float32),
                           device, world_rank=0, world_size=1, _opset_version=12)

        self._bert_helper(model.onnx_model_)

    def _bert_helper(self, onnx_model):
        # count layer_norm
        count_layer_norm = self.count_nodes(onnx_model, "LayerNormalization")
        assert count_layer_norm == 12

        # get expand node and check output shape
        expand_nodes = self.find_nodes(onnx_model, "Expand")
        assert len(expand_nodes) == 1

        model_info = onnx_model.graph.value_info
        assert model_info[0].name == expand_nodes[0].output[0]
        assert model_info[0].type == onnx_model.graph.input[0].type


if __name__ == '__main__':
    unittest.main(module=__name__, buffer=True)

