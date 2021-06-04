import os
import sys
import unittest
import math
import torch


class HuggingfaceGelu(torch.nn.Module):
    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class HuggingfaceFastGelu(torch.nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


class MegatronGelu(torch.nn.Module):
    def forward(self, x):
        # The original implementation using ones_like, which might cause problem for input with dynamic axes in onnx.
        # return x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype) + torch.ones_like(x).to(dtype=x.dtype))
        return x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype) + 1.0)


class MegatronFastGelu(torch.nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))


test_cases = [
    ('huggingface', 'Gelu', HuggingfaceGelu),
    ('huggingface', 'FastGelu', HuggingfaceFastGelu),
    ('megatron', 'Gelu', MegatronGelu),
    ('megatron', 'FastGelu', MegatronFastGelu)
]


class TestGeluFusions(unittest.TestCase):
    def verify_node_count(self, bert_model, expected_node_count, test_name):
        for op_type, count in expected_node_count.items():
            if len(bert_model.get_nodes_by_op_type(op_type)) != count:
                print(f"Counters is not expected in test: {test_name}")
                for op, counter in expected_node_count.items():
                    print("{}: {} expected={}".format(op, len(bert_model.get_nodes_by_op_type(op)), counter))
            self.assertEqual(len(bert_model.get_nodes_by_op_type(op_type)), count)

    def test_fusions(self):
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from onnxruntime.transformers.optimizer import optimize_model

        for test_case in test_cases:
            source, operator, model_class = test_case
            model = model_class()
            dummy_input = torch.ones(3, dtype=torch.float32)
            test_name = f"{operator}_{source}"
            onnx_path = f"{test_name}.onnx"
            torch.onnx.export(model, (dummy_input), onnx_path, input_names=['input'], output_names=['output'])
            optimizer = optimize_model(onnx_path, 'bert')
            # optimizer.save_model_to_file(f"{operator}_{source}_opt.onnx")
            os.remove(onnx_path)
            expected_node_count = {operator: 1}
            self.verify_node_count(optimizer, expected_node_count, test_name)


if __name__ == '__main__':
    unittest.main()
