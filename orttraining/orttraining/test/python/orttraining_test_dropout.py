import pytest
import torch
import copy
import onnxruntime
from onnxruntime.training.ortmodule import ORTModule, _utils
import _test_helpers

@pytest.mark.parametrize("device", ['cuda'])
def test_model_with_dropout(device):
    class NeuralNetWithDropout(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNetWithDropout, self).__init__()
 
            self.fc1_1 = torch.nn.Linear(input_size, hidden_size)
            self.dropout1 = torch.nn.Dropout(p=0.1)
            self.dropout2 = torch.nn.Dropout(p=0.1)
            self.dropout3 = torch.nn.Dropout(p=0.1)
            self.dropout4 = torch.nn.Dropout(p=0.1)
 
        def forward(self, input1):
            out1 = self.fc1_1(input1)
            out1 = self.dropout1(out1)
            out1 = self.dropout2(out1)
            out1 = self.dropout3(out1)
            out1 = self.dropout4(out1)
            
            return out1
 
    def run_step(model, x1):
        y1 = model(x1)
        loss = y1.sum()
        loss.backward()
        return y1
    N, D_in, H, D_out = 32, 784, 500, 10
    pt_model = NeuralNetWithDropout(D_in, H, D_out).to(device)
    ort_model = ORTModule(copy.deepcopy(pt_model))
    pt_x1 = torch.randn(N, D_in, device=device, requires_grad=True)
    ort_x1 = pt_x1.clone()
 
    torch.manual_seed(9999)
    pt_y1_1 = run_step(pt_model, pt_x1)
    torch.manual_seed(9999)
    pt_y1_2 = run_step(pt_model, pt_x1)
    _test_helpers.assert_values_are_close(pt_y1_1, pt_y1_2, atol=1e-06)
    onnxruntime.set_seed(9999)
    ort_y1_1 = run_step(ort_model, ort_x1)
    onnxruntime.set_seed(9999)
    ort_y1_2 = run_step(ort_model, ort_x1)
    # _test_helpers.assert_values_are_close(pt_y1, ort_y1_1, atol=1e-06)
    _test_helpers.assert_values_are_close(ort_y1_1, ort_y1_2, atol=1e-06)
    _test_helpers.assert_gradients_match_and_reset_gradient(ort_model, pt_model)