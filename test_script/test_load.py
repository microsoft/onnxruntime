import onnxruntime as ort
from onnxruntime.capi import _pybind_state as C

session_options = C.get_default_session_options()
sess = C.InferenceSession(session_options, "model.onnx", True, True)
sess.initialize_session(['my_ep'], 
                        [{'shared_lib_path':'C:/git/onnxruntime/build/Windows/Debug/Debug/my_execution_provider.dll',
                                     'device_id':'1', 'some_config':'val'}], 
                        set())
print("Inference session construction pass")

import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxruntime.training import optim, orttrainer, orttrainer_options as orttrainer_options
from onnxruntime.capi.ort_trainer import IODescription, ModelDescription

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def mnist_model_description():
    return {'inputs':  [('x', ['batch', 784]),
                              ('target', ['batch',])],
                  'outputs': [('loss', [], True),
                              ('predictions', ['batch', 10])]}

input_size = 784
hidden_size = 500
num_classes = 10
model = NeuralNet(input_size, hidden_size, num_classes)

opts = {'device' : {'id' : '0'}, 
        'provider_options' : {'my_ep' : {'shared_lib_path' : 'C:/git/onnxruntime/build/Windows/Debug/Debug/my_execution_provider.dll',
                                         'device_id':'1', 'some_config':'val'}}}

opts = orttrainer.ORTTrainerOptions(opts)
optim_config = optim.SGDConfig(lr=0.001)

def my_loss(x, target):
    return F.nll_loss(F.log_softmax(x, dim=1), target)

trainer = orttrainer.ORTTrainer(model, mnist_model_description(), optim_config, loss_fn=my_loss, options=opts)
sample_input = torch.randn((10, input_size), dtype=torch.float32)
sample_lable = torch.randint(0, 10, (10,))
trainer._init_onnx_model((sample_input, sample_lable))
print("ORTTrainer init ok")