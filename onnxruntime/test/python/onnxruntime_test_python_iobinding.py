import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
import unittest

from onnx import numpy_helper


class AtomicModel(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels):
        super(AtomicModel, self).__init__()

        self.conv = nn.Conv2d(kernel_size=kernel_size, 
                              in_channels=in_channels,
                              out_channels=out_channels)
    def forward(self, x):
        return self.conv(x)

class TestIOBinding(unittest.TestCase):

    def create_model_and_input(self):
        kernel_size = 5
        channels = 16
        device = torch.device('cuda')
        batch_size = 1
        sample_dim = 256

        model = AtomicModel(kernel_size=kernel_size, in_channels=channels, out_channels=channels)
        model.eval()
        model.to(device)
        
        # Run Pytorch
        touch_input = torch.randn(batch_size, channels, sample_dim, sample_dim).to(device)
        torch_output = model(touch_input)
        torch_output_vals = torch_output.cpu().detach().numpy()
        
        # Run ORT
        input_names = [ "input" ]
        output_names = [ "output" ]
        torch.onnx.export(model, touch_input, "model.onnx", 
                          input_names=input_names, output_names=output_names,
                          dynamic_axes={"input":{0:"batch_size"}, "output":{0:"batch_size"}})
        
        return touch_input, torch_output, torch_output_vals

    def test_bind_input_only(self):
        torch_input, torch_output, torch_output_vals = self.create_model_and_input()

        session = onnxruntime.InferenceSession('model.onnx')
        io_binding = session.io_binding()
        
        # Bind input to CUDA
        io_binding.bind_input('input', torch_input.device.type, 0, np.float32, list(torch_input.size()), torch_input.data_ptr())

        # Bind output to CPU
        io_binding.bind_output('output')
        
        # Invoke Run
        session.run_with_iobinding(io_binding)
        
        # Get outputs over to CPU (the outputs which were bound to CUDA will get copied over to the host here)
        ort_output = io_binding.copy_outputs_to_cpu()[0]

        # Validate results
        self.assertTrue(np.array_equal(torch_output_vals, ort_output))

    def test_bind_input_to_cpu_arr(self):
        torch_input, torch_output, torch_output_vals = self.create_model_and_input()

        session = onnxruntime.InferenceSession('model.onnx')
        io_binding = session.io_binding()
        
        # Bind Numpy object (input) to CUDA
        io_binding.bind_cpu_input('input', torch_input.cpu().detach().numpy())
        
        # Bind output to CPU
        io_binding.bind_output('output')
        
        # Invoke Run
        session.run_with_iobinding(io_binding)
        
        # Get outputs over to CPU (the outputs which were bound to CUDA will get copied over to the host here)
        ort_output = io_binding.copy_outputs_to_cpu()[0]

        # Validate results
        self.assertTrue(np.array_equal(torch_output_vals, ort_output))

    def test_bind_input_and_preallocated_output(self):
        torch_input, torch_output, torch_output_vals = self.create_model_and_input()

        session = onnxruntime.InferenceSession('model.onnx')
        io_binding = session.io_binding()

        # Bind input to CUDA
        io_binding.bind_input('input', torch_input.device.type, 0, np.float32, list(torch_input.size()), torch_input.data_ptr())

        # Bind output to CUDA
        ort_output = torch.empty(list(torch_output.size())).cuda()
        io_binding.bind_output('output', torch_output.device.type, 0, np.float32, list(torch_output.size()), ort_output.data_ptr())

        # Invoke Run
        session.run_with_iobinding(io_binding)
        
        # Get outputs over to CPU (the outputs which were bound to CUDA will get copied over to the host here)
        ort_output_vals = io_binding.copy_outputs_to_cpu()[0]        
        # Validate results
        self.assertTrue(np.array_equal(torch_output_vals, ort_output_vals))
        
        # Validate if ORT actually wrote to pre-allocated buffer by copying the Torch allocated buffer
        # to the host and validating its contents
        ort_output_vals = ort_output.cpu().detach().numpy()
        # Validate results
        self.assertTrue(np.array_equal(torch_output_vals, ort_output_vals))
    
    def test_bind_input_and_non_preallocated_output(self):
        torch_input, torch_output, torch_output_vals = self.create_model_and_input()

        session = onnxruntime.InferenceSession('model.onnx')
        io_binding = session.io_binding()

        # Bind input to CUDA
        io_binding.bind_input('input', torch_input.device.type, 0, np.float32, list(torch_input.size()), torch_input.data_ptr())

        # Bind output to CUDA
        # DISCLAIMER: This is only useful for ORT benchmarking as there is really no way to access the 
        # ORT allocated device memory for this bound output at this point as ORT doesn't provide
        # a handle over this memory yet and to access the contents of this memory, we would have to copy contents
        # over to the host (CPU) using copy_outputs_to_cpu() as done in the next steps.
        # In future, we will provide a handle over the ORT allocated device memory for the bound output.
        # For now, this is a useful tool for benchmarking as Run() doesn't include the device-host copy latency
        # by doing this.
        io_binding.bind_output('output', 'cuda')

        # Invoke Run
        session.run_with_iobinding(io_binding)
        
        # Get outputs over to CPU (the outputs which were bound to CUDA will get copied over to the host here)
        ort_output = io_binding.copy_outputs_to_cpu()[0]
        
        # Validate results
        self.assertTrue(np.array_equal(torch_output_vals, ort_output))



if __name__ == '__main__':
    unittest.main()
