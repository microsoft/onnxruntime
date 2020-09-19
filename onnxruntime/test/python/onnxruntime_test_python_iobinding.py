import numpy as np
import onnx
import onnxruntime
import unittest

from onnx import numpy_helper
from helper import get_name

class TestIOBinding(unittest.TestCase):

    def create_ortvalue_input_on_gpu(self):
        return onnxruntime.OrtValue.tensor_from_numpy(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32), 'cuda', 0)

    def create_ortvalue_alternate_input_on_gpu(self):
        return onnxruntime.OrtValue.tensor_from_numpy(np.array([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]], dtype=np.float32), 'cuda', 0)

    def create_uninitialized_ortvalue_input_on_gpu(self):
        return onnxruntime.OrtValue.tensor_from_shape_and_type([3, 2], np.float32, 'cuda', 0)

    def create_numpy_input(self):
        return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    def create_expected_output(self):
        return np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)

    def create_expected_output_alternate(self):
        return np.array([[2.0, 8.0], [18.0, 32.0], [50.0, 72.0]], dtype=np.float32)

    def test_bind_input_only(self):
        input = self.create_ortvalue_input_on_gpu()

        session = onnxruntime.InferenceSession(get_name("mul_1.onnx"))
        io_binding = session.io_binding()
        
        # Bind input to CUDA
        io_binding.bind_input('X', 'cuda', 0, np.float32, [3, 2], input.data_ptr())

        # Bind output to CPU
        io_binding.bind_output('Y')
        
        # Invoke Run
        session.run_with_iobinding(io_binding)
        
        # Get outputs over to CPU (the outputs which were bound to CUDA will get copied over to the host here)
        ort_output = io_binding.copy_outputs_to_cpu()[0]

        # Validate results
        self.assertTrue(np.array_equal(self.create_expected_output(), ort_output))

    def test_bind_input_to_cpu_arr(self):
        input = self.create_numpy_input()

        session = onnxruntime.InferenceSession(get_name("mul_1.onnx"))
        io_binding = session.io_binding()
        
        # Bind Numpy object (input) to CUDA
        io_binding.bind_cpu_input('X', self.create_numpy_input())
        
        # Bind output to CPU
        io_binding.bind_output('Y')
        
        # Invoke Run
        session.run_with_iobinding(io_binding)
        
        # Get outputs over to CPU (the outputs which were bound to CUDA will get copied over to the host here)
        ort_output = io_binding.copy_outputs_to_cpu()[0]

        # Validate results
        self.assertTrue(np.array_equal(self.create_expected_output(), ort_output))

    def test_bind_input_and_preallocated_output(self):
        input = self.create_ortvalue_input_on_gpu()

        session = onnxruntime.InferenceSession(get_name("mul_1.onnx"))
        io_binding = session.io_binding()
        
        # Bind input to CUDA
        io_binding.bind_input('X', 'cuda', 0, np.float32, [3, 2], input.data_ptr())

        # Bind output to CUDA
        output = self.create_uninitialized_ortvalue_input_on_gpu()
        io_binding.bind_output('Y', 'cuda', 0, np.float32, [3, 2], output.data_ptr())

        # Invoke Run
        session.run_with_iobinding(io_binding)
        
        # Get outputs over to CPU (the outputs which were bound to CUDA will get copied over to the host here)
        ort_output_vals = io_binding.copy_outputs_to_cpu()[0]
        # Validate results
        self.assertTrue(np.array_equal(self.create_expected_output(), ort_output_vals))
        
        # Validate if ORT actually wrote to pre-allocated buffer by copying the Torch allocated buffer
        # to the host and validating its contents
        ort_output_vals_in_cpu = output.numpy()
        # Validate results
        self.assertTrue(np.array_equal(self.create_expected_output(), ort_output_vals_in_cpu))


    def test_bind_input_and_non_preallocated_output(self):
        session = onnxruntime.InferenceSession(get_name("mul_1.onnx"))
        io_binding = session.io_binding()
        
        # Bind input to CUDA
        io_binding.bind_input('X', 'cuda', 0, np.float32, [3, 2], self.create_ortvalue_input_on_gpu().data_ptr())

        # Bind output to CUDA
        io_binding.bind_output('Y', 'cuda')

        # Invoke Run
        session.run_with_iobinding(io_binding)

        # This call returns an OrtValue which has data allocated by ORT on CUDA
        ort_outputs = io_binding.get_outputs()
        self.assertEqual(len(ort_outputs), 1)
        self.assertEqual(ort_outputs[0].device_name(), "cuda")
        # Validate results (by copying results to CPU by creating a Numpy object)
        self.assertTrue(np.array_equal(self.create_expected_output(), ort_outputs[0].numpy()))
        
        # We should be able to repeat the above process as many times as we want - try once more
        ort_outputs = io_binding.get_outputs()
        self.assertEqual(len(ort_outputs), 1)
        self.assertEqual(ort_outputs[0].device_name(), "cuda")
        # Validate results (by copying results to CPU by creating a Numpy object)
        self.assertTrue(np.array_equal(self.create_expected_output(), ort_outputs[0].numpy()))

        # Change the bound input and validate the results in the same bound OrtValue
        # Bind alternate input to CUDA
        io_binding.bind_input('X', 'cuda', 0, np.float32, [3, 2], self.create_ortvalue_alternate_input_on_gpu().data_ptr())
        
        # Invoke Run
        session.run_with_iobinding(io_binding)

        # This call returns an OrtValue which has data allocated by ORT on CUDA
        ort_outputs = io_binding.get_outputs()
        self.assertEqual(len(ort_outputs), 1)
        self.assertEqual(ort_outputs[0].device_name(), "cuda")
        # Validate results (by copying results to CPU by creating a Numpy object)
        self.assertTrue(np.array_equal(self.create_expected_output_alternate(), ort_outputs[0].numpy()))


if __name__ == '__main__':
    unittest.main()
