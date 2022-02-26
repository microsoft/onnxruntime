import time
import onnxruntime as ort
import torch


def create_io_binding(sess, inputs, outputs):
    io_binding = sess.io_binding()
    device = "cuda"

    for name, array in inputs.items():
      tensor = torch.from_numpy(array).to(device)
      io_binding.bind_input(name, tensor.device.type, 0, array.dtype, tensor.shape, tensor.data_ptr())

    for name, array in outputs.items():
      tensor = torch.from_numpy(array).to(device)
      io_binding.bind_output(name, tensor.device.type, 0, array.dtype, tensor.shape, tensor.data_ptr())
  
    return io_binding


def create_session(onnx_file, provider, profiling):
    sess_opt = ort.SessionOptions()
    sess_opt.enable_profiling = profiling
    if provider == "rocm":
        execution_provider = ["ROCMExecutionProvider"] 
    else:
        execution_provider = ["CUDAExecutionProvider"] 
        
    sess = ort.InferenceSession(onnx_file, sess_options=sess_opt, providers=execution_provider)
    return sess
 

def benchmark(onnx_file, inputs, outputs, provider, profiling=False):
    sess = create_session(onnx_file, provider, profiling)
    io_binding = create_io_binding(sess, inputs, outputs)

    # warm up    
    for iter in range(10):
      sess.run_with_iobinding(io_binding)    
    
    # measure 
    max_iters = 100
    start_time = time.time()
    for iter in range(max_iters):
        sess.run_with_iobinding(io_binding)    
    
    # time is in milliseconds
    elapsed_time = (time.time() - start_time) * 1000 / max_iters
    return elapsed_time
