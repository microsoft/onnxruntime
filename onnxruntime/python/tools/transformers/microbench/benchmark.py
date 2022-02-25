import time
import onnxruntime as ort


def create_session(onnx_file, provider, profiling=False):
    sess_opt = ort.SessionOptions()
    sess_opt.enable_profiling = profiling
    if provider == "rocm":
        execution_provider = ["ROCMExecutionProvider"] 
    else:
        execution_provider = ["CUDAExecutionProvider"] 
        
    sess = ort.InferenceSession(onnx_file, sess_options=sess_opt, providers=execution_provider)
    return sess
 

def benchmark(sess, io_binding):
    # warm up    
    for iter in range(10):
      sess.run_with_iobinding(io_binding)    
    
    # measure 
    max_iters = 100
    start_time = time.time()
    for iter in range(max_iters):
        sess.run_with_iobinding(io_binding)    
    
    # time is in milliseconds
    elapsed_time = (time.time() - start_time)*1000/max_iters
    return elapsed_time
