
def clear_all_grad_fns():
    from onnxruntime.training.ortmodule.torch_cpp_extensions import torch_interop_utils
    torch_interop_utils.clear_all_grad_fns()
