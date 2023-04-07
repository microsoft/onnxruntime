def clear_all_grad_fns():
    from onnxruntime.training.ortmodule.torch_cpp_extensions import torch_interop_utils

    torch_interop_utils.clear_all_grad_fns()


import atexit  # noqa: E402

# Clear all gradient functions, to avoid a deadlock issue.
# Check the called function for more detailed comments.
atexit.register(clear_all_grad_fns)
