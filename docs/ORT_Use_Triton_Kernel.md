## Description

In some scenarios, the Triton written kernels are more performant than CK or other handwritten kernels, so we implement a framework that enables onnxruntime to use these Triton written kernels.

Here we use `softmax` op as an example to show how to integrate a Triton written kernel into onnxruntime CUDA/ROCm EP.

### Write and compile Triton kernel

We have implemented a softmax kernel using Triton at `onnxruntime/core/providers/rocm/math/softmax_triton.py`

```python
@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # softmax implementations
    ...
    ...
```

This is a very simple implementation. The `n_cols` parameter should be smaller than `BLOCK_SIZE`. And `BLOCK_SIZE` MUST be determined at compile time.

In order to support different input shapes, we compile multiple kernels with different `BLOCK_SIZE`s.

Each kernel with different `BLOCK_SIZE` generates different `num_warps` and shared memory usage, we call them `metadata`, and these metadata are needed when launching kernels in onnxruntime.

We develop a script `tools/ci_build/compile_triton.py` to compile kernel and generate metadata for kernel launching.

To generate metadata for softmax, we need to add description info and implement a `get_function_table` function in `softmax_triton.py`:

```python
# kernel dtype and BLOCK_SIZE to generate.
dtypes = ['fp32', 'fp16']
blocks = [1024, 2048, 4096, 8192, 16384]
name_pattern = 'softmax_{}_{}'
sig_pattern = '*{},*{},i32,i32,i32'
group_pattern = 'softmax_{}'

"""
SHOULD implement a function that returns a metadata list with format:

function_table = [
    {
        'name': xx,
        'group': yy,
        'func': func,
        'sig': sig,
        'kwargs': kwargs
    }
]

The kwargs is a dict of {string: int} which is used for kernel constants. For example, BLOCK_SIZE of softmax.
"""

def get_function_table():
    ...
```

When compiling onnxruntime with `--use_triton_kernel` flag, this softmax kernel will be compiled and combined into `libonnxruntime_providers_rocm.so` for ROCm or `libonnxruntime_providers_cuda.so` for CUDA.

### onnxruntime C++ code modification

To use the Triton kernels in onnxruntime, we need to implement a C++ op that calls these Triton kernels.

Similar with CK, we implement a function that returns all possible Triton kernels, and the `TunableOp` will select the best one.

```cpp
template <typename T, typename OutputT>
auto GetSoftmaxTritonOps() {
  std::vector<std::pair<std::string, tunable::Op<SoftmaxParams<T, OutputT>>>> ret;
  auto group_name = GetSoftmaxTritonGroupName<T>();
  // here use group_name to get all kernel with same group_name
  // for example, 'softmax_fp16' represents a group of kernels with different BLOCK_SIZE for float16 softmax
  auto *kernel_list = GetOrtTritonKernelByGroup(group_name);
  if (kernel_list == nullptr) {
    return ret;
  }

  for (auto i : *kernel_list) {
    // check params match
    ...
  }
  return ret;
}
```

### Test

Using kernel_explorer, we can test this softmax kernel like:

```bash
export KERNEL_EXPLORER_BUILD_DIR=<ONNXRUNTIME_BUILD_DIR>

python onnxruntime/python/tools/kernel_explorer/kernels/softmax_test.py
```

and the result shows that `TunableOp` selects `softmax_fp16_2048` which is a Triton written kernel and better than others.

```text
SoftmaxTunable    float16 batch_count=1    softmax_elements=2048 is_log_softmax=0   4.27 us, 1.92 GB/s
softmax_fp16_2048 float16 batch_count=1    softmax_elements=2048 is_log_softmax=0   4.48 us, 1.83 GB/s
...
```
