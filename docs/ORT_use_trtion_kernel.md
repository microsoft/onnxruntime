## Description
In some scenarios, the triton written kernels are more performant than CK or other handwritten kernels, so we implement a framework that onnxruntime can use these triton written kernels.

Here we use `softmax` op as an example to show how to integrate a triton written kernel into onnxruntime cuda/rocm EP.

#### Write and compile triton kernel
We have implemented a softmax kernel using triton at `onnxruntime/core/providers/rocm/math/softmax.py`

```
@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # softmax implementings
    ....
    ....
```
This is a very simple implementation. The `n_cols` parameter should be smaller than BLOCK_SIZE. And BLOCK_SIZE MUST be determined at compile time.

In order to support different input shape, we compile multiple kernels with different BLOCK_SIZE.

Each kernel with different BLOCK_SIZE generates different num_warps and shared memory usage, we call them as `metadata`, and these metadata are needed when launching kernels in onnxruntime.

We develop a script `tools/ci_build/compile_triton.py` to compile kernel and generate metadata for kernel launching.

To generate metadta for softmax, it needs to add description info and implement a `get_function_table` function in `softmax.py`:
```
# kernel dtype and BLOCK_SIZE to generate.
dtypes = ['fp32', 'fp16']
blocks = [1024, 2048, 4096, 8192, 16384]
name_pattern = 'softmax_{}_{}'
sig_pattern = '*{},*{},i32,i32,i32'
group_pattern = 'softmax_{}'

"""
SHOULD implement a function that returns a metadata list with format:

function_table = [
        {'name': xx,
         'group': yy,
         'func': func,
         'sig': sig,
         'kwargs': kwargs
        }
]

The kwargs is a dict of {string: int} which is used for kernel constants. For example, BLOCK_SIZE of softmax.
"""

def get_funcion_table():
    ......
    ......
```

When compiling onnxruntime with `--use_triton_kernel` flag, this softmax kernel will be compiled and combined into libonnxruntime_providers_rocm.so for rocm or libonnxruntime_providers_cuda.so for cuda.

### onnxruntime c++ code moidfication
To use the triton kernels in onnxruntime, we need to implement a c++ op that calls these triton kernels.

And there are many implements for softmax, we implement a tunable op for softmax to choose the best performant one.

Same as CK, we implement a function that returns all possible triton kernels, and the tunable op will select best one.

```
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
    .....
  }
  return ret;
}
```

### Test
Using kernel_explorer, we can test this softmax kernel.
```
export KERNEL_EXPLORER_BUILD_DIR=/ws/code/onnxruntime/build_rocm/Debug

python onnxruntime/python/tools/kernel_explorer/kernels/softmax_test.py
```
and can see the results, here TunableOp select softmax_fp16_2048 which is a triton written kernel, it's better than others.
```
SoftmaxTunable	float16 batch_count=1    softmax_elements=2048 is_log_softmax=0   4.27 us, 1.92 GB/s
softmax_fp16_2048	float16 batch_count=1    softmax_elements=2048 is_log_softmax=0   4.48 us, 1.83 GB/s
....
....
```
