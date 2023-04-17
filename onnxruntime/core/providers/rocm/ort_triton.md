## Description
In some scenarios, the triton written kernel is more performant than CK or other handwritten kernels. We implement a framework for onnxruntime to use these triton written kernels.

Here we use `softmax` op as an example to show how to integrate a triton kernel into onnxruntime rocm EP.

#### Write and compile triton kernel
We have implemented a softmax kernel using triton at `onnxruntime/python/tools/kernel_explorer/kernels/rocm/triton/softmax.py`

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
This is a very simple implementation. The `n_cols` parameter should be smaller than BLOCK_SIZE. And BLOCK_SIZE is determined at compile time.

So in order to support different input shape, we should compile mulit kernels with different BLOCK_SIZE. All these kernels may have different BLOCK_SIZE, num_warps and shared memory usage. These parameters are needed by launching kernels in onnxruntime.

We develop a script `onnxruntime/python/tools/kernel_explorer/kernels/rocm/triton/compile_triton.py` to compile kernel and generate metadata for kernel launching.

First, it needs to add description info into `softmax.py`:
```
# kernel dtype and BLOCK_SIZE to generate.
dtypes = ['fp32', 'fp16']
blocks = [1024, 2048, 4096, 8192, 16384]
name_pattern = 'softmax_{}_{}'
sig_pattern = '*{},*{},i32,i32,i32'
group_pattern = 'softmax_{}'

# this function returns a description table that contains function name and metadata for launching kernels
def softmax_funcion_table():
    ......
    ......
```

And then add `softmax_funcion_table` function into  `onnxruntime/python/tools/kernel_explorer/kernels/rocm/triton/compile_triton.py` to register this kernel.
```
kernel_table = [
    softmax_funcion_table,  # softmax
]
```
Then run `python compile_triton.py`, it will generate the compiled kernel files and metadata in `libs`:

```
-rw-r--r-- 1 root root  1423 Apr 11 05:47 meta.json
-rw-r--r-- 1 root root  9720 Apr 11 05:47 softmax_fp16_1024.hasco
-rw-r--r-- 1 root root 17872 Apr 10 09:17 softmax_fp16_16384.hasco
-rw-r--r-- 1 root root  9720 Apr 11 05:47 softmax_fp16_2048.hasco
-rw-r--r-- 1 root root  9720 Apr 11 05:47 softmax_fp16_4096.hasco
-rw-r--r-- 1 root root 13776 Apr 10 09:17 softmax_fp16_8192.hasco
-rw-r--r-- 1 root root  9720 Apr 11 05:47 softmax_fp32_1024.hasco
-rw-r--r-- 1 root root 17872 Apr 10 09:17 softmax_fp32_16384.hasco
-rw-r--r-- 1 root root  9720 Apr 11 05:47 softmax_fp32_2048.hasco
-rw-r--r-- 1 root root  9720 Apr 11 05:47 softmax_fp32_4096.hasco
-rw-r--r-- 1 root root 13776 Apr 10 09:17 softmax_fp32_8192.hasco
```
The `meta.json` file contains metadata used for launching kernel, and `*.hsaco` files are compiled triton kernel for AMD GPU.

### onnxruntime c++ code moidfication
To use there kernels in onnxruntime, we need to open a compile flag `onnxruntime_USE_TRITON_KERNEL` in `cmake/CMakeList.txt`, and then build onnxruntime.
There is no dependency on triton.

Here we implement a tunable op for softmax to use triton kernel in `onnxruntime/core/providers/rocm/math/softmax_triton.cuh`
```
template <typename T, typename OutputT>
Status SoftmaxTritonOp(const SoftmaxParams<T, OutputT>* params) {
  if (params->is_log_softmax) {
    TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(true, "softmax triton not support log-softmax");
  }
  auto fname = GetSoftmaxTritonName<T>(params->softmax_elements);

  // construct args for launch kernel
  struct {
    void *out;
    const void *in;
    int in_stride;
    int out_stride;
    int n_cols;
  } args = {(void*)params->output, (const void*)params->input, params->input_stride, params->output_stride, params->softmax_elements};

  // grid dim is (batch_count, 1, 1)
  return LaunchTritonKernel(params->stream, fname, params->batch_count, 1, 1, &args, sizeof(args));
}

```
### runtime
The onnxruntime needs to load compiled triton kernel when executing operators.
Here is an environment variable `ORT_TRITON_LIB_PATH` that needs to be set to the path of compiled trtion kernels, and should use absolute path, for example:
`export ORT_TRITON_LIB_PATH=/ws/work/onnxruntime/onnxruntime/python/tools/kernel_explorer/kernels/rocm/triton/libs`

### Test
Using kernel_explorer, we can test this softmax kernel.
```
export KERNEL_EXPLORER_BUILD_DIR=/ws/code/onnxruntime/build_rocm/Debug
export ORT_TRITON_LIB_PATH=/ws/code/onnxruntime/onnxruntime/python/tools/kernel_explorer/kernels/rocm/triton/libs

python onnxruntime/python/tools/kernel_explorer/kernels/softmax_test.py
```
and can see the results, here TunableOp select SoftmaxTriton, it's better than others.
```
SoftmaxTunable               float16 batch_count=1    softmax_elements=2048 is_log_softmax=0   7.05 us, 1.16 GB/s
SoftmaxTriton                float16 batch_count=1    softmax_elements=2048 is_log_softmax=0   5.17 us, 1.59 GB/s
SoftmaxBlockwise_2           float16 batch_count=1    softmax_elements=2048 is_log_softmax=0   9.48 us, 0.86 GB/s
....
....
```
