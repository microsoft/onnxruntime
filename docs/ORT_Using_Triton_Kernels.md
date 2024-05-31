# Using Triton kernels in ONNX Runtime

In some scenarios, kernels written in Triton can be more performant than CUDA or other handwritten kernels, so we implement a framework that enables ONNX Runtime to use Triton kernels for the CUDA and ROCm execution providers.

## Write and compile a Triton kernel
### As a new ONNX Runtime operator
We first need a simple kernel written in Triton – which we've placed in [`onnxruntime/contrib_ops/cuda/my_triton_kernel.py`](onnxruntime/contrib_ops/cuda/my_triton_kernel.py):

```python
@triton.jit
def my_triton_kernel(output_ptr, input_ptr, N0, BLOCK_SIZE: tl.constexpr):
    """A simple kernel that just multiplies a 1-dimensional tensor by -1.

    Args:
        output_ptr: The pointer to the memory that the result will be written to
        input_ptr: The pointer to the memory containing the input tensor
        N0: The number of elements in the input tensor
        BLOCK_SIZE: The constant block size set when compiling this kernel
    """
    pid_0 = tl.program_id(0)
    block_offset = BLOCK_SIZE * pid_0
    block_range = block_offset + tl.arange(0, BLOCK_SIZE)

    # Read in one block of the input tensor
    x = tl.load(input_ptr + block_range, block_range < N0)

    # Write out -x to one block of the output tensor
    tl.store(output_ptr + block_range, -x, block_range < N0)
```

We then add the path to this file to the list of Triton kernels to be compiled within [`cmake/onnxruntime_compile_triton_kernel.cmake`](cmake/onnxruntime_compile_triton_kernel.cmake):
```cmake
if(onnxruntime_USE_CUDA)
  set(triton_kernel_scripts
      "onnxruntime/contrib_ops/cuda/my_triton_kernel.py"
  )
endif()
```

Now, if you build ONNX Runtime with the `--use_triton_kernel` flag, this Triton softmax kernel will be compiled and combined into `libonnxruntime_providers_rocm.so` for ROCm or `libonnxruntime_providers_cuda.so` for CUDA, but will _not_ yet be available in any operator.

To do that, we have to write some wrapper code in C++ that defines and registers an operator, and calls the compiled Triton kernel when run. In our example, this code is found in [`onnxruntime/contrib_ops/cuda/my_triton_kernel.h`](onnxruntime/contrib_ops/cuda/my_triton_kernel.h):

```cpp
// my_triton_kernel.h
#pragma once

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class MyTritonKernel final : public onnxruntime::cuda::CudaKernel {
  public:
    MyTritonKernel(const OpKernelInfo& info);
    Status ComputeInternal(OpKernelContext* context) const override;
  private:
    int64_t input_size;
    int64_t block_size;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
```

… and in [`onnxruntime/contrib_ops/cuda/my_triton_kernel.cc`](onnxruntime/contrib_ops/cuda/my_triton_kernel.cc)
```cpp
// my_triton_kernel.cc
#include "contrib_ops/cuda/my_triton_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

template <typename T>
std::string GetTypeAsString();

#define TYPE_TO_STRING(T,S)           \
  template <>                         \
  std::string GetTypeAsString<T>() {  \
    return S;                         \
  }

TYPE_TO_STRING(MLFloat16, "fp16");
TYPE_TO_STRING(float, "fp32");

template <typename T>
std::string GetMyTritonFunctionName(int64_t block_size) {
  std::string ret = "my_triton_kernel_";
  ret += GetTypeAsString<T>();
  ret += "_" + std::to_string(block_size);

  return ret;
}

}  // end of namespace

#define REGISTER_KERNEL_TYPED(T)                                \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                \
    MyTritonKernel,                                             \
    kMSDomain,                                                  \
    1,                                                          \
    T,                                                          \
    kCudaExecutionProvider,                                     \
    (*KernelDefBuilder::Create())                               \
        .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
    MyTritonKernel<T>);

REGISTER_KERNEL_TYPED(MLFloat16);
REGISTER_KERNEL_TYPED(float);

template <typename T>
MyTritonKernel<T>::MyTritonKernel(const OpKernelInfo& info) : CudaKernel{info} {
  input_size = info.GetAttrOrDefault<int64_t>("input_size", int64_t{128});
  block_size = info.GetAttrOrDefault<int64_t>("block_size", int64_t{64});
}

template <typename T>
Status MyTritonKernel<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  const TensorShape& X_shape = X->Shape();
  Tensor* Y = ctx->Output(0, X_shape);

  std::string function_name = GetMyTritonFunctionName<T>(block_size);
  int64_t grid_size = (X_shape[0] + block_size - 1) / block_size;
  cudaStream_t stream = Stream(ctx);

  struct {
    void* output_ptr;
    const void* input_ptr;
    int32_t input_size_;
  } args = {
    (void*)Y,
    (const void*)X,
    static_cast<int32_t>(input_size),
  };

  return onnxruntime::cuda::LaunchTritonKernel(stream, function_name, grid_size, 1, 1, &args, sizeof(args));
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
```

This operator then must be registered in two other files, so that ONNX Runtime is aware of it.

First, in [`onnxruntime/contrib_ops/cuda/cuda_contrib_kernels.cc`](onnxruntime/contrib_ops/cuda/cuda_contrib_kernels.cc):
```cpp
namespace onnxruntime {
namespace contrib {
namespace cuda {
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, MyTritonKernel);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, MyTritonKernel);

...

Status RegisterCudaContribKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
    BuildKernelCreateInfo<void>,  // default entry to avoid the list become empty after ops-reducing
    BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, MyTritonKernel)>,
    BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, MyTritonKernel)>,
    ...
  };
...
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
```

Then in [`onnxruntime/core/graph/contrib_ops/contrib_defs.cc`](onnxruntime/core/graph/contrib_ops/contrib_defs.cc):
```cpp
void RegisterContribSchemas() {
  ...

  ONNX_CONTRIB_OPERATOR_SCHEMA(MyTritonKernel)
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
      .SetDoc("MyTritonKernel")
      .Attr("input_size",
            "The number of elements in the input vector",
            AttributeProto::INT, static_cast<int64_t>(128))
      .Attr("block_size",
            "Kernel block size",
            AttributeProto::INT, static_cast<int64_t>(64))
      .Input(0, "X", "Input data tensor", "T")
      .Output(0, "Y", "Output data tensor.", "T")
      .TypeConstraint(
          "T",
          {"tensor(float16)", "tensor(float)"},
          "Constrain input X type to float tensors.")
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        propagateShapeFromInputToOutput(ctx, 0, 0);
      });
  ...
}
```

You can now build and use ONNX Runtime! This must be done on a machine with a GPU. If you're building ONNX Runtime using the dockerfiles, ensuring access to the GPU during the `docker build` step can be difficult. If this is the case, you can use [`tools/scripts/compile_triton_kernels.sh`](tools/scripts/compile_triton_kernels.sh) to compile the kernels ahead of time, after which you can run `docker build` normally.

### As an additional kernel for a `TunableOp`
A `TunableOp` in ONNX Runtime is an operator that contains multiple kernels, all implementations of the same operation. Often, the individual kernels will have different performance benefits and tradeoffs in different hardware environments and input sizes/shapes. The `TunableOp` briefly profiles each of the available kernels, then uses the best performing kernel for the remainder of the session.

This is the perfect use case for adding a Triton kernel -- which might offer a different performance profile than existing CUDA implementations, but could also have situations in which it would be better to fall back to the CUDA implementations.

As an example of this, we have implemented a softmax kernel in Triton at `onnxruntime/core/providers/rocm/math/softmax_triton.py`:

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

Each kernel's unique `BLOCK_SIZE` leads to different optimal settings for `num_warps` and shared memory usage, which we call `metadata`. These metadata are then used when launching kernels in ONNX Runtime.

The same script as above, `tools/ci_build/compile_triton.py`, can then be used to compile the kernels and set the optimal metadata for each.

To set this metadata, we implement the function `get_function_table` in `softmax_triton.py`:

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

When compiling ONNX Runtime with `--use_triton_kernel` flag, this Triton softmax kernel will be compiled and combined into `libonnxruntime_providers_rocm.so` for ROCm or `libonnxruntime_providers_cuda.so` for CUDA, but will _not_ yet be available in any operator.

#### Adding the Triton kernel to the C++ implementation of the `TunableOp`

The next step we need is to implement a C++ operator that will actually call these Triton kernels.

Just as with `TunableOp`s with CUDA kernels, we implement a function that returns a list of all implemented variations of the Triton kernel. We do this in [`onnxruntime/core/providers/rocm/math/softmax_triton.cuh`](onnxruntime/core/providers/rocm/math/softmax_triton.cuh):

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
    // construct a lambda function that will run this kernel
    ...
    // add it to the list of kernels, `ret`
  }
  return ret;
}
```

We then register this list of kernels with the main softmax `TunableOp`, defined in [`onnxruntime/core/providers/rocm/math/softmax_tunable_op.cuh`](onnxruntime/core/providers/rocm/math/softmax_tunable_op.cuh):
```cpp
template <typename InputT, typename OutputT, typename AccT>
class SoftmaxTunableOp : public tunable::TunableOp<SoftmaxParams<InputT, OutputT>> {
 public:
  SoftmaxTunableOp() {
    // Various softmax kernels written in CUDA
    this->RegisterOp(SoftmaxStaticSelection<InputT, OutputT, AccT>);
    this->RegisterOp(SoftmaxWarpwiseStaticSelection<InputT, OutputT, AccT>);
    this->RegisterOp(SoftmaxBlockwiseStaticSelection<InputT, OutputT, AccT>);
    this->RegisterOp(SoftmaxBlockwiseOp<InputT, OutputT, AccT, 1>);
    this->RegisterOp(SoftmaxBlockwiseOp<InputT, OutputT, AccT, 2>);
    this->RegisterOp(SoftmaxBlockwiseOp<InputT, OutputT, AccT, 4>);
    this->RegisterOp(SoftmaxBlockwiseOp<InputT, OutputT, AccT, 8>);
    this->RegisterOp(SoftmaxBlockwiseOp<InputT, OutputT, AccT, 16>);

    ...

#ifdef USE_TRITON_KERNEL
    // If Triton kernels are enabled, we use the function we defined above to register our new kernels
    for (auto&& [_, op] : GetSoftmaxTritonOps<InputT, OutputT>()) {
      ORT_UNUSED_PARAMETER(_);
      this->RegisterOp(std::move(op));
    }
#endif
  }
};
```

#### Testing the `TunableOp`

Using `kernel_explorer`, we can test this `TunableOp`, and make sure it can use our Triton kernels, as follows:

```bash
export KERNEL_EXPLORER_BUILD_DIR=<ONNXRUNTIME_BUILD_DIR>

python onnxruntime/python/tools/kernel_explorer/kernels/softmax_test.py
```

and the result shows that the `TunableOp` selects `softmax_fp16_2048` – one of the variations of Triton kernel we wrote. Even better, it outperforms the CUDA kernels:

```text
SoftmaxTunable    float16 batch_count=1    softmax_elements=2048 is_log_softmax=0   4.27 us, 1.92 GB/s
softmax_fp16_2048 float16 batch_count=1    softmax_elements=2048 is_log_softmax=0   4.48 us, 1.83 GB/s
...
```
