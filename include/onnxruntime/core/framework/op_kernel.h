// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>

#include "core/common/exceptions.h"
#include "core/common/logging/logging.h"
#include "core/common/status.h"
#include "core/common/profiler.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/ml_value.h"
#include "core/framework/op_kernel_info.h"
#include "core/framework/op_node_proto_helper.h"
#include "core/framework/tensor.h"
#include "core/framework/sparse_tensor.h"
#include "core/graph/constants.h"
#include "core/graph/graph_viewer.h"
#include "gsl/gsl"
#include "onnx/defs/schema.h"

namespace onnxruntime {
class IExecutionFrame;
class OpKernelContext;
class OpKernelWrapper;
namespace concurrency {
class ThreadPool;
}

class OpKernel {
 public:
  using DoneCallback = std::function<void()>;

  explicit OpKernel(const OpKernelInfo& info) : op_kernel_info_(info) {}
  virtual ~OpKernel() = default;

  const onnxruntime::Node& Node() const {
    return op_kernel_info_.node();
  }

  const onnxruntime::KernelDef& KernelDef() const {
    return op_kernel_info_.GetKernelDef();
  }

  virtual Status Compute(OpKernelContext* context) const ORT_MUST_USE_RESULT = 0;

  virtual Status ComputeAsync(OpKernelContext*, DoneCallback) const ORT_MUST_USE_RESULT {
    ORT_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }

  const OrtMemoryInfo& Allocator(int id, OrtMemType mem_type) const {
    return op_kernel_info_.GetMemoryInfo(id, mem_type);
  }

  const OpKernelInfo& Info() const { return op_kernel_info_; }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OpKernel);
  OpKernelInfo op_kernel_info_;
};

class OpKernelContext {
 public:
  using ArgMap = std::unordered_map<std::string, size_t>;

  explicit OpKernelContext(IExecutionFrame* frame,
                           const OpKernel* kernel,
                           concurrency::ThreadPool* threadpool,
                           const logging::Logger& logger,
                           profiling::Profiler* profiler = nullptr);

  virtual ~OpKernelContext() = default;

  /**
  Return the number of inputs for a variadic argument.
  @param arg_num The operator argument number.
  @returns Number of inputs the argument has.
  */
  int NumVariadicInputs(size_t arg_num) const;

  MLDataType InputType(int index) const;
  MLDataType OutputType(int index) const;

  template <typename T>
  const T* Input(int index) const {
    const OrtValue* p_ml_value = GetInputMLValue(index);
    try {
      return p_ml_value ? &(p_ml_value->Get<T>()) : nullptr;
    } catch (const std::exception& /*e*/) {
      throw OnnxRuntimeException(ORT_WHERE_WITH_STACK, "Missing Input: " + kernel_->Node().InputDefs()[index]->Name());
    }
  }

  // Fetch output (non-tensor) with specified index.
  template <typename T>
  T* Output(int index) {
    if (index < 0 || index >= OutputCount())
      return nullptr;

    OrtValue* p_ml_value = GetOrCreateOutputMLValue(index);
    return p_ml_value ? p_ml_value->GetMutable<T>() : nullptr;
  }

  // In the case that memory allocation has not been done for an output tensor,
  // The memory allocation will be done on-the-fly with given tensor shape.
  // Return nullptr if the output is an unused optional output.
  Tensor* Output(int index, const TensorShape& shape);

  // Fetch a sparse-tensor output corresponding to the specified index.
  // num_values must specify the number of non-zero values (commonly known as NNZ/nnz),
  // and shape must specify the shape of the underlying dense-tensor.
  // Memory allocation for the output may happen when this method is invoked,
  // unless static optimization pre-allocates it.
  SparseTensor* Output(int index, size_t num_values, const TensorShape& shape);

  const logging::Logger& Logger() const {
    return *logger_;
  }

  profiling::Profiler* Profiler() const {
    return profiler_;
  }

  // always >= 0
  int InputCount() const {
    return static_cast<int>(kernel_->Node().InputDefs().size());
  }

  // always >= 0
  int ImplicitInputCount() const {
    return static_cast<int>(kernel_->Node().ImplicitInputDefs().size());
  }

  // always >= 0
  int OutputCount() const {
    return static_cast<int>(kernel_->Node().OutputDefs().size());
  }

  /**
   * return an allocator on device 0, with memtype of OrtMemTypeDefault
   *
   */
  Status GetTempSpaceAllocator(AllocatorPtr* output) const;

  /**
  Return the fence of current node's input.
  @param index The index of the input.
  @returns Point to the Fence of the input OrtValue.
  It is null if the input OrtValue doesn't have fence or the input is optional.
  */
  Fence_t InputFence(int index) const;

  /**
  Return the fence of current node's implicit input.
  @param index The index of the implicit input.
  @returns Point to the Fence of the implicit input OrtValue.
  It is null if the input OrtValue doesn't have fence or the input is optional.
  */
  Fence_t ImplicitInputFence(int index) const;

  /**
  Return the fence of current node's output identifed by index.
  @param index The index of the output.
  @returns Point to the Fence of the output OrtValue.
  It is null if the output OrtValue doesn't have fence or the output is optional.
  */
  Fence_t OutputFence(int index) const;

  /**
  Returns the opset domain of the underlying kernel
  **/
  const std::string& GetOpDomain() const;

  /** 
  Returns the intra-op threadpool, if available.
  */
  _Ret_maybenull_ onnxruntime::concurrency::ThreadPool* GetOperatorThreadPool() const { return threadpool_; }

 protected:
  onnxruntime::NodeIndex GetNodeIndex() const;

  const OrtValue* GetInputMLValue(int index) const;
  const OrtValue* GetImplicitInputMLValue(int index) const;
  OrtValue* GetOutputMLValue(int index);

  // Creates the OrtValue* based on the shape, if it does not exist
  // The parameter nnz is used only for sparse-tensors and indicates the
  // number of non-zero values (the number of elements in the values buffer allocated).
  OrtValue* OutputMLValue(int index, const TensorShape& shape, size_t nnz = 0);

 private:
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(OpKernelContext);

  OrtValue* GetOrCreateOutputMLValue(int index);

  int GetInputArgIndex(int index) const;
  int GetImplicitInputArgIndex(int index) const;
  int GetOutputArgIndex(int index) const;

  IExecutionFrame* execution_frame_{nullptr};
  const OpKernel* kernel_{nullptr};
  concurrency::ThreadPool* threadpool_{nullptr};
  const logging::Logger* logger_{nullptr};
  profiling::Profiler* profiler_{nullptr};

  // The argument starting index in ExecutionFrame.
  int node_input_start_index_{-1};
  int node_implicit_input_start_index_{-1};
  int node_output_start_index_{-1};
};

// Fetching output tensor without shape is not allowed except when it already exists
template <>
inline Tensor* OpKernelContext::Output<Tensor>(int index) {
  OrtValue* p_ml_value = GetOutputMLValue(index);
  ORT_ENFORCE(p_ml_value, "Please fetch output tensor with specified shape.");
  return p_ml_value->GetMutable<Tensor>();
}

using KernelCreateFn = std::function<OpKernel*(const OpKernelInfo& info)>;
using KernelCreatePtrFn = std::add_pointer<OpKernel*(const OpKernelInfo& info)>::type;

struct KernelCreateInfo {
  std::unique_ptr<KernelDef> kernel_def;  // Owned and stored in the global kernel registry.
  KernelCreateFn kernel_create_func;
  Status status;

  KernelCreateInfo(std::unique_ptr<KernelDef> definition,
                   KernelCreateFn create_func)
      : kernel_def(std::move(definition)),
        kernel_create_func(create_func) {}

  KernelCreateInfo(KernelCreateInfo&& other) noexcept
      : kernel_def(std::move(other.kernel_def)),
        kernel_create_func(std::move(other.kernel_create_func)) {}
};

using KernelCreateMap = std::multimap<std::string, KernelCreateInfo>;

// Forward declarations for the non-specialized BuildKernelCreateInfo method.
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

namespace ml {
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();
}  // namespace ml

namespace contrib {
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();
}  // namespace contrib

namespace automl {
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();
}  // namespace automl

namespace contrib {
namespace cuda {
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();
}  // namespace cuda
}  // namespace contrib

using BuildKernelCreateInfoFn = KernelCreateInfo (*)();

// Naming convention for operator kernel classes
#define ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, ver, name) \
  provider##_##name##_##domain##_ver##ver

#define ONNX_CPU_OPERATOR_KERNEL(name, ver, builder, ...) \
  ONNX_OPERATOR_KERNEL_EX(name, kOnnxDomain, ver, kCpuExecutionProvider, builder, __VA_ARGS__)

#define ONNX_CPU_OPERATOR_ML_KERNEL(name, ver, builder, ...) \
  ONNX_OPERATOR_KERNEL_EX(name, kMLDomain, ver, kCpuExecutionProvider, builder, __VA_ARGS__)

#define ONNX_OPERATOR_KERNEL_EX(name, domain, ver, provider, builder, ...)                                            \
  class ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, ver, name);                                                 \
  template <>                                                                                                         \
  KernelCreateInfo                                                                                                    \
  BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, ver, name)>() {                             \
    return KernelCreateInfo(                                                                                          \
        builder.SetName(#name)                                                                                        \
            .SetDomain(domain)                                                                                        \
            .SinceVersion(ver)                                                                                        \
            .Provider(provider)                                                                                       \
            .Build(),                                                                                                 \
        static_cast<KernelCreatePtrFn>([](const OpKernelInfo& info) -> OpKernel* { return new __VA_ARGS__(info); })); \
  }

#define ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(provider, domain, startver, endver, name) \
  provider##_##name##_##domain##_ver##startver##_##endver

#define ONNX_CPU_OPERATOR_VERSIONED_KERNEL(name, startver, endver, builder, ...) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(name, kOnnxDomain, startver, endver, kCpuExecutionProvider, builder, __VA_ARGS__)

#define ONNX_CPU_OPERATOR_VERSIONED_ML_KERNEL(name, startver, endver, builder, ...) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(name, kMLDomain, startver, endver, kCpuExecutionProvider, builder, __VA_ARGS__)

#define ONNX_OPERATOR_VERSIONED_KERNEL_EX(name, domain, startver, endver, provider, builder, ...)                     \
  class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(provider, domain, startver, endver, name);                          \
  template <>                                                                                                         \
  KernelCreateInfo                                                                                                    \
  BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(provider, domain, startver, endver, name)>() {      \
    return KernelCreateInfo(                                                                                          \
        builder.SetName(#name)                                                                                        \
            .SetDomain(domain)                                                                                        \
            .SinceVersion(startver, endver)                                                                           \
            .Provider(provider)                                                                                       \
            .Build(),                                                                                                 \
        static_cast<KernelCreatePtrFn>([](const OpKernelInfo& info) -> OpKernel* { return new __VA_ARGS__(info); })); \
  }

#define ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type, name) \
  provider##_##name##_##domain##_ver##ver##_##type

#define ONNX_CPU_OPERATOR_TYPED_KERNEL(name, ver, type, builder, ...) \
  ONNX_OPERATOR_TYPED_KERNEL_EX(name, kOnnxDomain, ver, type, kCpuExecutionProvider, builder, __VA_ARGS__)

#define ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(name, ver, type, builder, ...) \
  ONNX_OPERATOR_TYPED_KERNEL_EX(name, kMLDomain, ver, type, kCpuExecutionProvider, builder, __VA_ARGS__)

#define ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(name, ver, type, builder, ...) \
  ONNX_OPERATOR_TYPED_KERNEL_EX(name, kMSDomain, ver, type, kCpuExecutionProvider, builder, __VA_ARGS__)

#define ONNX_OPERATOR_TYPED_KERNEL_EX(name, domain, ver, type, provider, builder, ...)                                \
  class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type, name);                                     \
  template <>                                                                                                         \
  KernelCreateInfo                                                                                                    \
  BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type, name)>() {                 \
    return KernelCreateInfo(                                                                                          \
        builder.SetName(#name)                                                                                        \
            .SetDomain(domain)                                                                                        \
            .SinceVersion(ver)                                                                                        \
            .Provider(provider)                                                                                       \
            .Build(),                                                                                                 \
        static_cast<KernelCreatePtrFn>([](const OpKernelInfo& info) -> OpKernel* { return new __VA_ARGS__(info); })); \
  }

#define ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(provider, domain, startver, endver, type, name) \
  provider##_##name##_##domain##_ver##startver##_##endver##_##type

#define ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(name, startver, endver, type, builder, ...) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(name, kOnnxDomain, startver, endver, type, kCpuExecutionProvider, builder, __VA_ARGS__)

#define ONNX_CPU_OPERATOR_VERSIONED_TYPED_ML_KERNEL(name, startver, endver, type, builder, ...) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(name, kMLDomain, startver, endver, type, kCpuExecutionProvider, builder, __VA_ARGS__)

#define ONNX_CPU_OPERATOR_VERSIONED_TYPED_MS_KERNEL(name, startver, endver, type, builder, ...) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(name, kMSDomain, startver, endver, type, kCpuExecutionProvider, builder, __VA_ARGS__)

#define ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(name, domain, startver, endver, type, provider, builder, ...)                \
  class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(provider, domain, startver, endver, type, name);                     \
  template <>                                                                                                                \
  KernelCreateInfo                                                                                                           \
  BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(provider, domain, startver, endver, type, name)>() { \
    return KernelCreateInfo(                                                                                                 \
        builder.SetName(#name)                                                                                               \
            .SetDomain(domain)                                                                                               \
            .SinceVersion(startver, endver)                                                                                  \
            .Provider(provider)                                                                                              \
            .Build(),                                                                                                        \
        static_cast<KernelCreatePtrFn>([](const OpKernelInfo& info) -> OpKernel* { return new __VA_ARGS__(info); }));        \
  }

}  // namespace onnxruntime
