// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "boost/mp11.hpp"

#ifndef SHARED_PROVIDER
#include <functional>
#include "core/common/exceptions.h"
#include "core/common/logging/logging.h"
#include "core/common/status.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/ml_value.h"
#include "core/framework/op_kernel_info.h"
#include "core/framework/op_node_proto_helper.h"
#include "core/framework/tensor.h"
#include "core/framework/sparse_tensor.h"
#include "core/graph/constants.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/onnx_protobuf.h"
#include "gsl/gsl"
namespace onnxruntime {
class OpKernelContext;
}
#endif

namespace onnxruntime {

std::unique_ptr<OpKernelInfo> CopyOpKernelInfo(const OpKernelInfo& info);

class OpKernel {
 public:
  using DoneCallback = std::function<void()>;

  explicit OpKernel(const OpKernelInfo& info) : op_kernel_info_(CopyOpKernelInfo(info)) {}
  virtual ~OpKernel() = default;

  const onnxruntime::Node& Node() const;
  const onnxruntime::KernelDef& KernelDef() const;

  virtual Status Compute(_Inout_ OpKernelContext* context) const ORT_MUST_USE_RESULT = 0;

  virtual Status ComputeAsync(_Inout_ OpKernelContext*, DoneCallback) const ORT_MUST_USE_RESULT {
    ORT_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }

  struct PrepackParam {
    int input_idx = 0;           // Input index
    int sparse_flags = 0;   // Sparse initializer annotations
    int64_t ell_block_size = 8; // ell block size when BlockedEll format is used
    std::string name;
    bool Is2x4Format() const noexcept {
      return (sparse_flags & static_cast<int>(OrtSparseFlags::FORMAT_2x4)) != 0;
    }
    bool UseCsrFormat() const noexcept {
      return (sparse_flags & static_cast<int>(OrtSparseFlags::USE_CSR_FORMAT)) != 0;
    }
    bool UseCooFormat() const noexcept {
      return (sparse_flags & static_cast<int>(OrtSparseFlags::USE_COO_FORMAT)) != 0;
    }
    bool UseEllFormat() const noexcept {
      return (sparse_flags & static_cast<int>(OrtSparseFlags::USE_ELL_FORMAT)) != 0;
    }
  };

  // Override this function to PrePack initialized constant tensor to the format as needed.
  // For example, MatMul kernel can pack the input B if it is constant like code below.
  //   Status PrePack(const Tensor& tensor, const PrepackParam& param, bool& is_packed) override {
  //     is_packed = false;
  //     if (param.input_idx == 1) {
  //       if(param.Is2x4Format() && Verify2x4Format(tensor)) {
  //          // Compress and copy to GPU
  //       } else {
  //          this.Pack(tensor, this.buffer_);
  //        }
  //       is_packed = true;
  //     }
  //     return Status::OK();
  //   }
  // Please refer to MatMulIntegerToFloatBase for a complete example
  // @param tensor: The initialized constant tensor
  // @param input_idx: The input index of the tensor in this kernel
  // @param is_packed: Set it to true if the kernel packed the tensor or to false
  //                   The kernel is responsible for keeping the packed data and related metadata if is_packed is true,
  //                   And the original initialized constant tensor will be released and not accessible anymore in Compute function.
  virtual Status PrePack(const Tensor& /*tensor*/, const PrepackParam& /* param */, bool& is_packed) {
    is_packed = false;
    return Status::OK();
  }

  const OrtMemoryInfo& Allocator(int id, OrtMemType mem_type) const;
  const OpKernelInfo& Info() const { return *op_kernel_info_; }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OpKernel);
  std::unique_ptr<OpKernelInfo> op_kernel_info_;
};

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

  KernelCreateInfo() = default;
};

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

namespace featurizers {
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();
}  // namespace featurizers

namespace contrib {
namespace cuda {
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();
}  // namespace cuda
}  // namespace contrib

namespace contrib {
namespace rocm {
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();
}  // namespace rocm
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

#define ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type1, type2, name) \
  provider##_##name##_##domain##_ver##ver##_##type1##_##type2

#define ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(name, domain, ver, type1, type2, provider, builder, ...)                    \
  class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type1, type2, name);                         \
  template <>                                                                                                         \
  KernelCreateInfo                                                                                                    \
  BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type1, type2, name)>() {     \
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

#define ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(name, startver, endver, type, builder, ...)                         \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(name, kOnnxDomain, startver, endver, type, kCpuExecutionProvider, builder, \
                                          __VA_ARGS__)

#define ONNX_CPU_OPERATOR_VERSIONED_TYPED_ML_KERNEL(name, startver, endver, type, builder, ...)                    \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(name, kMLDomain, startver, endver, type, kCpuExecutionProvider, builder, \
                                          __VA_ARGS__)

#define ONNX_CPU_OPERATOR_VERSIONED_TYPED_MS_KERNEL(name, startver, endver, type, builder, ...)                    \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(name, kMSDomain, startver, endver, type, kCpuExecutionProvider, builder, \
                                          __VA_ARGS__)

#define ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(name, domain, startver, endver, type, provider, builder, ...)         \
  class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(provider, domain, startver, endver, type, name);              \
  template <>                                                                                                         \
  KernelCreateInfo                                                                                                    \
  BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(provider, domain, startver, endver,           \
                                                                        type, name)>() {                              \
    return KernelCreateInfo(                                                                                          \
        builder.SetName(#name)                                                                                        \
            .SetDomain(domain)                                                                                        \
            .SinceVersion(startver, endver)                                                                           \
            .Provider(provider)                                                                                       \
            .Build(),                                                                                                 \
        static_cast<KernelCreatePtrFn>([](const OpKernelInfo& info) -> OpKernel* { return new __VA_ARGS__(info); })); \
  }

#define ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(provider, domain, startver, endver, type1, type2, name) \
  provider##_##name##_##domain##_ver##startver##_##endver##_##type1##_##type2

#define ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_EX(name, domain, startver, endver, type1, type2,                     \
                                                    provider, builder, ...)                                           \
  class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(provider, domain, startver, endver, type1, type2, name);  \
  template <>                                                                                                         \
  KernelCreateInfo                                                                                                    \
  BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(provider, domain, startver, endver,       \
                                                                            type1, type2, name)>() {                  \
    return KernelCreateInfo(                                                                                          \
        builder.SetName(#name)                                                                                        \
            .SetDomain(domain)                                                                                        \
            .SinceVersion(startver, endver)                                                                           \
            .Provider(provider)                                                                                       \
            .Build(),                                                                                                 \
        static_cast<KernelCreatePtrFn>([](const OpKernelInfo& info) -> OpKernel* { return new __VA_ARGS__(info); })); \
  }

template <typename... Types>
struct BuildKernelDefConstraintsImpl {
  std::vector<MLDataType> operator()() const {
    return {DataTypeImpl::GetTensorType<Types>()...};
  }
};

// Use within macro definitions to create a custom vector of constraints.
// Example: #define REG_KERNEL(OP, VERSION, KERNEL_CLASS, Type, ...)
//  .TypeConstraint("T", BuildKernelDefConstraints<Type, __VA_ARGS_>())
template <typename... Types>
inline std::vector<MLDataType> BuildKernelDefConstraints() {
  return BuildKernelDefConstraintsImpl<Types...>{}();
}

// version of BuildKernelDefConstraints() which takes a type list
template <typename L>
std::vector<MLDataType> BuildKernelDefConstraintsFromTypeList() {
  return boost::mp11::mp_apply<BuildKernelDefConstraintsImpl, L>{}();
}

}  // namespace onnxruntime

#ifndef SHARED_PROVIDER
#include "core/framework/op_kernel_context.h"
#endif
