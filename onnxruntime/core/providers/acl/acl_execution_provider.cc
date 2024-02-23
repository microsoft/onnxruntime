// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019-2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#include "acl_execution_provider.h"
#include "core/framework/allocator.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"
#include "contrib_ops/cpu/cpu_contrib_kernels.h"
#include "acl_fwd.h"

namespace onnxruntime {

constexpr const char* ACL = "Acl";
constexpr const char* ACL_CPU = "AclCpu";

namespace acl {

// Forward declarations of op kernels
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 6, Relu);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 7, 8, Gemm);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 9, 10, Gemm);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 11, Gemm);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, Conv);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 8, 11, float, MaxPool);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 12, MaxPool);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, 8, float, GlobalAveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, 8, float, GlobalMaxPool);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 10, 10, float, AveragePool);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 11, AveragePool);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 7, 9, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 4, 10, Concat);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 11, Concat);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kMSDomain, 1, float, FusedConv);

Status RegisterACLKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      // BuildKernelCreateInfo<void>,  //default entry to avoid the list become empty after ops-reducing
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 6, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 7, 8, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 9, 10, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 11, Gemm)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, Conv)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 8, 11, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 12, MaxPool)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, 8, float, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, 8, float, GlobalMaxPool)>,

      // Opset 10
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 10, 10, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 11, AveragePool)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 7, 9, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 4, 10, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 11, Concat)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kMSDomain, 1, float, FusedConv)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

  return Status::OK();
}

std::shared_ptr<KernelRegistry> GetAclKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  ORT_THROW_IF_ERROR(RegisterACLKernels(*kernel_registry));
  return kernel_registry;
}

}  // namespace acl

ACLExecutionProvider::ACLExecutionProvider(const ACLExecutionProviderInfo&)
    : IExecutionProvider{onnxruntime::kAclExecutionProvider} {}

ACLExecutionProvider::~ACLExecutionProvider() {}

std::shared_ptr<KernelRegistry> ACLExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = onnxruntime::acl::GetAclKernelRegistry();
  return kernel_registry;
}

}  // namespace onnxruntime
