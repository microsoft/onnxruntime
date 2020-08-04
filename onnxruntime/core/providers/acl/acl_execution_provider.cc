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
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 7, 9, Gemm);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, Conv);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 8, 11, float, MaxPool);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, 8, float, GlobalAveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, 8, float, GlobalMaxPool);

// Opset 10
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 10, 10, float, AveragePool);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 7, 9, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 4, 10, Concat);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kMSDomain, 1, float, FusedConv);

static void RegisterACLKernels(KernelRegistry& kernel_registry) {
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 6, Relu)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 7, 9, Gemm)>());

  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, Conv)>());

  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 8, 11, float, MaxPool)>());

  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, 8, float, GlobalAveragePool)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, 8, float, GlobalMaxPool)>());

  // Opset 10
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 10, 10, float, AveragePool)>());

  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 7, 9, BatchNormalization)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 4, 10, Concat)>());

  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kMSDomain, 1, float, FusedConv)>());
}

std::shared_ptr<KernelRegistry> GetAclKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  RegisterACLKernels(*kernel_registry);
  return kernel_registry;
}

}  // namespace acl

ACLExecutionProvider::ACLExecutionProvider(const ACLExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kAclExecutionProvider} {
  ORT_UNUSED_PARAMETER(info);

  DeviceAllocatorRegistrationInfo default_memory_info{
      OrtMemTypeDefault,
      [](int) {
        return onnxruntime::make_unique<CPUAllocator>(OrtMemoryInfo(ACL, OrtAllocatorType::OrtDeviceAllocator));
      },
      std::numeric_limits<size_t>::max()};

  InsertAllocator(CreateAllocator(default_memory_info, 0, info.create_arena));

  DeviceAllocatorRegistrationInfo cpu_memory_info{
      OrtMemTypeCPUOutput,
      [](int) {
        return onnxruntime::make_unique<CPUAllocator>(
            OrtMemoryInfo(ACL_CPU, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput));
      },
      std::numeric_limits<size_t>::max()};

  InsertAllocator(CreateAllocator(cpu_memory_info, 0, info.create_arena));
}

ACLExecutionProvider::~ACLExecutionProvider() {
}

std::shared_ptr<KernelRegistry> ACLExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = onnxruntime::acl::GetAclKernelRegistry();
  return kernel_registry;
}

std::vector<std::unique_ptr<ComputeCapability>>
ACLExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                    const std::vector<const KernelRegistry*>& kernel_registries) const {
  std::vector<std::unique_ptr<ComputeCapability>>
      result = IExecutionProvider::GetCapability(graph, kernel_registries);

  return result;
}

}  // namespace onnxruntime
