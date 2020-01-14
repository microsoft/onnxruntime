// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#include "acl_execution_provider.h"
#include "core/framework/allocator.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"
#include "contrib_ops/cpu_contrib_kernels.h"
#include "acl_fwd.h"

namespace onnxruntime {

constexpr const char* ACL = "Acl";
constexpr const char* ACL_CPU = "AclCpu";

namespace acl {

// Forward declarations of op kernels
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 6, Relu);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 7, 9, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, 9, float, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, 9, double, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 9, 9, int32_t, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 9, 9, uint32_t, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 9, 9, int64_t, MatMul);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 9, 9, uint64_t, MatMul);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, Conv);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 8, 9, float, MaxPool);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, 8, float, GlobalAveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, 8, float, GlobalMaxPool);

// Opset 10
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 10, 10, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 10, 10, float, AveragePool);

static void RegisterACLKernels(KernelRegistry& kernel_registry) {
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 6, Relu)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 7, 9, Gemm)>());

  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, 9, float, MatMul)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, 9, double, MatMul)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 9, 9, int32_t, MatMul)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 9, 9, uint32_t, MatMul)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 9, 9, int64_t, MatMul)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 9, 9, uint64_t, MatMul)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, Conv)>());

  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 8, 9, float, MaxPool)>());

  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, 8, float, GlobalAveragePool)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, 8, float, GlobalMaxPool)>());

  // Opset 10
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 10, 10, float, MaxPool)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 10, 10, float, AveragePool)>());
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

  auto default_allocator_factory = [](int) {
    auto memory_info = onnxruntime::make_unique<OrtMemoryInfo>(ACL, OrtAllocatorType::OrtDeviceAllocator);
    return onnxruntime::make_unique<CPUAllocator>(std::move(memory_info));
  };

  DeviceAllocatorRegistrationInfo default_memory_info{
      OrtMemTypeDefault,
      std::move(default_allocator_factory),
      std::numeric_limits<size_t>::max()};

  InsertAllocator(CreateAllocator(default_memory_info));

  auto cpu_allocator_factory = [](int) {
    auto memory_info = onnxruntime::make_unique<OrtMemoryInfo>(
        ACL_CPU, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput);
    return onnxruntime::make_unique<CPUAllocator>(std::move(memory_info));
  };

  DeviceAllocatorRegistrationInfo cpu_memory_info{
      OrtMemTypeCPUOutput,
      std::move(cpu_allocator_factory),
      std::numeric_limits<size_t>::max()};

  InsertAllocator(CreateAllocator(cpu_memory_info));
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
