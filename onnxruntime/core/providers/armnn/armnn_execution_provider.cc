// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#include "armnn_execution_provider.h"
#include "core/framework/allocator.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"
#include "contrib_ops/cpu/cpu_contrib_kernels.h"
#include "armnn_fwd.h"

namespace onnxruntime {

constexpr const char* ArmNN = "ArmNN";
constexpr const char* ArmNN_CPU = "ArmNNCpu";

namespace armnn_ep {

// Forward declarations of op kernels
#ifdef RELU_ARMNN
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 6, Relu);
#endif
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 1, 10, Conv);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 11, Conv);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 7, 8, Gemm);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 9, 10, Gemm);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 11, Gemm);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 10, 10, float, AveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 11, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 8, 11, float, MaxPool);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 1, float, GlobalAveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 1, float, GlobalMaxPool);

#ifdef BN_ARMNN
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 7, 9, BatchNormalization);
#endif
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 4, 10, Concat);

template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

static void RegisterArmNNKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<void>, //default entry to avoid the list become empty after ops-reducing
#ifdef RELU_ARMNN
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 6, Relu)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 1, 10, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 11, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 7, 8, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 9, 10, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 11, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 10, 10, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 11, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 8, 11, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 1, float, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 1, float, GlobalMaxPool)>,
#ifdef BN_ARMNN
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 7, 9, BatchNormalization)>,
#endif
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kArmNNExecutionProvider, kOnnxDomain, 4, 10, Concat)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_ENFORCE(kernel_registry.Register(std::move(info)).IsOK());
    }
  }
}

std::shared_ptr<KernelRegistry> GetArmNNKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  RegisterArmNNKernels(*kernel_registry);

  return kernel_registry;
}

}  // namespace armnn_ep

ArmNNExecutionProvider::ArmNNExecutionProvider(const ArmNNExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kArmNNExecutionProvider} {
  ORT_UNUSED_PARAMETER(info);

  DeviceAllocatorRegistrationInfo default_memory_info{
      OrtMemTypeDefault,
      [](int) {
        return onnxruntime::make_unique<CPUAllocator>(OrtMemoryInfo(ArmNN, OrtAllocatorType::OrtDeviceAllocator));
      },
      std::numeric_limits<size_t>::max()};

  InsertAllocator(CreateAllocator(default_memory_info));

  DeviceAllocatorRegistrationInfo cpu_memory_info{
      OrtMemTypeCPUOutput,
      [](int) {
        return onnxruntime::make_unique<CPUAllocator>(
            OrtMemoryInfo(ArmNN_CPU, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput));
      },
      std::numeric_limits<size_t>::max()};

  InsertAllocator(CreateAllocator(cpu_memory_info));
}

ArmNNExecutionProvider::~ArmNNExecutionProvider() {
}

std::shared_ptr<KernelRegistry> ArmNNExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = onnxruntime::armnn_ep::GetArmNNKernelRegistry();
  return kernel_registry;
}

std::vector<std::unique_ptr<ComputeCapability>>
ArmNNExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                    const std::vector<const KernelRegistry*>& kernel_registries) const {
  std::vector<std::unique_ptr<ComputeCapability>>
      result = IExecutionProvider::GetCapability(graph, kernel_registries);

  return result;
}

}  // namespace onnxruntime
