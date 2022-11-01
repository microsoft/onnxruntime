// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "js_execution_provider.h"

#include "core/graph/function_utils.h"
#include "core/framework/compute_capability.h"
#include "core/framework/kernel_registry.h"
#include "core/providers/shared/node_unit/node_unit.h"

#include "allocator.h"
#include "data_transfer.h"

namespace onnxruntime {

namespace js {
template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

#define KERNEL_CREATE_INFO_VERSIONED(Start, End, Op) \
  BuildKernelCreateInfo<                             \
      ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSInternalNHWCDomain, Start, End, Op)>

#define KERNEL_CREATE_INFO(Start, Op) \
  BuildKernelCreateInfo<              \
      ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSInternalNHWCDomain, Start, Op)>

#define KERNEL_CREATE_INFO_TYPED(Start, type, Op) \
  BuildKernelCreateInfo<                          \
      ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSInternalNHWCDomain, Start, type, Op)>

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 1, 14, Abs);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSInternalNHWCDomain, 1, 10, float, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSInternalNHWCDomain, 11, float, Conv);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 1, 10, float, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 11, float, Conv);

// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSInternalNHWCDomain, 11, Conv);
// class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSInternalNHWCDomain, 11, 11, MaxPool);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSInternalNHWCDomain, 12, MaxPool);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSInternalNHWCDomain, 11, AveragePool);
// class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 1, 12, Softmax);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 13, Softmax);

// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSInternalNHWCDomain, 10, uint8_t, QLinearConv);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSInternalNHWCDomain, 10, int8_t, QLinearConv);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSInternalNHWCDomain, 1, QLinearAveragePool);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider,
//                                       kDynamicDomainByCreate, 1, QLinearSoftmax);

std::unique_ptr<KernelRegistry> RegisterKernels() {
  auto kernel_registry = std::make_unique<onnxruntime::KernelRegistry>();

  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<void>,  // default entry to avoid the list becoming empty after ops-reducing
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 1, 14, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSInternalNHWCDomain, 1, 10, float, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kMSInternalNHWCDomain, 11, float, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 1, 10, float, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 11, float, Conv)>,
      // KERNEL_CREATE_INFO(11, Conv),
      // KERNEL_CREATE_INFO_VERSIONED(11, 11, MaxPool),
      // KERNEL_CREATE_INFO(12, MaxPool),
      // KERNEL_CREATE_INFO(11, AveragePool),
      // // layout insensitive, use ONNX-domain directly
      // BuildKernelCreateInfo<
      //     ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 13, Softmax)>,
      // BuildKernelCreateInfo<
      //     ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kJsExecutionProvider, kOnnxDomain, 1, 12, Softmax)>,

      // //  quantization op
      // KERNEL_CREATE_INFO_TYPED(10, uint8_t, QLinearConv),
      // KERNEL_CREATE_INFO_TYPED(10, int8_t, QLinearConv),
      // KERNEL_CREATE_INFO(1, QLinearAveragePool),
      // BuildKernelCreateInfo<
      //     ONNX_OPERATOR_KERNEL_CLASS_NAME(kJsExecutionProvider, kDynamicDomainByCreate, 1, QLinearSoftmax)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_THROW_IF_ERROR(kernel_registry->Register(std::move(info)));
    }
  }

  return kernel_registry;
}

}  // namespace js

using namespace js;

JsExecutionProvider::JsExecutionProvider(const JsExecutionProviderInfo& info)
    : IExecutionProvider{kJsExecutionProvider, true} {
}

// implement RegisterAllocator to test/validate sharing the CPU EP's allocator
void JsExecutionProvider::RegisterAllocator(AllocatorManager& allocator_manager) {
  AllocatorCreationInfo default_memory_info([&](int) { return std::make_unique<js::JsCPUAllocator>(); });

  AllocatorPtr default_allocator = CreateAllocator(default_memory_info);
  InsertAllocator(default_allocator);

  // use_arena might have some issue, for this to work need to change
  // https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/framework/execution_frame.cc#L507
  AllocatorCreationInfo memory_info(
      [&](int) { return std::make_unique<js::JsCustomAllocator>(); }, 0, false);

  AllocatorPtr allocator = CreateAllocator(memory_info);
  InsertAllocator(allocator);
}

std::vector<std::unique_ptr<ComputeCapability>> JsExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph,
    const IKernelLookup& kernel_lookup) const {

  return IExecutionProvider::GetCapability(graph, kernel_lookup);
}

std::shared_ptr<KernelRegistry> JsExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> registry = js::RegisterKernels();
  return registry;
}

std::unique_ptr<onnxruntime::IDataTransfer> JsExecutionProvider::GetDataTransfer() const {
  return std::make_unique<js::DataTransfer>();
}

JsExecutionProvider::~JsExecutionProvider() {
}

}  // namespace onnxruntime
