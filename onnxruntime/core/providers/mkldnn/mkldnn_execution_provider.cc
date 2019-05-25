// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mkldnn_execution_provider.h"
#include "core/framework/allocator.h"
#include "core/framework/memcpy.h"
#include "core/framework/kernel_registry.h"
#include "mkldnn_fwd.h"

namespace onnxruntime {

constexpr const char* MKLDNN = "MklDnn";
constexpr const char* MKLDNN_CPU = "MklDnnCpu";

namespace mkl_dnn {

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kMklDnnExecutionProvider,
    KernelDefBuilder().InputMemoryType<OrtMemTypeCPUInput>(0).TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kMklDnnExecutionProvider,
    KernelDefBuilder().OutputMemoryType<OrtMemTypeCPUOutput>(0).TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Memcpy);

}  // namespace mkl_dnn

MKLDNNExecutionProvider::MKLDNNExecutionProvider(const MKLDNNExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kMklDnnExecutionProvider} {
  DeviceAllocatorRegistrationInfo default_allocator_info({OrtMemTypeDefault,
                                                          [](int) { return std::make_unique<CPUAllocator>(std::make_unique<OrtAllocatorInfo>(MKLDNN, OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault)); }, std::numeric_limits<size_t>::max()});

  DeviceAllocatorRegistrationInfo cpu_allocator_info({OrtMemTypeCPUOutput,
                                                      [](int) { return std::make_unique<CPUAllocator>(std::make_unique<OrtAllocatorInfo>(MKLDNN_CPU, OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeCPUOutput)); }, std::numeric_limits<size_t>::max()});

  if (info.create_arena) {
    InsertAllocator(CreateAllocator(default_allocator_info));

    InsertAllocator(CreateAllocator(cpu_allocator_info));
  } else {
    InsertAllocator(std::shared_ptr<IArenaAllocator>(
        std::make_unique<DummyArena>(default_allocator_info.factory(0))));

    InsertAllocator(std::shared_ptr<IArenaAllocator>(
        std::make_unique<DummyArena>(cpu_allocator_info.factory(0))));
  }
}  // namespace onnxruntime

MKLDNNExecutionProvider::~MKLDNNExecutionProvider() {
}

Status MKLDNNExecutionProvider::CopyTensor(const Tensor& src, Tensor& dst) const {
  // Support CPU <-> MKLDNN for now
  if (!(strcmp(src.Location().name, MKLDNN) == 0 && strcmp(dst.Location().name, CPU) == 0) &&
      !(strcmp(src.Location().name, CPU) == 0 && strcmp(dst.Location().name, MKLDNN) == 0) &&
      !(strcmp(src.Location().name, MKLDNN) == 0 && strcmp(dst.Location().name, MKLDNN_CPU) == 0)) {
    ORT_NOT_IMPLEMENTED(src.Location().name, " copy to ", dst.Location().name, " is not implemented");
  }

  // Todo: Copy for now. May optimize later to avoid copy.
  size_t bytes = src.DataType()->Size() * src.Shape().Size();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();
  memcpy(dst_data, src_data, bytes);

  return Status::OK();
}

namespace mkl_dnn {
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, Conv);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 7, Gemm);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, MemcpyToHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 6, Relu);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 6, Sum);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 7, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 7, 8, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, 8, float, GlobalAveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 8, 8, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, 8, float, GlobalMaxPool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, float, LRN);

void RegisterMKLDNNKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 7, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 6, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 6, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 7, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 7, 8, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, 8, float, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 8, 8, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, 8, float, GlobalMaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, float, LRN)>,
  };

  for (auto& function_table_entry : function_table) {
    kernel_registry.Register(function_table_entry());
  }
}

std::shared_ptr<KernelRegistry> GetMklDnnKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  RegisterMKLDNNKernels(*kernel_registry);
  return kernel_registry;
}
}  // namespace mkl_dnn

std::shared_ptr<KernelRegistry> MKLDNNExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = onnxruntime::mkl_dnn::GetMklDnnKernelRegistry();
  return kernel_registry;
}
}  // namespace onnxruntime
