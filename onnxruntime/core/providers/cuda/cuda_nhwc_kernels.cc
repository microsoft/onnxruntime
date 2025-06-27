// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#ifdef ENABLE_CUDA_NHWC_OPS

#include <utility>

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_fwd.h"

#include "core/providers/cuda/cuda_nhwc_kernels.h"

// Macros to avoid long line length
#define CUDA_NHWC_OP_CLASS_NAME(ver, name) \
  ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, ver, name)
#define CUDA_NHWC_OP_TYPED_CLASS_NAME(ver, type, name) \
  ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, ver, type, name)
#define CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(start_ver, end_ver, type, name)                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, \
                                                  start_ver, end_ver, type, name)
#define CUDA_NHWC_OP_VERSIONED_CLASS_NAME(start_ver, end_ver, name) \
  ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSInternalNHWCDomain, start_ver, end_ver, name)

namespace onnxruntime::cuda {

// When adding new supported NHWC operations make sure to also integrate them into
// CUDAExecutionProvider::ShouldConvertNodeLayout()

class CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(7, 8, float, BatchNormalization);
class CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(7, 8, double, BatchNormalization);
class CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(7, 8, MLFloat16, BatchNormalization);
class CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(9, 13, float, BatchNormalization);
class CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(9, 13, double, BatchNormalization);
class CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(9, 13, MLFloat16, BatchNormalization);
class CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(1, 10, float, Conv);
class CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(1, 10, MLFloat16, Conv);
class CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(1, 10, float, ConvTranspose);
class CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(1, 10, MLFloat16, ConvTranspose);
class CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(7, 9, float, AveragePool);
class CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(7, 9, MLFloat16, AveragePool);
class CUDA_NHWC_OP_TYPED_CLASS_NAME(1, float, GlobalAveragePool);
class CUDA_NHWC_OP_TYPED_CLASS_NAME(1, MLFloat16, GlobalAveragePool);
class CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(1, 7, float, MaxPool);
class CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(1, 7, MLFloat16, MaxPool);
class CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(8, 9, float, MaxPool);
class CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(8, 9, MLFloat16, MaxPool);
class CUDA_NHWC_OP_TYPED_CLASS_NAME(1, float, GlobalMaxPool);
class CUDA_NHWC_OP_TYPED_CLASS_NAME(1, MLFloat16, GlobalMaxPool);
class CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(10, 10, float, AveragePool);
class CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(10, 10, MLFloat16, AveragePool);
class CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(10, 10, float, MaxPool);
class CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(10, 10, MLFloat16, MaxPool);
class CUDA_NHWC_OP_TYPED_CLASS_NAME(11, float, Conv);
class CUDA_NHWC_OP_TYPED_CLASS_NAME(11, MLFloat16, Conv);
class CUDA_NHWC_OP_TYPED_CLASS_NAME(11, float, ConvTranspose);
class CUDA_NHWC_OP_TYPED_CLASS_NAME(11, MLFloat16, ConvTranspose);
class CUDA_NHWC_OP_TYPED_CLASS_NAME(11, float, AveragePool);
class CUDA_NHWC_OP_TYPED_CLASS_NAME(11, MLFloat16, AveragePool);
class CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(11, 11, float, MaxPool);
class CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(11, 11, MLFloat16, MaxPool);
class CUDA_NHWC_OP_TYPED_CLASS_NAME(12, float, MaxPool);
class CUDA_NHWC_OP_TYPED_CLASS_NAME(12, MLFloat16, MaxPool);
class CUDA_NHWC_OP_TYPED_CLASS_NAME(12, int8_t, MaxPool);
class CUDA_NHWC_OP_TYPED_CLASS_NAME(12, uint8_t, MaxPool);
class CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(14, 14, float, BatchNormalization);
class CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(14, 14, double, BatchNormalization);
class CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(14, 14, MLFloat16, BatchNormalization);
class CUDA_NHWC_OP_TYPED_CLASS_NAME(15, float, BatchNormalization);
class CUDA_NHWC_OP_TYPED_CLASS_NAME(15, double, BatchNormalization);
class CUDA_NHWC_OP_TYPED_CLASS_NAME(15, MLFloat16, BatchNormalization);
class CUDA_NHWC_OP_VERSIONED_CLASS_NAME(1, 10, DepthToSpace);
class CUDA_NHWC_OP_VERSIONED_CLASS_NAME(11, 12, DepthToSpace);
class CUDA_NHWC_OP_CLASS_NAME(13, DepthToSpace);
class CUDA_NHWC_OP_VERSIONED_CLASS_NAME(1, 12, SpaceToDepth);
class CUDA_NHWC_OP_CLASS_NAME(13, SpaceToDepth);
class CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(1, 12, float, LRN);
class CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(1, 12, double, LRN);
class CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(1, 12, MLFloat16, LRN);
class CUDA_NHWC_OP_TYPED_CLASS_NAME(13, float, LRN);
class CUDA_NHWC_OP_TYPED_CLASS_NAME(13, double, LRN);
class CUDA_NHWC_OP_TYPED_CLASS_NAME(13, MLFloat16, LRN);

Status RegisterCudaNhwcKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn nhwc_function_table[] = {
      BuildKernelCreateInfo<void>,  // default entry to avoid the list become empty after ops-reducing
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(7, 8, MLFloat16, BatchNormalization)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(7, 8, float, BatchNormalization)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(7, 8, double, BatchNormalization)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(9, 13, MLFloat16, BatchNormalization)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(9, 13, float, BatchNormalization)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(9, 13, double, BatchNormalization)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(14, 14, MLFloat16, BatchNormalization)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(14, 14, float, BatchNormalization)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(14, 14, double, BatchNormalization)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_TYPED_CLASS_NAME(15, MLFloat16, BatchNormalization)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_TYPED_CLASS_NAME(15, float, BatchNormalization)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_TYPED_CLASS_NAME(15, double, BatchNormalization)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(1, 10, MLFloat16, Conv)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(1, 10, float, Conv)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_TYPED_CLASS_NAME(11, float, Conv)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_TYPED_CLASS_NAME(11, MLFloat16, Conv)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(7, 9, float, AveragePool)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(7, 9, MLFloat16, AveragePool)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_TYPED_CLASS_NAME(1, float, GlobalAveragePool)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_TYPED_CLASS_NAME(1, MLFloat16, GlobalAveragePool)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(1, 7, float, MaxPool)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(1, 7, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(8, 9, float, MaxPool)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(8, 9, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_TYPED_CLASS_NAME(1, float, GlobalMaxPool)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_TYPED_CLASS_NAME(1, MLFloat16, GlobalMaxPool)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(10, 10, float, AveragePool)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(10, 10, MLFloat16, AveragePool)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(10, 10, float, MaxPool)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(10, 10, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_TYPED_CLASS_NAME(11, float, AveragePool)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_TYPED_CLASS_NAME(11, MLFloat16, AveragePool)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(11, 11, float, MaxPool)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(11, 11, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_TYPED_CLASS_NAME(12, float, MaxPool)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_TYPED_CLASS_NAME(12, MLFloat16, MaxPool)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_TYPED_CLASS_NAME(12, int8_t, MaxPool)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_TYPED_CLASS_NAME(12, uint8_t, MaxPool)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_TYPED_CLASS_NAME(11, float, ConvTranspose)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_TYPED_CLASS_NAME(11, MLFloat16, ConvTranspose)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(1, 10, float, ConvTranspose)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(1, 10, MLFloat16, ConvTranspose)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_CLASS_NAME(1, 10, DepthToSpace)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_CLASS_NAME(11, 12, DepthToSpace)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_CLASS_NAME(13, DepthToSpace)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_CLASS_NAME(1, 12, SpaceToDepth)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_CLASS_NAME(13, SpaceToDepth)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(1, 12, float, LRN)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(1, 12, double, LRN)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_VERSIONED_TYPED_CLASS_NAME(1, 12, MLFloat16, LRN)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_TYPED_CLASS_NAME(13, float, LRN)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_TYPED_CLASS_NAME(13, double, LRN)>,
      BuildKernelCreateInfo<CUDA_NHWC_OP_TYPED_CLASS_NAME(13, MLFloat16, LRN)>,
  };

  for (auto& function_table_entry : nhwc_function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }
  return Status::OK();
}
}  // namespace onnxruntime::cuda

#ifndef DISABLE_CONTRIB_OPS
namespace onnxruntime::contrib::cuda {

class CUDA_NHWC_OP_TYPED_CLASS_NAME(16, float, GridSample);

onnxruntime::common::Status RegisterCudaNhwcContribKernels(onnxruntime::KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn nhwc_function_table[] = {
      BuildKernelCreateInfo<void>,  // default entry to avoid the list become empty after ops-reducing
      BuildKernelCreateInfo<CUDA_NHWC_OP_TYPED_CLASS_NAME(16, float, GridSample)>,
  };

  for (auto& function_table_entry : nhwc_function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }
  return Status::OK();
}

}  // namespace onnxruntime::contrib::cuda
#endif

#endif
