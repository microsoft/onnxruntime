// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "featurizers_ops/cpu_featurizers_kernels.h"

#include "core/graph/constants.h"
#include "core/framework/data_types.h"

namespace onnxruntime {
namespace featurizers {

// Forward declarations
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, float_t, CatImputerTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, double_t, CatImputerTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, string, CatImputerTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, DateTimeTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int8_t, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int16_t, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint8_t, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint16_t, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, float_t, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int32_t, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int64_t, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint32_t, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint64_t, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, double_t, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int8_t, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int16_t, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int32_t, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int64_t, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint8_t, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint16_t, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint32_t, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint64_t, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, float_t, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, double_t, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, bool, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, string, StringTransformer);

Status RegisterCpuAutoMLKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, float_t, CatImputerTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, double_t, CatImputerTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, string, CatImputerTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, DateTimeTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int8_t, MaxAbsScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int16_t, MaxAbsScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint8_t, MaxAbsScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint16_t, MaxAbsScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, float_t, MaxAbsScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int32_t, MaxAbsScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int64_t, MaxAbsScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint32_t, MaxAbsScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint64_t, MaxAbsScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, double_t, MaxAbsScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int8_t, StringTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int16_t, StringTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int32_t, StringTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int64_t, StringTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint8_t, StringTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint16_t, StringTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint32_t, StringTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint64_t, StringTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, float_t, StringTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, double_t, StringTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, bool, StringTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, string, StringTransformer)>
  };

  for (auto& function_table_entry : function_table) {
    ORT_RETURN_IF_ERROR(kernel_registry.Register(function_table_entry()));
  }

  return Status::OK();
}

} // namespace featurizers
} // namespace onnxruntime
