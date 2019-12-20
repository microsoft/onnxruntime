// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "featurizers_ops/cpu_featurizers_kernels.h"

#include "core/graph/constants.h"
#include "core/framework/data_types.h"

namespace onnxruntime {
namespace featurizers {

// Forward declarations
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, float, CatImputerTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, double, CatImputerTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, string, CatImputerTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, DateTimeTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int8, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int16, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint8, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint16, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, float, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int32, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int64, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint32, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint64, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, double, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int8, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int16, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int32, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int64, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint8, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint16, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint32, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint64, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, float, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, double, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, bool, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, string, StringTransformer);

Status RegisterCpuMSFeaturizersKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, float, CatImputerTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, double, CatImputerTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, string, CatImputerTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, DateTimeTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int8, MaxAbsScalarTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int16, MaxAbsScalarTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint8, MaxAbsScalarTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint16, MaxAbsScalarTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, float, MaxAbsScalarTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int32, MaxAbsScalarTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int64, MaxAbsScalarTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint32, MaxAbsScalarTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint64, MaxAbsScalarTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, double, MaxAbsScalarTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int8, StringTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int16, StringTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int32, StringTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int64, StringTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint8, StringTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint16, StringTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint32, StringTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint64, StringTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, float, StringTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, double, StringTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, bool, StringTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, string, StringTransformer)>};

  for (auto& function_table_entry : function_table) {
    ORT_RETURN_IF_ERROR(kernel_registry.Register(function_table_entry()));
  }

  return Status::OK();
}

}  // namespace featurizers
}  // namespace onnxruntime
