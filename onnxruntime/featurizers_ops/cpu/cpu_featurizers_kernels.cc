// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "featurizers_ops/cpu/cpu_featurizers_kernels.h"

#include "core/graph/constants.h"
#include "core/framework/data_types.h"

namespace onnxruntime {
namespace featurizers {

// Forward declarations
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, AnalyticalRollingWindowTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, CatImputerTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, CountVectorizerTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, DateTimeTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, ForecastingPivotTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, FromStringTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, HashOneHotVectorizerTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, ImputationMarkerTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, LabelEncoderTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, LagLeadOperatorTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, MaxAbsScalerTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, MinMaxScalerTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, MeanImputerTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, MedianImputerTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, MinMaxImputerTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, MissingDummiesTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, ModeImputerTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, NormalizeTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, NumericalizeTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, OneHotEncoderTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, PCATransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, RobustScalerTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, ShortGrainDropperTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, SimpleRollingWindowTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, StandardScaleWrapperTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, StringTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, TfidfVectorizerTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, TimeSeriesImputerTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, TruncatedSVDTransformer);

Status RegisterCpuMSFeaturizersKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, AnalyticalRollingWindowTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, CatImputerTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, CountVectorizerTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, DateTimeTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, ForecastingPivotTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, FromStringTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, HashOneHotVectorizerTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, ImputationMarkerTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, LabelEncoderTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, LagLeadOperatorTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, MaxAbsScalerTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, MeanImputerTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, MedianImputerTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, MinMaxImputerTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, MinMaxScalerTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, MissingDummiesTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, ModeImputerTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, NormalizeTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, NumericalizeTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, OneHotEncoderTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, PCATransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, RobustScalerTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, ShortGrainDropperTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, SimpleRollingWindowTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, StandardScaleWrapperTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, StringTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, TfidfVectorizerTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, TimeSeriesImputerTransformer)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, TruncatedSVDTransformer)>
  };

  for (auto& function_table_entry : function_table) {
    ORT_RETURN_IF_ERROR(kernel_registry.Register(function_table_entry()));
  }

  return Status::OK();
}

}  // namespace featurizers
}  // namespace onnxruntime
