// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "featurizers_ops/cpu_featurizers_kernels.h"

#include "core/graph/constants.h"
#include "core/framework/data_types.h"

namespace onnxruntime {
namespace featurizers {

// Forward declarations
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, CatImputerTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, DateTimeTransformer);

//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int8, HashOneHotVectorizerTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int16, HashOneHotVectorizerTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int32, HashOneHotVectorizerTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int64, HashOneHotVectorizerTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint8, HashOneHotVectorizerTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint16, HashOneHotVectorizerTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint32, HashOneHotVectorizerTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint64, HashOneHotVectorizerTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, float, HashOneHotVectorizerTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, double, HashOneHotVectorizerTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, bool, HashOneHotVectorizerTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, string, HashOneHotVectorizerTransformer);

//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, float, ImputationMarkerTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, double, ImputationMarkerTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, string, ImputationMarkerTransformer);

//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int8, LabelEncoderTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int16, LabelEncoderTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int32, LabelEncoderTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int64, LabelEncoderTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint8, LabelEncoderTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint16, LabelEncoderTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint32, LabelEncoderTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint64, LabelEncoderTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, float, LabelEncoderTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, double, LabelEncoderTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, bool, LabelEncoderTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, string, LabelEncoderTransformer);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, MaxAbsScalarTransformer);

//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int8, MinMaxScalarTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int16, MinMaxScalarTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int32, MinMaxScalarTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int64, MinMaxScalarTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint8, MinMaxScalarTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint16, MinMaxScalarTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint32, MinMaxScalarTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint64, MinMaxScalarTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, float, MinMaxScalarTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, double, MinMaxScalarTransformer);

//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, float, MissingDummiesTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, double, MissingDummiesTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, string, MissingDummiesTransformer);

//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int8, OneHotEncoderTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int16, OneHotEncoderTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int32, OneHotEncoderTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int64, OneHotEncoderTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint8, OneHotEncoderTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint16, OneHotEncoderTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint32, OneHotEncoderTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint64, OneHotEncoderTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, float, OneHotEncoderTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, double, OneHotEncoderTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, bool, OneHotEncoderTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, string, OneHotEncoderTransformer);

//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int8, RobustScalarTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int16, RobustScalarTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint8, RobustScalarTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint16, RobustScalarTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, float, RobustScalarTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int32, RobustScalarTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int64, RobustScalarTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint32, RobustScalarTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint64, RobustScalarTransformer);
//class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, double, RobustScalarTransformer);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, StringTransformer);

Status RegisterCpuMSFeaturizersKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, CatImputerTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, DateTimeTransformer)>,

    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int8, HashOneHotVectorizerTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int16, HashOneHotVectorizerTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int32, HashOneHotVectorizerTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int64, HashOneHotVectorizerTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint8, HashOneHotVectorizerTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint16, HashOneHotVectorizerTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint32, HashOneHotVectorizerTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint64, HashOneHotVectorizerTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, float, HashOneHotVectorizerTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, double, HashOneHotVectorizerTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, bool, HashOneHotVectorizerTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, string, HashOneHotVectorizerTransformer)>,

    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, float, ImputationMarkerTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, double, ImputationMarkerTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, string, ImputationMarkerTransformer)>,

    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int8, LabelEncoderTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int16, LabelEncoderTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int32, LabelEncoderTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int64, LabelEncoderTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint8, LabelEncoderTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint16, LabelEncoderTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint32, LabelEncoderTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint64, LabelEncoderTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, float, LabelEncoderTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, double, LabelEncoderTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, bool, LabelEncoderTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, string, LabelEncoderTransformer)>,

    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, MaxAbsScalarTransformer)>,

    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int8, MinMaxScalarTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int16, MinMaxScalarTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int32, MinMaxScalarTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int64, MinMaxScalarTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint8, MinMaxScalarTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint16, MinMaxScalarTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint32, MinMaxScalarTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint64, MinMaxScalarTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, float, MinMaxScalarTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, double, MinMaxScalarTransformer)>,

    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, float, MissingDummiesTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, double, MissingDummiesTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, string, MissingDummiesTransformer)>,

    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int8, OneHotEncoderTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int16, OneHotEncoderTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int32, OneHotEncoderTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int64, OneHotEncoderTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint8, OneHotEncoderTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint16, OneHotEncoderTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint32, OneHotEncoderTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint64, OneHotEncoderTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, float, OneHotEncoderTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, double, OneHotEncoderTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, bool, OneHotEncoderTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, string, OneHotEncoderTransformer)>,

    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int8, RobustScalarTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int16, RobustScalarTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint8, RobustScalarTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint16, RobustScalarTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, float, RobustScalarTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int32, RobustScalarTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, int64, RobustScalarTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint32, RobustScalarTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, uint64, RobustScalarTransformer)>,
    //BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, double, RobustScalarTransformer)>,

    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSFeaturizersDomain, 1, StringTransformer)>,
  };

  for (auto& function_table_entry : function_table) {
    ORT_RETURN_IF_ERROR(kernel_registry.Register(function_table_entry()));
  }

  return Status::OK();
}

}  // namespace featurizers
}  // namespace onnxruntime
