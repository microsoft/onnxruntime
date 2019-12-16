// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "automl_ops/cpu_automl_kernels.h"

#include "core/graph/constants.h"
#include "core/framework/data_types.h"

namespace onnxruntime {
namespace automl {

// Forward declarations
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, float_t, CatImputerTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, double_t, CatImputerTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, string, CatImputerTransformer);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, DateTimeTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int8_t, HashOneHotVectorizerTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int16_t, HashOneHotVectorizerTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int32_t, HashOneHotVectorizerTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int64_t, HashOneHotVectorizerTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint8_t, HashOneHotVectorizerTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint16_t, HashOneHotVectorizerTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint32_t, HashOneHotVectorizerTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint64_t, HashOneHotVectorizerTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, float_t, HashOneHotVectorizerTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, double_t, HashOneHotVectorizerTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, bool, HashOneHotVectorizerTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, string, HashOneHotVectorizerTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, float_t, ImputationMarkerTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, double_t, ImputationMarkerTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, string, ImputationMarkerTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int8_t, LabelEncoderTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int16_t, LabelEncoderTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int32_t, LabelEncoderTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int64_t, LabelEncoderTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint8_t, LabelEncoderTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint16_t, LabelEncoderTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint32_t, LabelEncoderTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint64_t, LabelEncoderTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, float_t, LabelEncoderTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, double_t, LabelEncoderTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, bool, LabelEncoderTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, string, LabelEncoderTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int8_t, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int16_t, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint8_t, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint16_t, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, float_t, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int32_t, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int64_t, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint32_t, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint64_t, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, double_t, MaxAbsScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int8_t, MinMaxScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int16_t, MinMaxScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int32_t, MinMaxScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int64_t, MinMaxScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint8_t, MinMaxScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint16_t, MinMaxScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint32_t, MinMaxScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint64_t, MinMaxScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, float_t, MinMaxScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, double_t, MinMaxScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, float_t, MissingDummiesTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, double_t, MissingDummiesTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, string, MissingDummiesTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int8_t, OneHotEncoderTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int16_t, OneHotEncoderTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int32_t, OneHotEncoderTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int64_t, OneHotEncoderTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint8_t, OneHotEncoderTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint16_t, OneHotEncoderTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint32_t, OneHotEncoderTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint64_t, OneHotEncoderTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, float_t, OneHotEncoderTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, double_t, OneHotEncoderTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, bool, OneHotEncoderTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, string, OneHotEncoderTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int8_t, RobustScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int16_t, RobustScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint8_t, RobustScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint16_t, RobustScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, float_t, RobustScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int32_t, RobustScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int64_t, RobustScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint32_t, RobustScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint64_t, RobustScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, double_t, RobustScalarTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int8_t, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int16_t, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int32_t, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int64_t, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint8_t, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint16_t, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint32_t, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint64_t, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, float_t, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, double_t, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, bool, StringTransformer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, string, StringTransformer);

Status RegisterCpuAutoMLKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, float_t, CatImputerTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, double_t, CatImputerTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, string, CatImputerTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, DateTimeTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int8_t, HashOneHotVectorizerTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int16_t, HashOneHotVectorizerTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int32_t, HashOneHotVectorizerTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int64_t, HashOneHotVectorizerTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint8_t, HashOneHotVectorizerTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint16_t, HashOneHotVectorizerTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint32_t, HashOneHotVectorizerTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint64_t, HashOneHotVectorizerTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, float_t, HashOneHotVectorizerTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, double_t, HashOneHotVectorizerTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, bool, HashOneHotVectorizerTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, string, HashOneHotVectorizerTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, float_t, ImputationMarkerTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, double_t, ImputationMarkerTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, string, ImputationMarkerTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int8_t, LabelEncoderTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int16_t, LabelEncoderTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int32_t, LabelEncoderTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int64_t, LabelEncoderTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint8_t, LabelEncoderTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint16_t, LabelEncoderTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint32_t, LabelEncoderTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint64_t, LabelEncoderTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, float_t, LabelEncoderTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, double_t, LabelEncoderTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, bool, LabelEncoderTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, string, LabelEncoderTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int8_t, MaxAbsScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int16_t, MaxAbsScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint8_t, MaxAbsScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint16_t, MaxAbsScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, float_t, MaxAbsScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int32_t, MaxAbsScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int64_t, MaxAbsScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint32_t, MaxAbsScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint64_t, MaxAbsScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, double_t, MaxAbsScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int8_t, MinMaxScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int16_t, MinMaxScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int32_t, MinMaxScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int64_t, MinMaxScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint8_t, MinMaxScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint16_t, MinMaxScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint32_t, MinMaxScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint64_t, MinMaxScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, float_t, MinMaxScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, double_t, MinMaxScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, float_t, MissingDummiesTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, double_t, MissingDummiesTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, string, MissingDummiesTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int8_t, OneHotEncoderTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int16_t, OneHotEncoderTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int32_t, OneHotEncoderTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int64_t, OneHotEncoderTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint8_t, OneHotEncoderTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint16_t, OneHotEncoderTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint32_t, OneHotEncoderTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint64_t, OneHotEncoderTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, float_t, OneHotEncoderTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, double_t, OneHotEncoderTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, bool, OneHotEncoderTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, string, OneHotEncoderTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int8_t, RobustScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int16_t, RobustScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint8_t, RobustScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint16_t, RobustScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, float_t, RobustScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int32_t, RobustScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int64_t, RobustScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint32_t, RobustScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint64_t, RobustScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, double_t, RobustScalarTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int8_t, StringTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int16_t, StringTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int32_t, StringTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, int64_t, StringTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint8_t, StringTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint16_t, StringTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint32_t, StringTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, uint64_t, StringTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, float_t, StringTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, double_t, StringTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, bool, StringTransformer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kMSAutoMLDomain, 1, string, StringTransformer)>
  };

  for (auto& function_table_entry : function_table) {
    ORT_RETURN_IF_ERROR(kernel_registry.Register(function_table_entry()));
  }

  return Status::OK();
}

} // namespace automl
} // namespace onnxruntime
