// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"

#include "Featurizers/MaxAbsScalarFeaturizer.h"

namespace feat = Microsoft::Featurizer::Featurizers;

namespace onnxruntime {
namespace featurizers {

template <typename T> struct OutputTypeMapper {};
template <> struct OutputTypeMapper<int8_t> { using type = float_t; };
template <> struct OutputTypeMapper<int16_t> { using type = float_t; };
template <> struct OutputTypeMapper<uint8_t> { using type = float_t; };
template <> struct OutputTypeMapper<uint16_t> { using type = float_t; };
template <> struct OutputTypeMapper<float_t> { using type = float_t; };
template <> struct OutputTypeMapper<int32_t> { using type = double_t; };
template <> struct OutputTypeMapper<int64_t> { using type = double_t; };
template <> struct OutputTypeMapper<uint32_t> { using type = double_t; };
template <> struct OutputTypeMapper<uint64_t> { using type = double_t; };
template <> struct OutputTypeMapper<double_t> { using type = double_t; };

template <typename InputT>
class MaxAbsScalarTransformer final : public OpKernel {
public:
  explicit MaxAbsScalarTransformer(const OpKernelInfo &info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext *ctx) const override {
    // Create the transformer
    feat::MaxAbsScalarTransformer<InputT, typename OutputTypeMapper<InputT>::type> transformer(
      [ctx](void) {
        const auto * state_tensor(ctx->Input<Tensor>(0));
        const uint8_t * const state_data(state_tensor->Data<uint8_t>());

        Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().GetDims()[0]);
        return feat::MaxAbsScalarTransformer<InputT, typename OutputTypeMapper<InputT>::type>(archive);
      }()
    );

    // Get the input
    const auto* input_tensor(ctx->Input<Tensor>(1));
    const InputT * input_data(input_tensor->Data<InputT>());

    // Prepare the output
    Tensor * output_tensor(ctx->Output(0, input_tensor->Shape()));
    typename OutputTypeMapper<InputT>::type * output_data(output_tensor->MutableData<typename OutputTypeMapper<InputT>::type>());

    // Execute
    const int64_t length(input_tensor->Shape().Size());

    for(int64_t i = 0; i < length; ++i) {
      output_data[i] = transformer.execute(input_data[i]);
    }

    return Status::OK();
  }
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MaxAbsScalarTransformer,
    kMSFeaturizersDomain,
    1,
    int8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<int8_t>()),
    MaxAbsScalarTransformer<int8_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MaxAbsScalarTransformer,
    kMSFeaturizersDomain,
    1,
    int16_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<int16_t>()),
    MaxAbsScalarTransformer<int16_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MaxAbsScalarTransformer,
    kMSFeaturizersDomain,
    1,
    uint8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<uint8_t>()),
    MaxAbsScalarTransformer<uint8_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MaxAbsScalarTransformer,
    kMSFeaturizersDomain,
    1,
    uint16_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<uint16_t>()),
    MaxAbsScalarTransformer<uint16_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MaxAbsScalarTransformer,
    kMSFeaturizersDomain,
    1,
    float_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<float_t>()),
    MaxAbsScalarTransformer<float_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MaxAbsScalarTransformer,
    kMSFeaturizersDomain,
    1,
    int32_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<int32_t>()),
    MaxAbsScalarTransformer<int32_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MaxAbsScalarTransformer,
    kMSFeaturizersDomain,
    1,
    int64_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<int64_t>()),
    MaxAbsScalarTransformer<int64_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MaxAbsScalarTransformer,
    kMSFeaturizersDomain,
    1,
    uint32_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<uint32_t>()),
    MaxAbsScalarTransformer<uint32_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MaxAbsScalarTransformer,
    kMSFeaturizersDomain,
    1,
    uint64_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<uint64_t>()),
    MaxAbsScalarTransformer<uint64_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MaxAbsScalarTransformer,
    kMSFeaturizersDomain,
    1,
    double_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<double_t>()),
    MaxAbsScalarTransformer<double_t>
);

} // namespace featurizers
} // namespace onnxruntime
