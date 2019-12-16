// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"

#include "Featurizers/MinMaxScalarFeaturizer.h"

namespace featurizers = Microsoft::Featurizer::Featurizers;

namespace onnxruntime {
namespace automl {

template <typename InputT>
class MinMaxScalarTransformer final : public OpKernel {
public:
  explicit MinMaxScalarTransformer(const OpKernelInfo &info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext *ctx) const override {
    // Create the transformer
    featurizers::MinMaxScalarTransformer<InputT> transformer(
      [ctx](void) {
        const auto * state_tensor(ctx->Input<Tensor>(0));
        const uint8_t * const state_data(state_tensor->Data<uint8_t>());

        Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().GetDims()[0]);
        return featurizers::MinMaxScalarTransformer<InputT>(archive);
      }()
    );

    // Get the input
    const auto* input_tensor(ctx->Input<Tensor>(1));
    const InputT * input_data(input_tensor->Data<InputT>());

    // Prepare the output
    Tensor * output_tensor(ctx->Output(0, input_tensor->Shape()));
    std::double_t * output_data(output_tensor->MutableData<std::double_t>());

    // Execute
    const int64_t length(input_tensor->Shape().Size());

    for(int64_t i = 0; i < length; ++i) {
      output_data[i] = transformer.execute(input_data[i]);
    }

    return Status::OK();
  }
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MinMaxScalarTransformer,
    kMSAutoMLDomain,
    1,
    int8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<int8_t>()),
    MinMaxScalarTransformer<int8_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MinMaxScalarTransformer,
    kMSAutoMLDomain,
    1,
    int16_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<int16_t>()),
    MinMaxScalarTransformer<int16_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MinMaxScalarTransformer,
    kMSAutoMLDomain,
    1,
    int32_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<int32_t>()),
    MinMaxScalarTransformer<int32_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MinMaxScalarTransformer,
    kMSAutoMLDomain,
    1,
    int64_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<int64_t>()),
    MinMaxScalarTransformer<int64_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MinMaxScalarTransformer,
    kMSAutoMLDomain,
    1,
    uint8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<uint8_t>()),
    MinMaxScalarTransformer<uint8_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MinMaxScalarTransformer,
    kMSAutoMLDomain,
    1,
    uint16_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<uint16_t>()),
    MinMaxScalarTransformer<uint16_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MinMaxScalarTransformer,
    kMSAutoMLDomain,
    1,
    uint32_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<uint32_t>()),
    MinMaxScalarTransformer<uint32_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MinMaxScalarTransformer,
    kMSAutoMLDomain,
    1,
    uint64_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<uint64_t>()),
    MinMaxScalarTransformer<uint64_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MinMaxScalarTransformer,
    kMSAutoMLDomain,
    1,
    float_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<float_t>()),
    MinMaxScalarTransformer<float_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MinMaxScalarTransformer,
    kMSAutoMLDomain,
    1,
    double_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<double_t>()),
    MinMaxScalarTransformer<double_t>
);

} // namespace automl
} // namespace onnxruntime
