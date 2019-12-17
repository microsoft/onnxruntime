// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"

#include "Archive.h"
#include "Featurizers/LabelEncoderFeaturizer.h"

namespace featurizers = Microsoft::Featurizer::Featurizers;

namespace onnxruntime {
namespace automl {

template <typename InputT>
class LabelEncoderTransformer final : public OpKernel {
public:
  explicit LabelEncoderTransformer(const OpKernelInfo &info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext *ctx) const override {
    // Create the transformer
    featurizers::LabelEncoderTransformer<InputT> transformer(
      [ctx](void) {
        const auto * state_tensor(ctx->Input<Tensor>(0));
        const uint8_t * const state_data(state_tensor->Data<uint8_t>());

        Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().GetDims()[0]);
        return featurizers::LabelEncoderTransformer<InputT>(archive);
      }()
    );

    // Get the input
    const auto* input_tensor(ctx->Input<Tensor>(1));
    const InputT * input_data(input_tensor->Data<InputT>());

    // Prepare the output
    Tensor * output_tensor(ctx->Output(0, input_tensor->Shape()));
    std::uint32_t * output_data(output_tensor->MutableData<std::uint32_t>());

    // Execute
    const int64_t length(input_tensor->Shape().Size());

    for(int64_t i = 0; i < length; ++i) {
      output_data[i] = transformer.execute(input_data[i]);
    }

    return Status::OK();
  }
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    LabelEncoderTransformer,
    kMSAutoMLDomain,
    1,
    int8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<int8_t>()),
    LabelEncoderTransformer<int8_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    LabelEncoderTransformer,
    kMSAutoMLDomain,
    1,
    int16_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<int16_t>()),
    LabelEncoderTransformer<int16_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    LabelEncoderTransformer,
    kMSAutoMLDomain,
    1,
    int32_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<int32_t>()),
    LabelEncoderTransformer<int32_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    LabelEncoderTransformer,
    kMSAutoMLDomain,
    1,
    int64_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<int64_t>()),
    LabelEncoderTransformer<int64_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    LabelEncoderTransformer,
    kMSAutoMLDomain,
    1,
    uint8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<uint8_t>()),
    LabelEncoderTransformer<uint8_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    LabelEncoderTransformer,
    kMSAutoMLDomain,
    1,
    uint16_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<uint16_t>()),
    LabelEncoderTransformer<uint16_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    LabelEncoderTransformer,
    kMSAutoMLDomain,
    1,
    uint32_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<uint32_t>()),
    LabelEncoderTransformer<uint32_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    LabelEncoderTransformer,
    kMSAutoMLDomain,
    1,
    uint64_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<uint64_t>()),
    LabelEncoderTransformer<uint64_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    LabelEncoderTransformer,
    kMSAutoMLDomain,
    1,
    float_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<float_t>()),
    LabelEncoderTransformer<float_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    LabelEncoderTransformer,
    kMSAutoMLDomain,
    1,
    double_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<double_t>()),
    LabelEncoderTransformer<double_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    LabelEncoderTransformer,
    kMSAutoMLDomain,
    1,
    bool,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<bool>()),
    LabelEncoderTransformer<bool>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    LabelEncoderTransformer,
    kMSAutoMLDomain,
    1,
    string,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<std::string>()),
    LabelEncoderTransformer<std::string>
);

} // namespace automl
} // namespace onnxruntime
