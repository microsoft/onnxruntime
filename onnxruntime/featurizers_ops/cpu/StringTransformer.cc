// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"

#include "Featurizers/StringFeaturizer.h"

namespace feat = Microsoft::Featurizer::Featurizers;

namespace onnxruntime {
namespace featurizers {

template <typename InputT>
class StringTransformer final : public OpKernel {
public:
  explicit StringTransformer(const OpKernelInfo &info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext *ctx) const override {
    // Create the transformer
    feat::StringTransformer<InputT> transformer(
      [ctx](void) {
        const auto * state_tensor(ctx->Input<Tensor>(0));
        const uint8_t * const state_data(state_tensor->Data<uint8_t>());

        Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().GetDims()[0]);
        return feat::StringTransformer<InputT>(archive);
      }()
    );

    // Get the input
    const auto* input_tensor(ctx->Input<Tensor>(1));
    const InputT * input_data(input_tensor->Data<InputT>());

    // Prepare the output
    Tensor * output_tensor(ctx->Output(0, input_tensor->Shape()));
    std::string * output_data(output_tensor->MutableData<std::string>());

    // Execute
    const int64_t length(input_tensor->Shape().Size());

    for(int64_t i = 0; i < length; ++i) {
      output_data[i] = transformer.execute(input_data[i]);
    }

    return Status::OK();
  }
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    StringTransformer,
    kMSFeaturizersDomain,
    1,
    int8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<int8_t>()),
    StringTransformer<int8_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    StringTransformer,
    kMSFeaturizersDomain,
    1,
    int16_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<int16_t>()),
    StringTransformer<int16_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    StringTransformer,
    kMSFeaturizersDomain,
    1,
    int32_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<int32_t>()),
    StringTransformer<int32_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    StringTransformer,
    kMSFeaturizersDomain,
    1,
    int64_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<int64_t>()),
    StringTransformer<int64_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    StringTransformer,
    kMSFeaturizersDomain,
    1,
    uint8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<uint8_t>()),
    StringTransformer<uint8_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    StringTransformer,
    kMSFeaturizersDomain,
    1,
    uint16_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<uint16_t>()),
    StringTransformer<uint16_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    StringTransformer,
    kMSFeaturizersDomain,
    1,
    uint32_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<uint32_t>()),
    StringTransformer<uint32_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    StringTransformer,
    kMSFeaturizersDomain,
    1,
    uint64_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<uint64_t>()),
    StringTransformer<uint64_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    StringTransformer,
    kMSFeaturizersDomain,
    1,
    float_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<float_t>()),
    StringTransformer<float_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    StringTransformer,
    kMSFeaturizersDomain,
    1,
    double_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<double_t>()),
    StringTransformer<double_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    StringTransformer,
    kMSFeaturizersDomain,
    1,
    bool,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<bool>()),
    StringTransformer<bool>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    StringTransformer,
    kMSFeaturizersDomain,
    1,
    string,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<std::string>()),
    StringTransformer<std::string>
);

} // namespace featurizers
} // namespace onnxruntime
