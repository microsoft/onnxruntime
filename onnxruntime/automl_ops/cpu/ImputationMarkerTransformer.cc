// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"

#include "Featurizers/ImputationMarkerFeaturizer.h"

namespace featurizers = Microsoft::Featurizer::Featurizers;

namespace onnxruntime {
namespace automl {

inline float_t const & PreprocessOptional(float_t const &value) { return value; }
inline double_t const & PreprocessOptional(double_t const &value) { return value; }
inline nonstd::optional<std::string> PreprocessOptional(std::string value) { 
  return value.empty() ? nonstd::optional<std::string>() : nonstd::optional<std::string>(std::move(value));
}

template <typename InputT>
class ImputationMarkerTransformer final : public OpKernel {
public:
  explicit ImputationMarkerTransformer(const OpKernelInfo &info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext *ctx) const override {
    // Create the transformer
    featurizers::ImputationMarkerTransformer<InputT> transformer(
      [ctx](void) {
        const auto * state_tensor(ctx->Input<Tensor>(0));
        const uint8_t * const state_data(state_tensor->Data<uint8_t>());

        Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().GetDims()[0]);
        return featurizers::ImputationMarkerTransformer<InputT>(archive);
      }()
    );

    // Get the input
    const auto* input_tensor(ctx->Input<Tensor>(1));
    const InputT * input_data(input_tensor->Data<InputT>());

    // Prepare the output
    Tensor * output_tensor(ctx->Output(0, input_tensor->Shape()));
    bool * output_data(output_tensor->MutableData<bool>());

    // Execute
    const int64_t length(input_tensor->Shape().Size());

    for(int64_t i = 0; i < length; ++i) {
      output_data[i] = transformer.execute(PreprocessOptional(input_data[i]));
    }

    return Status::OK();
  }
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    ImputationMarkerTransformer,
    kMSAutoMLDomain,
    1,
    float_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<float_t>()),
    ImputationMarkerTransformer<float_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    ImputationMarkerTransformer,
    kMSAutoMLDomain,
    1,
    double_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<double_t>()),
    ImputationMarkerTransformer<double_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    ImputationMarkerTransformer,
    kMSAutoMLDomain,
    1,
    string,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<std::string>()),
    ImputationMarkerTransformer<std::string>
);

} // namespace automl
} // namespace onnxruntime
