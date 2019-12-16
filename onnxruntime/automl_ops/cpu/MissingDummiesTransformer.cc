// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"

#include "Featurizers/MissingDummiesFeaturizer.h"

namespace featurizers = Microsoft::Featurizer::Featurizers;

namespace onnxruntime {
namespace automl {

inline float_t const & PreprocessOptional(float_t const &value) { return value; }
inline double_t const & PreprocessOptional(double_t const &value) { return value; }
inline nonstd::optional<string> PreprocessOptional(string value) { return value.empty() ? nonstd::optional<string>() : nonstd::optional<string>(std::move(value)); }

template <typename InputT>
class MissingDummiesTransformer final : public OpKernel {
public:
  explicit MissingDummiesTransformer(const OpKernelInfo &info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext *ctx) const override {
    // Create the transformer
    featurizers::MissingDummiesTransformer<InputT> transformer(
      [ctx](void) {
        const auto * state_tensor(ctx->Input<Tensor>(0));
        const uint8_t * const state_data(state_tensor->Data<uint8_t>());

        Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().GetDims()[0]);
        return featurizers::MissingDummiesTransformer<InputT>(archive);
      }()
    );

    // Get the input
    const auto* input_tensor(ctx->Input<Tensor>(1));
    const InputT * input_data(input_tensor->Data<InputT>());

    // Prepare the output
    Tensor * output_tensor(ctx->Output(0, input_tensor->Shape()));
    std::int8_t * output_data(output_tensor->MutableData<std::int8_t>());

    // Execute
    const int64_t length(input_tensor->Shape().Size());

    for(int64_t i = 0; i < length; ++i) {
      output_data[i] = transformer.execute(PreprocessOptional(input_data[i]));
    }

    return Status::OK();
  }
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MissingDummiesTransformer,
    kMSAutoMLDomain,
    1,
    float_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<float_t>()),
    MissingDummiesTransformer<float_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MissingDummiesTransformer,
    kMSAutoMLDomain,
    1,
    double_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<double_t>()),
    MissingDummiesTransformer<double_t>
);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MissingDummiesTransformer,
    kMSAutoMLDomain,
    1,
    string,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<std::string>()),
    MissingDummiesTransformer<std::string>
);

} // namespace automl
} // namespace onnxruntime
