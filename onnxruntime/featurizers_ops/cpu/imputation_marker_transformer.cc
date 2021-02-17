// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"

#include "Featurizers/ImputationMarkerFeaturizer.h"
#include "Featurizers/../Archive.h"

namespace onnxruntime {
namespace featurizers {

inline float const& PreprocessOptional(float const& value) { return value; }
inline double const& PreprocessOptional(double const& value) { return value; }
inline nonstd::optional<std::string> PreprocessOptional(std::string value) {
  return value.empty() ? nonstd::optional<std::string>() : nonstd::optional<std::string>(std::move(value));
}

template <typename InputT>
struct ImputationMarkerTransformerImpl {
  void operator()(OpKernelContext* ctx) const {
    // Create the transformer
    Microsoft::Featurizer::Featurizers::ImputationMarkerTransformer<InputT> transformer(
        [ctx](void) {
          const auto* state_tensor(ctx->Input<Tensor>(0));
          const uint8_t* const state_data(state_tensor->Data<uint8_t>());

          Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().Size());
          return Microsoft::Featurizer::Featurizers::ImputationMarkerTransformer<InputT>(archive);
        }());

    // Get the input
    const auto* input_tensor(ctx->Input<Tensor>(1));
    const InputT* input_data(input_tensor->Data<InputT>());

    // Prepare the output
    Tensor* output_tensor(ctx->Output(0, input_tensor->Shape()));
    bool* output_data(output_tensor->MutableData<bool>());

    // Execute
    const int64_t length(input_tensor->Shape().Size());

    for (int64_t i = 0; i < length; ++i) {
      output_data[i] = transformer.execute(PreprocessOptional(input_data[i]));
    }
  }
};

class ImputationMarkerTransformer final : public OpKernel {
 public:
  explicit ImputationMarkerTransformer(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* ctx) const override {
    utils::MLTypeCallDispatcher<float, double, std::string> t_disp(ctx->Input<Tensor>(1)->GetElementType());
    t_disp.Invoke<ImputationMarkerTransformerImpl>(ctx);
    return Status::OK();
  }
};

ONNX_OPERATOR_KERNEL_EX(
    ImputationMarkerTransformer,
    kMSFeaturizersDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T0", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("InputT", {DataTypeImpl::GetTensorType<float>(),
                                   DataTypeImpl::GetTensorType<double>(),
                                   DataTypeImpl::GetTensorType<std::string>()}),
    ImputationMarkerTransformer);

}  // namespace featurizers
}  // namespace onnxruntime
