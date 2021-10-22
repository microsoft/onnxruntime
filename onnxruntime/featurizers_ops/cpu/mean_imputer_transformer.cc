// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"

#include "Featurizers/MeanImputerFeaturizer.h"
#include "Featurizers/../Archive.h"

namespace onnxruntime {
namespace featurizers {

inline float const& PreprocessOptional(float const& value) { return value; }
inline double const& PreprocessOptional(double const& value) { return value; }
inline nonstd::optional<std::string> PreprocessOptional(std::string value) {
  return value.empty() ? nonstd::optional<std::string>() : nonstd::optional<std::string>(std::move(value));
}

// Hack
namespace FeatDetails = Microsoft::Featurizer::Featurizers::Details;

template <typename T>
struct MeanImputerTransformerT : public FeatDetails::MeanImputerEstimatorImpl<T, std::numeric_limits<size_t>::max()>::TransformerType {
  using BaseType = typename FeatDetails::MeanImputerEstimatorImpl<T, std::numeric_limits<size_t>::max()>::TransformerType;
  explicit MeanImputerTransformerT(Microsoft::Featurizer::Archive& ar) : BaseType(ar) {}
};

template <typename InputT>
struct MeanImputerTransformerImpl {
  void operator()(OpKernelContext* ctx) const {
    // Create the transformer
    MeanImputerTransformerT<InputT> transformer(
        [ctx](void) {
          const auto* state_tensor(ctx->Input<Tensor>(0));
          const uint8_t* const state_data(state_tensor->Data<uint8_t>());

          Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().Size());
          return MeanImputerTransformerT<InputT>(archive);
        }());

    // Get the input
    const auto* input_tensor(ctx->Input<Tensor>(1));
    const InputT* input_data(input_tensor->Data<InputT>());

    // Prepare the output
    Tensor* output_tensor(ctx->Output(0, input_tensor->Shape()));
    double* output_data(output_tensor->MutableData<double>());

    // Execute
    const int64_t length(input_tensor->Shape().Size());

    for (int64_t i = 0; i < length; ++i) {
      output_data[i] = transformer.execute(PreprocessOptional(input_data[i]));
    }
  }
};

class MeanImputerTransformer final : public OpKernel {
 public:
  explicit MeanImputerTransformer(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* ctx) const override {
    utils::MLTypeCallDispatcher<float, double> t_disp(ctx->Input<Tensor>(1)->GetElementType());
    t_disp.Invoke<MeanImputerTransformerImpl>(ctx);
    return Status::OK();
  }
};

ONNX_OPERATOR_KERNEL_EX(
    MeanImputerTransformer,
    kMSFeaturizersDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T0", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("InputT", {DataTypeImpl::GetTensorType<float>(),
                                   DataTypeImpl::GetTensorType<double>()}),
    MeanImputerTransformer);

}  // namespace featurizers
}  // namespace onnxruntime
