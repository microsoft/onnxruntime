// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"

#include "Featurizers/MedianImputerFeaturizer.h"
#include "Featurizers/../Archive.h"

namespace onnxruntime {
namespace featurizers {

template <typename T>
struct OutputTypeMapper {};
template <>
struct OutputTypeMapper<float> { using type = double; };
template <>
struct OutputTypeMapper<double> { using type = double; };
template <>
struct OutputTypeMapper<std::string> { using type = std::string; };

inline float const& PreprocessOptional(float const& value) { return value; }
inline double const& PreprocessOptional(double const& value) { return value; }
inline nonstd::optional<std::string> PreprocessOptional(std::string value) {
  return value.empty() ? nonstd::optional<std::string>() : nonstd::optional<std::string>(std::move(value));
}

namespace FeatDetails = Microsoft::Featurizer::Featurizers::Details;

template <typename T>
struct MedianImputerTransformerT : public FeatDetails::MedianImputerEstimatorImpl<T,
                                                                                  typename OutputTypeMapper<T>::type,
                                                                                  FeatDetails::ShouldInterpolateValues<T>::value,
                                                                                  std::numeric_limits<size_t>::max()>::TransformerType {
  using BaseType = typename FeatDetails::MedianImputerEstimatorImpl<T,
                                                                    typename OutputTypeMapper<T>::type,
                                                                    FeatDetails::ShouldInterpolateValues<T>::value,
                                                                    std::numeric_limits<size_t>::max()>::TransformerType;
  explicit MedianImputerTransformerT(Microsoft::Featurizer::Archive& ar) : BaseType(ar) {}
};

template <typename InputT>
struct MedianImputerTransformerImpl {
  void operator()(OpKernelContext* ctx) const {
    // Create the transformer
    MedianImputerTransformerT<InputT> transformer(
        [ctx](void) {
          const auto* state_tensor(ctx->Input<Tensor>(0));
          const uint8_t* const state_data(state_tensor->Data<uint8_t>());

          Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().Size());
          return MedianImputerTransformerT<InputT>(archive);
        }());

    // Get the input
    const auto* input_tensor(ctx->Input<Tensor>(1));
    const InputT* input_data(input_tensor->Data<InputT>());

    // Prepare the output
    Tensor* output_tensor(ctx->Output(0, input_tensor->Shape()));
    typename OutputTypeMapper<InputT>::type* output_data(output_tensor->MutableData<typename OutputTypeMapper<InputT>::type>());

    // Execute
    const int64_t length(input_tensor->Shape().Size());

    for (int64_t i = 0; i < length; ++i) {
      output_data[i] = transformer.execute(PreprocessOptional(input_data[i]));
    }
  }
};

class MedianImputerTransformer final : public OpKernel {
 public:
  explicit MedianImputerTransformer(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* ctx) const override {
    utils::MLTypeCallDispatcher<float, double, std::string> t_disp(ctx->Input<Tensor>(1)->GetElementType());
    t_disp.Invoke<MedianImputerTransformerImpl>(ctx);
    return Status::OK();
  }
};

ONNX_OPERATOR_KERNEL_EX(
    MedianImputerTransformer,
    kMSFeaturizersDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T0", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("InputT", {DataTypeImpl::GetTensorType<float>(),
                                   DataTypeImpl::GetTensorType<double>(),
                                   DataTypeImpl::GetTensorType<std::string>()}),
    MedianImputerTransformer);

}  // namespace featurizers
}  // namespace onnxruntime
