// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"

#include "Featurizers/CatImputerFeaturizer.h"

namespace feat = Microsoft::Featurizer::Featurizers;

namespace onnxruntime {
namespace featurizers {

template <typename T>
struct OutputTypeMapper {};
template <>
struct OutputTypeMapper<float_t> { using type = float_t; };
template <>
struct OutputTypeMapper<double_t> { using type = double_t; };
template <>
struct OutputTypeMapper<std::string> { using type = std::string; };

inline float_t const& PreprocessOptional(float_t const& value) { return value; }
inline double_t const& PreprocessOptional(double_t const& value) { return value; }
inline nonstd::optional<std::string> PreprocessOptional(std::string value) {
  return value.empty() ? nonstd::optional<std::string>() : nonstd::optional<std::string>(std::move(value));
}

template <typename T>
class CatImputerTransformer final : public OpKernel {
 public:
  explicit CatImputerTransformer(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* ctx) const override {
    // Create the transformer
    feat::CatImputerTransformer<T> transformer(
        [ctx](void) {
          const auto* state_tensor(ctx->Input<Tensor>(0));
          const uint8_t* const state_data(state_tensor->Data<uint8_t>());

          Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().GetDims()[0]);
          return feat::CatImputerTransformer<T>(archive);
        }());

    // Get the input
    const auto* input_tensor(ctx->Input<Tensor>(1));
    const T* input_data(input_tensor->Data<T>());

    // Prepare the output
    Tensor* output_tensor(ctx->Output(0, input_tensor->Shape()));
    typename OutputTypeMapper<T>::type* output_data(output_tensor->MutableData<typename OutputTypeMapper<T>::type>());

    // Execute
    const int64_t length(input_tensor->Shape().Size());

    for (int64_t i = 0; i < length; ++i) {
      output_data[i] = transformer.execute(PreprocessOptional(input_data[i]));
    }

    return Status::OK();
  }
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    CatImputerTransformer,
    kMSFeaturizersDomain,
    1,
    float_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float_t>()),
    CatImputerTransformer<float_t>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    CatImputerTransformer,
    kMSFeaturizersDomain,
    1,
    double_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<double_t>()),
    CatImputerTransformer<double_t>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    CatImputerTransformer,
    kMSFeaturizersDomain,
    1,
    string,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<std::string>()),
    CatImputerTransformer<std::string>);

}  // namespace featurizers
}  // namespace onnxruntime
