// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"

#include "Archive.h"
#include "Featurizers/OneHotEncoderFeaturizer.h"

namespace featurizers = Microsoft::Featurizer::Featurizers;

namespace onnxruntime {
namespace automl {

template <typename InputT>
class OneHotEncoderTransformer final : public OpKernel {
 public:
  explicit OneHotEncoderTransformer(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* ctx) const override {
    // Create the transformer
    featurizers::OneHotEncoderTransformer<InputT> transformer(
        [ctx](void) {
          const auto* state_tensor(ctx->Input<Tensor>(0));
          const uint8_t* const state_data(state_tensor->Data<uint8_t>());

          Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().GetDims()[0]);
          return featurizers::OneHotEncoderTransformer<InputT>(archive);
        }());

    // Get the input
    const auto* input_tensor(ctx->Input<Tensor>(1));
    const InputT* input_data(input_tensor->Data<InputT>());

    // Prepare the output
    Tensor* index_tensor(ctx->Output(0, input_tensor->Shape()));
    Tensor* size_tensor(ctx->Output(1, input_tensor->Shape()));
    Tensor* appearances_tensor(ctx->Output(2, input_tensor->Shape()));

    uint32_t* index_data(index_tensor->MutableData<uint32_t>());
    uint32_t* size_data(size_tensor->MutableData<uint32_t>());
    uint32_t* appearances_data(appearances_tensor->MutableData<uint32_t>());

    // Execute
    const int64_t length(input_tensor->Shape().Size());

    for (int64_t i = 0; i < length; ++i) {
      auto result(transformer.execute(input_data[i]));

      index_data[i] = std::move(result.index);
      size_data[i] = std::move(result.size);
      appearances_data[i] = std::move(result.appearances);
    }

    return Status::OK();
  }
};

#define ADD_TYPED_HOT_ENCODER_OP(data_type)                                     \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      OneHotEncoderTransformer,                                                 \
      kMSAutoMLDomain,                                                          \
      1,                                                                        \
      data_type,                                                                \
      kCpuExecutionProvider,                                                    \
      KernelDefBuilder()                                                        \
          .TypeConstraint("OutputT0", DataTypeImpl::GetTensorType<uint32_t>()), \
      OneHotEncoderTransformer<data_type>);

ADD_TYPED_HOT_ENCODER_OP(int8_t)
ADD_TYPED_HOT_ENCODER_OP(int16_t)
ADD_TYPED_HOT_ENCODER_OP(int32_t)
ADD_TYPED_HOT_ENCODER_OP(int64_t)

ADD_TYPED_HOT_ENCODER_OP(uint8_t)
ADD_TYPED_HOT_ENCODER_OP(uint16_t)
ADD_TYPED_HOT_ENCODER_OP(uint32_t)
ADD_TYPED_HOT_ENCODER_OP(uint64_t)

ADD_TYPED_HOT_ENCODER_OP(float)
ADD_TYPED_HOT_ENCODER_OP(double)
ADD_TYPED_HOT_ENCODER_OP(bool)
using string = std::string;
ADD_TYPED_HOT_ENCODER_OP(string)

}  // namespace automl
}  // namespace onnxruntime
