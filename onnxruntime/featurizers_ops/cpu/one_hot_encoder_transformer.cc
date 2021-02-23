// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"

#include "Featurizers/OneHotEncoderFeaturizer.h"
#include "Featurizers/../Archive.h"

namespace onnxruntime {
namespace featurizers {

template <typename InputT>
struct OneHotEncoderTransformerImpl {
  void operator()(OpKernelContext* ctx) const {
    // Create the transformer
    Microsoft::Featurizer::Featurizers::OneHotEncoderTransformer<InputT> transformer(
        [ctx](void) {
          const auto* state_tensor(ctx->Input<Tensor>(0));
          const uint8_t* const state_data(state_tensor->Data<uint8_t>());

          Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().Size());
          return Microsoft::Featurizer::Featurizers::OneHotEncoderTransformer<InputT>(archive);
        }());

    // Get the input
    const auto* input_tensor(ctx->Input<Tensor>(1));
    const InputT* input_data(input_tensor->Data<InputT>());

    // Prepare the output
    Tensor* NumElements_tensor(ctx->Output(0, input_tensor->Shape()));
    Tensor* Value_tensor(ctx->Output(1, input_tensor->Shape()));
    Tensor* Index_tensor(ctx->Output(2, input_tensor->Shape()));

    uint64_t* NumElements_data(NumElements_tensor->MutableData<uint64_t>());
    uint8_t* Value_data(Value_tensor->MutableData<uint8_t>());
    uint64_t* Index_data(Index_tensor->MutableData<uint64_t>());

    // Execute
    const int64_t length(input_tensor->Shape().Size());

    for (int64_t i = 0; i < length; ++i) {
      auto result(transformer.execute(input_data[i]));

      NumElements_data[i] = std::move(result.NumElements);
      Value_data[i] = std::move(result.Value);
      Index_data[i] = std::move(result.Index);
    }
  }
};

class OneHotEncoderTransformer final : public OpKernel {
 public:
  explicit OneHotEncoderTransformer(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* ctx) const override {
    utils::MLTypeCallDispatcher<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t,
                                int64_t, uint64_t, float, double, bool, std::string>
        t_disp(ctx->Input<Tensor>(1)->GetElementType());
    t_disp.Invoke<OneHotEncoderTransformerImpl>(ctx);
    return Status::OK();
  }
};

ONNX_OPERATOR_KERNEL_EX(
    OneHotEncoderTransformer,
    kMSFeaturizersDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T0", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("InputT", {DataTypeImpl::GetTensorType<int8_t>(),
                                   DataTypeImpl::GetTensorType<uint8_t>(),
                                   DataTypeImpl::GetTensorType<int16_t>(),
                                   DataTypeImpl::GetTensorType<uint16_t>(),
                                   DataTypeImpl::GetTensorType<int32_t>(),
                                   DataTypeImpl::GetTensorType<uint32_t>(),
                                   DataTypeImpl::GetTensorType<int64_t>(),
                                   DataTypeImpl::GetTensorType<uint64_t>(),
                                   DataTypeImpl::GetTensorType<float>(),
                                   DataTypeImpl::GetTensorType<double>(),
                                   DataTypeImpl::GetTensorType<bool>(),
                                   DataTypeImpl::GetTensorType<std::string>()}),
    OneHotEncoderTransformer);

}  // namespace featurizers
}  // namespace onnxruntime
