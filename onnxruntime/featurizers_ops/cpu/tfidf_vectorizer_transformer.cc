// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

#include "Featurizers/TfidfVectorizerFeaturizer.h"
#include "Featurizers/../Archive.h"

namespace NS = Microsoft::Featurizer;

namespace onnxruntime {
namespace featurizers {

void TfidfVectorizerTransformerImpl(OpKernelContext* ctx) {
  // Create the transformer
  Microsoft::Featurizer::Featurizers::TfidfVectorizerTransformer transformer(
      [ctx]() {
        const auto* state_tensor(ctx->Input<Tensor>(0));
        const uint8_t* const state_data(state_tensor->Data<uint8_t>());

        Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().Size());
        return Microsoft::Featurizer::Featurizers::TfidfVectorizerTransformer(archive);
      }());

  // Get the input
  const auto* input_tensor = ctx->Input<Tensor>(1);
  const std::string* input_data = input_tensor->template Data<std::string>();

    // Prepare the callback that would output directly to output memory
  std::function<void(NS::Featurizers::SparseVectorEncoding<float>)> callback;
  bool callback_allow = true;
  callback = [ctx, callback_allow](NS::Featurizers::SparseVectorEncoding<float> result) {
    // Prepare output
    ORT_ENFORCE(callback_allow, "callback function can only be called during execute() and special flush() when needed");
    ORT_ENFORCE(result.NumElements < static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "NumElements in SparseVectorEncoding is GE than max(int64)");
    auto* output_tensor = ctx->Output(0, TensorShape{static_cast<int64_t>(result.NumElements)});
    float* output_data = output_tensor->template MutableData<float>();
    std::fill(output_data, output_data + result.NumElements, 0.f);
    for (const auto& el : result.Values) {
      output_data[el.Index] = el.Value;
    }
  };
  transformer.execute(*input_data, callback);
  // The flush() does nothing but shows Featurizers concept
  callback_allow = false;
  transformer.flush(callback);
}

class TfidfVectorizerTransformer final : public OpKernel {
 public:
  explicit TfidfVectorizerTransformer(const OpKernelInfo& info) : OpKernel(info) {
  }
  Status Compute(OpKernelContext* ctx) const override {
    TfidfVectorizerTransformerImpl(ctx);
    return Status::OK();
  }
};

ONNX_OPERATOR_KERNEL_EX(
    TfidfVectorizerTransformer,
    kMSFeaturizersDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T0", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<std::string>()),
    TfidfVectorizerTransformer);

}  // namespace featurizers
}  // namespace onnxruntime
