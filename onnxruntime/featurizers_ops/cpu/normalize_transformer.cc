// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"

#include "Featurizers/Base/NormalizeFeaturizer.h"
#include "Featurizers/../Archive.h"

namespace onnxruntime {
namespace featurizers {

template <typename InputT>
struct NormalizeTransformerImpl {
  void operator()(OpKernelContext* ctx) const {
    // Create the transformer
    using IterRangeT = std::pair<const InputT*, const InputT*>;
    using Transformer = Microsoft::Featurizer::Featurizers::Base::NormalizeTransformer<IterRangeT>;
    Transformer transformer(
        [ctx](void) {
          const auto* state_tensor(ctx->Input<Tensor>(0));
          const uint8_t* const state_data(state_tensor->Data<uint8_t>());

          Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().Size());
          return Transformer(archive);
        }());

    const auto* input_tensor(ctx->Input<Tensor>(1));
    const int64_t dim_num = input_tensor->Shape().NumDimensions();
    ORT_ENFORCE((dim_num == 1 || dim_num == 2), "Input 1 shape must have 1 or 2 dimensions");
    const auto rows = (dim_num == 1) ? 1 : input_tensor->Shape()[0];
    const auto row_size = (dim_num == 1) ? input_tensor->Shape()[0] : input_tensor->Shape()[1];

    const InputT* input_data(input_tensor->template Data<InputT>());

    // Read input directly from input tensor
    // TODO: Rework Normalize featurizers so they take length of input
    // TODO: Rework Normalize featurizers to output directly into output buffer
    auto* output_tensor(ctx->Output(0, input_tensor->Shape()));
    double* output_data(output_tensor->template MutableData<double>());

    std::vector<double> result;
    std::function<void(std::vector<double>)> callback;
    bool callback_allow = true;
    callback = [&result, callback_allow](std::vector<double> val) {
      ORT_ENFORCE(callback_allow, "callback function can only be called during execute() and special flush() when needed");
      result = std::move(val);
    };

    for (int64_t row = 0; row < rows; ++row) {
      auto row_begin = input_data + row * row_size;
      auto input_range = std::make_pair(row_begin, row_begin + row_size);
      result.clear();
      // Flush is not required here
      transformer.execute(input_range, callback);
      ORT_ENFORCE(static_cast<int64_t>(result.size()) == row_size,
                  "Expecting the same output size as input");
      std::copy(result.cbegin(), result.cend(), output_data);
      output_data += row_size;
    }
    // The flush() does nothing but shows Featurizers concept
    callback_allow = false;
    transformer.flush(callback);
  }
};

class NormalizeTransformer final : public OpKernel {
 public:
  explicit NormalizeTransformer(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* ctx) const override {
    utils::MLTypeCallDispatcher<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t,
                                int64_t, uint64_t, float, double>
        t_disp(ctx->Input<Tensor>(1)->GetElementType());
    t_disp.Invoke<NormalizeTransformerImpl>(ctx);
    return Status::OK();
  }
};

ONNX_OPERATOR_KERNEL_EX(
    NormalizeTransformer,
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
                                   DataTypeImpl::GetTensorType<double>()}),
    NormalizeTransformer);
}  // namespace featurizers
}  // namespace onnxruntime