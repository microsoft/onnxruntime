// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"

#include "Featurizers/FromStringFeaturizer.h"
#include "Featurizers/../Archive.h"

namespace onnxruntime {
namespace featurizers {

template <typename T>
struct FromStringTransformerImpl {
  void operator()(OpKernelContext* ctx) const {
    // Create the transformer
    Microsoft::Featurizer::Featurizers::FromStringTransformer<T> transformer(
        [ctx]() {
          const auto* state_tensor(ctx->Input<Tensor>(0));
          const uint8_t* const state_data(state_tensor->Data<uint8_t>());

          Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().Size());
          return Microsoft::Featurizer::Featurizers::FromStringTransformer<T>(archive);
        }());

    const auto* input_tensor(ctx->Input<Tensor>(1));
    const std::string* input_data(input_tensor->Data<std::string>());

    // Prepare the output
    Tensor* output_tensor(ctx->Output(0, input_tensor->Shape()));
    T* output_data(output_tensor->MutableData<T>());

    // Execute
    const int64_t length(input_tensor->Shape().Size());
    for (int64_t i = 0; i < length; ++i) {
      output_data[i] = transformer.execute(input_data[i]);
    }
  }
};

template <>
struct FromStringTransformerImpl<std::string> {
  void operator()(OpKernelContext* ctx) const {
    const auto* input_tensor(ctx->Input<Tensor>(1));
    const std::string* input_data(input_tensor->Data<std::string>());
    const int64_t num_items = input_tensor->Shape().Size();

    // Prepare the output
    Tensor* output_tensor(ctx->Output(0, input_tensor->Shape()));
    std::string* output_data(output_tensor->MutableData<std::string>());
    std::copy(input_data, input_data + num_items, output_data);
  }
};

class FromStringTransformer final : public OpKernel {
 public:
  explicit FromStringTransformer(const OpKernelInfo& info) : OpKernel(info),
                                                             result_type_(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UNDEFINED) {
    int64_t result_type;
    ORT_ENFORCE(info.GetAttr<int64_t>("result_type", &result_type).IsOK(), "result_type is a mandatory attribute");
    ORT_ENFORCE(ONNX_NAMESPACE::TensorProto::DataType_IsValid(static_cast<int>(result_type)), "Invalid result_type value");
    result_type_ = static_cast<ONNX_NAMESPACE::TensorProto::DataType>(result_type);
  }

  Status Compute(OpKernelContext* ctx) const override {
    utils::MLTypeCallDispatcher<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t,
                                int64_t, uint64_t, float, double, bool, std::string>
        t_disp(result_type_);
    t_disp.Invoke<FromStringTransformerImpl>(ctx);
    return Status::OK();
  }

 private:
  ONNX_NAMESPACE::TensorProto::DataType result_type_;
};

ONNX_OPERATOR_KERNEL_EX(
    FromStringTransformer,
    kMSFeaturizersDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T0", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("InputT", DataTypeImpl::GetTensorType<std::string>()),
    FromStringTransformer);

}  // namespace featurizers
}  // namespace onnxruntime