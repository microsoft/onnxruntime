// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

#include "Featurizers/ShortGrainDropperFeaturizer.h"
#include "Featurizers/../Archive.h"

namespace NS = Microsoft::Featurizer;

namespace onnxruntime {
namespace featurizers {

template <typename T>
struct CopyNonDroppedColumnsImpl {
  void operator()(const Tensor* variadic_input_tensor, Tensor* output_after_drop_tensor,
                  const std::vector<bool>& rows_to_drop, int64_t input_row_size) const {
      const T* input_data(variadic_input_tensor->template Data<T>());
      T* output_after_drop_data = output_after_drop_tensor->MutableData<T>();

      for (int row_idx = 0; row_idx < static_cast<int>(rows_to_drop.size()); ++row_idx) {
        if (!rows_to_drop[row_idx]) {
          output_after_drop_data = std::copy(input_data, input_data + input_row_size, output_after_drop_data);
        }
        input_data += input_row_size;
      }
  }
};

void ShortGrainDropperTransformerImpl(OpKernelContext* ctx) {
  // Create the transformer
  Microsoft::Featurizer::Featurizers::ShortGrainDropperTransformer transformer(
      [ctx]() {
        const auto* state_tensor(ctx->Input<Tensor>(0));
        const uint8_t* const state_data(state_tensor->Data<uint8_t>());

        Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().Size());
        return Microsoft::Featurizer::Featurizers::ShortGrainDropperTransformer(archive);
      }());

  // Get the Grain input
  const auto* input_tensor = ctx->Input<Tensor>(1);
  const std::string* input_data = input_tensor->template Data<std::string>();
  const int64_t input_rows_num = input_tensor->Shape()[0];
  const int64_t strings_num = input_tensor->Shape()[1];

  ORT_ENFORCE(input_rows_num > 0, "input_rows_num > 0");

  // Record which row to drop
  std::vector<bool> rows_to_drop;
   // Transform
  std::vector<std::string> input_data_vec;
  input_data_vec.reserve(strings_num);
  for (int64_t rows_idx = 0; rows_idx < input_rows_num; ++rows_idx) {
    input_data_vec.clear();
    std::copy(input_data, input_data + strings_num, std::back_inserter(input_data_vec));
    rows_to_drop.push_back(transformer.execute(input_data_vec));
    input_data += strings_num;
  }

  // Calculate number of remaining rows
  int remaining_rows_num = static_cast<int>(std::count(rows_to_drop.begin(), rows_to_drop.end(), false));

  ORT_ENFORCE(remaining_rows_num > 0, "remaining_rows_num > 0");

  // Prepare the Grain output
  TensorShape grain_output_shape({remaining_rows_num, strings_num});
  Tensor* grain_output_tensor(ctx->Output(0, grain_output_shape));
  std::string* grain_output_data(grain_output_tensor->MutableData<std::string>());
  const std::string* input_grain_data = input_tensor->template Data<std::string>();
  for (int rows_idx = 0; rows_idx < static_cast<int>(input_rows_num); ++rows_idx) {
    if (!rows_to_drop[rows_idx]) {
      grain_output_data = std::copy(input_grain_data, input_grain_data + strings_num, grain_output_data);
    }
    input_grain_data += strings_num;
  }

  // Prepare other outputs. input(2)->output(1), input(3)->output(2), ...
  const int variadic_input_start_id = ctx->NumVariadicInputs(0) + ctx->NumVariadicInputs(1);
  const int variadic_input_end_id = variadic_input_start_id + ctx->NumVariadicInputs(2);
  for (int input_id = variadic_input_start_id; input_id < variadic_input_end_id; ++input_id) {

    const auto* variadic_input_tensor(ctx->Input<Tensor>(input_id)); //2-d tensor
    const int64_t input_row_size = variadic_input_tensor->Shape()[1];

    TensorShape output_after_drop_shape({static_cast<int64_t>(remaining_rows_num), input_row_size});
    Tensor* output_after_drop_tensor(ctx->Output(input_id - 1, output_after_drop_shape));

    const auto elem_type = variadic_input_tensor->GetElementType();

    utils::MLTypeCallDispatcher<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t,
                                float, double, bool, std::string>
        t_disp(elem_type);
    t_disp.Invoke<CopyNonDroppedColumnsImpl>(variadic_input_tensor, output_after_drop_tensor, rows_to_drop, input_row_size);
  }
};

class ShortGrainDropperTransformer final : public OpKernel {
 public:
  explicit ShortGrainDropperTransformer(const OpKernelInfo& info) : OpKernel(info) {
  }
  Status Compute(OpKernelContext* ctx) const override {
    ShortGrainDropperTransformerImpl(ctx);
    return Status::OK();
  }
};

ONNX_OPERATOR_KERNEL_EX(
    ShortGrainDropperTransformer,
    kMSFeaturizersDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T0", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<std::string>())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<int8_t>(),
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
    ShortGrainDropperTransformer);

}  // namespace featurizers
}  // namespace onnxruntime
