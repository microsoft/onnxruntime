// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"

#include "Featurizers/ForecastingPivotFeaturizer.h"
#include "Featurizers/../Archive.h"

namespace NS = Microsoft::Featurizer;
namespace onnxruntime {
namespace featurizers {

template <typename T>
struct ForecastingPivotTransformerImpl {
  void operator()(OpKernelContext* ctx) const {
    using MatrixT = NS::RowMajMatrix<typename NS::Traits<T>::nullable_type>;
    using InputType = std::vector<Eigen::Map<const MatrixT>>;
    using OutputType = std::vector<T>;
    using TransformerT = Microsoft::Featurizer::Featurizers::ForecastingPivotTransformer<std::tuple<typename InputType::iterator, typename InputType::iterator>>;

    //Get the transformer
    const auto* state_tensor(ctx->Input<Tensor>(0));
    const uint8_t* const state_data(state_tensor->Data<uint8_t>());
    Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().Size());
    TransformerT transformer(archive);

    // Get the Number of Rows
    const auto* input_tensor_temp(ctx->Input<Tensor>(1));
    const int64_t row_num = input_tensor_temp->Shape()[0];

    //Get the output for whole rows is inevitable because there is conceptually no way to determine the shape of output for each row
    std::vector<OutputType> output;
    std::function<void(OutputType const & value)> callback_fn;
    callback_fn = [&output](OutputType const & value) -> void {
      output.emplace_back(value);
    };

    // Transform
    const int input_node_0_count = ctx->NumVariadicInputs(0);
    const int input_node_1_count = ctx->NumVariadicInputs(1);
    InputType input;
    input.reserve(input_node_1_count);
    std::unordered_map<int, std::tuple<const T*,int64_t, int64_t>> dataPtrMap;
    for (int64_t row_idx = 0; row_idx < row_num; ++row_idx) {
      //Prepare Input and Output
      input.clear();
      for (int index = input_node_0_count; index < input_node_0_count + input_node_1_count; ++index) {
        if (row_idx == 0) {
          //Get the Input
          const auto* input_tensor(ctx->Input<Tensor>(index));
          const T* input_data(input_tensor->template Data<T>());
          // Matrix Eigen raw buffer mapping
          const int64_t input_dim_1 = input_tensor->Shape()[1];
          const int64_t input_dim_2 = input_tensor->Shape()[2];
          //store data pointer and dimension information
          std::tuple<const T*,int64_t, int64_t> info_tuple(input_data, input_dim_1, input_dim_2);
          dataPtrMap.insert(std::pair<int, std::tuple<const T*,int64_t, int64_t>>(index, info_tuple));
        }
        std::tuple<const T*,int64_t, int64_t> &inputTuple(dataPtrMap.at(index));
        const T* input_data(std::get<0>(inputTuple));
        const int64_t input_dim_1(std::get<1>(inputTuple));
        const int64_t input_dim_2(std::get<2>(inputTuple));
        input.push_back(typename InputType::value_type(input_data, input_dim_1, input_dim_2));
        //Increment data pointer
        input_data += input_dim_1 * input_dim_2;
      }
      //Execute
      transformer.execute(std::make_tuple(input.begin(), input.end()), callback_fn);
    }
    transformer.flush(callback_fn);

    // Prepare the Output
    TensorShape output_shape({static_cast<int64_t>(output.size()), static_cast<int64_t>(output[0].size())});
    Tensor* output_tensor(ctx->Output(0, output_shape));
    T* output_data = output_tensor->MutableData<T>();

    for (OutputType const & row : output)
      output_data = std::copy(row.begin(), row.end(), output_data);
  }
};

class ForecastingPivotTransformer final : public OpKernel {
 public:
  explicit ForecastingPivotTransformer(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* ctx) const override {
    utils::MLTypeCallDispatcher<ForecastingPivotTransformerImpl, float, double>
        t_disp(ctx->Input<Tensor>(1)->GetElementType());
    t_disp.Invoke(ctx);

    return Status::OK();
  }
};

ONNX_OPERATOR_KERNEL_EX(
    ForecastingPivotTransformer,
    kMSFeaturizersDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T0", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<double>()
                              }),
    ForecastingPivotTransformer);

}  // namespace featurizers
}  // namespace onnxruntime
