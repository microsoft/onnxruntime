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
struct CopyImputedColumnsImpl {
  void operator()(const Tensor* input_tensor, Tensor* output_tensor_imputed,
                  const std::vector<int64_t>& row_idx_record, int64_t input_matrix_size, int num_output_rows) const {
      const T* input_data(input_tensor->template Data<T>());
      T* output_data_imputed = output_tensor_imputed->MutableData<T>();

      for (int imputed_output_row_idx = 0; imputed_output_row_idx < num_output_rows; imputed_output_row_idx++) {
        output_data_imputed = std::copy(input_data + row_idx_record[imputed_output_row_idx] * input_matrix_size,
                                        input_data + (row_idx_record[imputed_output_row_idx] + 1) * input_matrix_size,
                                        output_data_imputed);
      }
  }
};

template <typename T>
struct ForecastingPivotTransformerImpl {
  void operator()(OpKernelContext* ctx, int64_t num_pivot_columns) const {

    ORT_ENFORCE(num_pivot_columns > 0, "num_pivot_columns > 0, otherwise there will be no input to pivot");

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
    std::vector<int64_t> row_idx_record;
    int64_t row_idx = 0;
    std::function<void(OutputType const & value)> callback_fn;
    callback_fn = [&output, &row_idx_record, &row_idx](OutputType const & value) -> void {
      output.emplace_back(value);
      row_idx_record.push_back(row_idx);
    };

    // Transform
    const int input_node_0_count = ctx->NumVariadicInputs(0);
    const int input_node_1_count = ctx->NumVariadicInputs(1);

    InputType input;
    input.reserve(num_pivot_columns);
    std::unordered_map<int, std::tuple<const T*,int64_t, int64_t>> dataPtrMap;
    std::vector<uint32_t> horizon_output_helper;
    for (row_idx = 0; row_idx < row_num; ++row_idx) {
      //Prepare Input and Output
      input.clear();
      for (int index = input_node_0_count; index < input_node_0_count + num_pivot_columns; ++index) {
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
        std::get<0>(inputTuple) = input_data;
      }

      // Get the horizon vector from input, since num_pivot_columns > 0. So input is not null
      const size_t matrix_cols_num = input[0].cols();
      for (size_t col_idx = 0; col_idx < matrix_cols_num; col_idx++) {
        bool has_nan = false;
        for (int input_matrix_id = 0; input_matrix_id < num_pivot_columns; input_matrix_id++) {
          const size_t matrix_rows_num = input[input_matrix_id].rows();
          auto matrix = input[input_matrix_id];
          for (int row_id = 0; row_id < static_cast<int>(matrix_rows_num); row_id++) {
            if (std::isnan(matrix(row_id, col_idx))) {
              has_nan = true;
              break;
            }
          }
          if (has_nan)
            break;
        }
        if (!has_nan) {
          horizon_output_helper.push_back(static_cast<uint32_t>(matrix_cols_num - col_idx));
        }
      }

      //Execute
      transformer.execute(std::make_tuple(input.begin(), input.end()), callback_fn);
    }
    transformer.flush(callback_fn);

    // Prepare the number of output rows
    ORT_ENFORCE(!output.empty(), "All rows dropped is an exception");
    int num_output_rows = static_cast<int>(output.size());
    ORT_ENFORCE(static_cast<int>(row_idx_record.size()) == num_output_rows, "row_idx_record.size() == num_output_rows");

    // Prepare the pivoted Output
    int num_pivot_output_columns = 0;
    if (!output.empty() && !output[0].empty()) {
      num_pivot_output_columns = static_cast<int>(output[0].size());

      for (int pivot_output_tensor_idx = 0; pivot_output_tensor_idx < num_pivot_output_columns; pivot_output_tensor_idx++) {
        TensorShape output_shape({static_cast<int64_t>(num_output_rows), 1});
        Tensor* output_tensor(ctx->Output(pivot_output_tensor_idx, output_shape));
        T* output_data = output_tensor->MutableData<T>();

        for (int pivot_output_row_idx = 0; pivot_output_row_idx < num_output_rows; pivot_output_row_idx++){
          *output_data++ = output[pivot_output_row_idx][pivot_output_tensor_idx];
        }
      }
    }

    // Prepare the non-pivot(imputed) Output
    for (int imputed_output_count_idx = 0; imputed_output_count_idx < input_node_1_count - num_pivot_columns; imputed_output_count_idx++) {

      int tensor_id = input_node_0_count + static_cast<int>(num_pivot_columns) + imputed_output_count_idx;
      const auto* input_tensor(ctx->Input<Tensor>(tensor_id));

      auto input_dims = input_tensor->Shape();
      int64_t input_matrix_size = 1;
      for (size_t dim_idx = 1; dim_idx < input_dims.NumDimensions(); dim_idx++)
        input_matrix_size *= input_dims[dim_idx];

      TensorShape output_shape_imputed({static_cast<int64_t>(num_output_rows), input_matrix_size});
      Tensor* output_tensor_imputed(ctx->Output(imputed_output_count_idx + num_pivot_output_columns, output_shape_imputed));

      const auto elem_type = input_tensor->GetElementType();

      utils::MLTypeCallDispatcher<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t,
                                  float, double, bool, std::string>
          t_disp(elem_type);
      t_disp.Invoke<CopyImputedColumnsImpl>(input_tensor, output_tensor_imputed, row_idx_record, input_matrix_size, num_output_rows);
    }

    // Prepare the horizon Output(uint32)
    TensorShape output_shape_horizon({static_cast<int64_t>(num_output_rows), 1});
    Tensor* output_tensor_horizon(ctx->Output(input_node_1_count + num_pivot_output_columns - static_cast<int>(num_pivot_columns), output_shape_horizon));
    uint32_t* output_data_horizon = output_tensor_horizon->MutableData<uint32_t>();

    std::copy(horizon_output_helper.begin(), horizon_output_helper.end(), output_data_horizon);
  }
};

class ForecastingPivotTransformer final : public OpKernel {
 public:
  explicit ForecastingPivotTransformer(const OpKernelInfo& info) :
    OpKernel(info), _num_pivot_columns(info.GetAttrOrDefault("num_pivot_columns", static_cast<int64_t>(0))) {
  }

  Status Compute(OpKernelContext* ctx) const override {
    utils::MLTypeCallDispatcher<float, double> t_disp(ctx->Input<Tensor>(1)->GetElementType());
    t_disp.Invoke<ForecastingPivotTransformerImpl>(ctx, _num_pivot_columns);

    return Status::OK();
  }
 private:
  const int64_t _num_pivot_columns;
};

ONNX_OPERATOR_KERNEL_EX(
    ForecastingPivotTransformer,
    kMSFeaturizersDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T0", DataTypeImpl::GetTensorType<uint8_t>())
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
    ForecastingPivotTransformer);

}  // namespace featurizers
}  // namespace onnxruntime
