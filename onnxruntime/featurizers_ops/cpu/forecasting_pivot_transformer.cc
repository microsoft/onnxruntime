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

// template <typename T> //float, double
// struct ForecastingPivotTransformerImpl {
//   void operator()(OpKernelContext* ctx) const {

//     using TransformerT = Microsoft::Featurizer::Featurizers::ForecastingPivotTransformer<T>;
//     using MatrixT = NS::RowMajMatrix<T>;
//     using InputMatrixT = Eigen::Map<const MatrixT>;
//     using InputType = std::vector<InputMatrixT>;
//     using OutputType = std::vector<T>;

//     //Get the transformer
//     const auto* state_tensor(ctx->Input<Tensor>(0));
//     const uint8_t* const state_data(state_tensor->Data<uint8_t>());
//     Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().GetDims()[0]);
//     typename TransformerT transformer(archive);

//     // Get the Number of Rows
//     const auto* input_tensor_temp(ctx->Input<Tensor>(1));
//     const int64_t row_num = input_tensor_temp->Shape()[0];

//     // Prepare the Output
//     double* output_data;
//     bool has_allocate_output_data = false;
//     std::function<void(OutputType)> callback_fn;
//     callback_fn = [ctx, &output_data, &has_allocate_output_data, row_num](OutputType value) -> void {
//       if (!has_allocate_output_data) {
//         TensorShape output_shape({row_num, static_cast<int64_t>(value.size())});
//         Tensor* output_tensor(ctx->Output(0, output_shape));
//         output_data = output_tensor->MutableData<double>();
//         has_allocate_output_data = true;
//       }
//       std::copy(value.begin(), value.end(), output_data);
//       output_data += value.size();
//     };

//     // Transform
//     const int64_t input_count = onnxruntime::OpKernel::Node().InputArgCount().front();
//     InputType input;
//     input.reserve(input_count - 1);
//     std::unordered_map<int, std::tuple<T*,int64_t, int64_t>> dataPtrMap;
//     for (int64_t i = 0; i < row_num; ++i) {
//       //Prepare Input and Output
//       input.clear();
//       for (int index = 1; index < input_count; ++index) {
//         if (i == 0) {
//           //Get the Input
//           const auto* input_tensor(ctx->Input<Tensor>(index));
//           const T* input_data(input_tensor->template Data<T>());
//           // Matrix Eigen raw buffer mapping
//           const int64_t input_dim_1 = input_tensor->Shape()[1];
//           const int64_t input_dim_2 = input_tensor->Shape()[2];

//           dataPtrMap.insert(std::pair<int, std::tuple<T*,int64_t, int64_t>>(index, std::make_tuple(input_data, input_dim_1, input_dim_2)));
//         }
//         const T* input_data = std::get<0>(dataPtrMap.at(index));
//         const int64_t input_dim_1 = std::get<1>(dataPtrMap.at(index));
//         const int64_t input_dim_2 = std::get<2>(dataPtrMap.at(index));
//         InputMatrixT input_matrix(input_data, input_dim_1, input_dim_2);
//         input.emplace_back(input_matrix);

//         input_data += input_matrix.size(); // maybe not correct
//       }
//       //Execute
//       transformer.execute(input, callback_fn);
//     }
//     transformer.flush(callback_fn);
//   }
// };

class ForecastingPivotTransformer final : public OpKernel {
 public:
  explicit ForecastingPivotTransformer(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* ctx) const override {
    // const int64_t input_count = onnxruntime::OpKernel::Node().InputArgCount().front();
    // std::cout << input_count << std::endl;
    // utils::MLTypeCallDispatcher<ForecastingPivotTransformerImpl, float, double>
    //     t_disp(ctx->Input<Tensor>(1)->GetElementType());
    // t_disp.Invoke(ctx);

    using T = float;

    using TransformerT = Microsoft::Featurizer::Featurizers::ForecastingPivotTransformer<T>;
    using MatrixT = NS::RowMajMatrix<T>;
    using InputMatrixT = Eigen::Map<const MatrixT>;
    using InputType = std::vector<InputMatrixT>;
    using OutputType = std::vector<T>;

    //Get the transformer
    const auto* state_tensor(ctx->Input<Tensor>(0));
    const uint8_t* const state_data(state_tensor->Data<uint8_t>());
    Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().GetDims()[0]);
    typename TransformerT transformer(archive);

    // Get the Number of Rows
    const auto* input_tensor_temp(ctx->Input<Tensor>(1));
    const int64_t row_num = input_tensor_temp->Shape()[0];

    // Prepare the Output
    double* output_data;
    bool has_allocate_output_data = false;
    std::function<void(OutputType)> callback_fn;
    callback_fn = [ctx, &output_data, &has_allocate_output_data, row_num](OutputType value) -> void {
      if (!has_allocate_output_data) {
        TensorShape output_shape({row_num, static_cast<int64_t>(value.size())});
        Tensor* output_tensor(ctx->Output(0, output_shape));
        output_data = output_tensor->MutableData<double>();
        has_allocate_output_data = true;
      }
      std::copy(value.begin(), value.end(), output_data);
      output_data += value.size();
    };

    // Transform
    const int64_t input_count = onnxruntime::OpKernel::Node().InputArgCount().front();
    InputType input;
    input.reserve(input_count - 1);
    std::unordered_map<int, std::tuple<const T*,int64_t, int64_t>> dataPtrMap;
    for (int64_t i = 0; i < row_num; ++i) {
      //Prepare Input and Output
      input.clear();
      for (int index = 1; index < input_count; ++index) {
        if (i == 0) {
          //Get the Input
          const auto* input_tensor(ctx->Input<Tensor>(index));
          const T* input_data(input_tensor->template Data<T>());
          // Matrix Eigen raw buffer mapping
          const int64_t input_dim_1 = input_tensor->Shape()[1];
          const int64_t input_dim_2 = input_tensor->Shape()[2];

          dataPtrMap.insert(std::pair<int, std::tuple<const T*,int64_t, int64_t>>(index, std::make_tuple<const T*,int64_t, int64_t>(input_data, input_dim_1, input_dim_2)));
        }
        const T* input_data = std::get<0>(dataPtrMap.at(index));
        const int64_t input_dim_1 = std::get<1>(dataPtrMap.at(index));
        const int64_t input_dim_2 = std::get<2>(dataPtrMap.at(index));
        InputMatrixT input_matrix(input_data, input_dim_1, input_dim_2);
        input.emplace_back(input_matrix);

        input_data += input_matrix.size(); // maybe not correct
      }
      //Execute
      transformer.execute(input, callback_fn);
    }
    transformer.flush(callback_fn);


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
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>()
                              //DataTypeImpl::GetTensorType<double>()
                              }),
    ForecastingPivotTransformer);
}  // namespace featurizers
}  // namespace onnxruntime
