// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"

#include "Featurizers/TruncatedSVDFeaturizer.h"
#include "Featurizers/../Archive.h"

namespace NS = Microsoft::Featurizer;

namespace onnxruntime {
namespace featurizers {

template <typename T>
struct TruncatedSVDTransformerImpl {
  void operator()(OpKernelContext* ctx) const {
    using MatrixT = NS::RowMajMatrix<T>;
    using InputMatrixT = Eigen::Map<const MatrixT>;
    // Create the transformer
    Microsoft::Featurizer::Featurizers::TruncatedSVDTransformer<InputMatrixT> transformer(
        [ctx]() {
          const auto* state_tensor(ctx->Input<Tensor>(0));
          const uint8_t* const state_data(state_tensor->Data<uint8_t>());

          Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().Size());
          return Microsoft::Featurizer::Featurizers::TruncatedSVDTransformer<InputMatrixT>(archive);
        }());

    // Get the input
    const auto* input_tensor(ctx->Input<Tensor>(1));
    const T* input_data(input_tensor->template Data<T>());
    // Matrix Eigen raw buffer mapping
    const auto input_dim_0 = input_tensor->Shape()[0];
    const auto input_dim_1 = input_tensor->Shape()[1];
    InputMatrixT input_matrix(input_data, input_dim_0, input_dim_1);

    // Prepare output shape which is [M, P] where P is the first dimension (rows)
    // of P matrix from the transformer
    const int64_t dim_0 = input_dim_0;
    const int64_t dim_1 = transformer.getEigenVectorColsNumber();
    TensorShape output_shape({dim_0, dim_1});
    auto* output_tensor(ctx->Output(0, output_shape));
    T* output_data = output_tensor->template MutableData<T>();
    Eigen::Map<MatrixT> output_matrix(output_data, dim_0, dim_1);

    std::function<void(MatrixT val)> callback;
    bool callback_allow = true;
    callback = [&output_matrix, callback_allow](MatrixT val) {
      ORT_ENFORCE(callback_allow, "callback function can only be called during execute() and special flush() when needed");
      output_matrix = val;
    };
    transformer.execute(input_matrix, callback);
    // The flush() does nothing but shows Featurizers concept
    callback_allow = false;
    transformer.flush(callback);
  }
};

class TruncatedSVDTransformer final : public OpKernel {
 public:
  explicit TruncatedSVDTransformer(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* ctx) const override {
    utils::MLTypeCallDispatcher<float, double> t_disp(ctx->Input<Tensor>(1)->GetElementType());
    t_disp.Invoke<TruncatedSVDTransformerImpl>(ctx);
    return Status::OK();
  }
};

ONNX_OPERATOR_KERNEL_EX(
    TruncatedSVDTransformer,
    kMSFeaturizersDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T0", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("InputT", {DataTypeImpl::GetTensorType<float>(),
                                   DataTypeImpl::GetTensorType<double>()}),
    TruncatedSVDTransformer);


}  // namespace featurizers
}  // namespace onnxruntime