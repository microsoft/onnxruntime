// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"

#include "Featurizers/AnalyticalRollingWindowFeaturizer.h"
#include "Featurizers/../Archive.h"

namespace onnxruntime {
namespace featurizers {

template <typename T>
struct RollingWindowTransformerImpl {
  void operator()(OpKernelContext* ctx) const {
    // Define the type
    using GrainT = std::vector<std::string>;
    using EstimatorT = Microsoft::Featurizer::Featurizers::GrainedAnalyticalRollingWindowEstimator<T>;
    using GrainedInputType = typename EstimatorT::InputType;
    using OutputType = typename EstimatorT::TransformedType;

    //Get the transformer
    const auto* state_tensor(ctx->Input<Tensor>(0));
    const uint8_t* const state_data(state_tensor->Data<uint8_t>());
    Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().Size());
    typename EstimatorT::TransformerType transformer(archive);

    // Get the Grains
    const auto* grains_tensor(ctx->Input<Tensor>(1));
    const std::string* grains_data(grains_tensor->Data<std::string>());
    const auto grains_num = grains_tensor->Shape()[1];

    // Get the Target
    const auto* target_tensor(ctx->Input<Tensor>(2));
    const T* target_data(target_tensor->Data<T>());

    // Prepare the output
    const auto output_dim_0 = grains_tensor->Shape()[0];

    double* output_data = nullptr;
    bool has_allocate_output_data = false;
    std::function<void(OutputType)> callback_fn;
    callback_fn = [ctx, &output_data, &has_allocate_output_data, output_dim_0](OutputType value) -> void {
      //Allocate tensor memory after first output is generated
      if(!has_allocate_output_data) {
        TensorShape output_shape({output_dim_0, static_cast<int64_t>(value.size())});
        Tensor* output_tensor(ctx->Output(0, output_shape));
        output_data = output_tensor->MutableData<double>();
        has_allocate_output_data = true;
      }
      Eigen::Map<OutputType> output_matrix_mapping(output_data, value.rows(), value.cols());
      output_matrix_mapping = value;
      output_data += value.size();
    };

    // Transform
    GrainT grains;
    grains.reserve(grains_num);
    for (int64_t i = 0; i < output_dim_0; ++i) {
      //Prepare Input
      grains.clear();
      std::copy(grains_data, grains_data + grains_num, std::back_inserter(grains));
      const GrainedInputType input_tuple(grains, *target_data);
      //Execute
      transformer.execute(input_tuple, callback_fn);
      //Increment Pointer
      target_data++;
      grains_data += grains_num;
    }
    transformer.flush(callback_fn);
  }
};

class RollingWindowTransformer final : public OpKernel {
 public:
  explicit RollingWindowTransformer(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* ctx) const override {
    utils::MLTypeCallDispatcher<RollingWindowTransformerImpl, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t,
                                int64_t, uint64_t, float, double>
        t_disp(ctx->Input<Tensor>(2)->GetElementType());
    t_disp.Invoke(ctx);
    return Status::OK();
  }
};

ONNX_OPERATOR_KERNEL_EX(
    RollingWindowTransformer,
    kMSFeaturizersDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T0", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("GrainT", DataTypeImpl::GetTensorType<std::string>())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<int8_t>(),
                              DataTypeImpl::GetTensorType<uint8_t>(),
                              DataTypeImpl::GetTensorType<int16_t>(),
                              DataTypeImpl::GetTensorType<uint16_t>(),
                              DataTypeImpl::GetTensorType<int32_t>(),
                              DataTypeImpl::GetTensorType<uint32_t>(),
                              DataTypeImpl::GetTensorType<int64_t>(),
                              DataTypeImpl::GetTensorType<uint64_t>(),
                              DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<double>()
                              }),
    RollingWindowTransformer);
}  // namespace featurizers
}  // namespace onnxruntime
