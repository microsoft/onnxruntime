// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"

#include "core/automl/featurizers/src/FeaturizerPrep/Featurizers/DateTimeFeaturizer.h"

namespace dtf = Microsoft::Featurizer::DateTimeFeaturizer;

namespace onnxruntime {
namespace automl {

class DateTimeTransformer final : public OpKernel {
 public:
  explicit DateTimeTransformer(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* context) const override;
};

Status DateTimeTransformer::Compute(OpKernelContext* ctx) const {
  Status s;
  auto input_tensor = ctx->Input<Tensor>(0);
  dtf::TimePoint* output = ctx->Output<dtf::TimePoint>(0);

  int64_t tp = *input_tensor->Data<int64_t>();
  std::chrono::system_clock::time_point sys_time{std::chrono::seconds(tp)};
  *output = dtf::SystemToDPTimePoint(sys_time);
  return s;
}

ONNX_OPERATOR_KERNEL_EX(
    DateTimeTransformer,
    kMSAutoMLDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T2", DataTypeImpl::GetType<Microsoft::Featurizer::DateTimeFeaturizer::TimePoint>()),
    DateTimeTransformer);
}  // namespace automl
}  // namespace onnxruntime
