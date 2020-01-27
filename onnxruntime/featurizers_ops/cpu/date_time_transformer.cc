// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"

#include "Featurizers/DateTimeFeaturizer.h"
#include "Archive.h"

namespace onnxruntime {
namespace featurizers {

class DateTimeTransformer final : public OpKernel {
 public:
  explicit DateTimeTransformer(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* ctx) const override {
    // Create the transformer
    Microsoft::Featurizer::Featurizers::DateTimeTransformer transformer(
        [ctx](void) {
          const auto* state_tensor(ctx->Input<Tensor>(0));
          const uint8_t* const state_data(state_tensor->Data<uint8_t>());

          Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().GetDims()[0]);
          return Microsoft::Featurizer::Featurizers::DateTimeTransformer(archive);
        }());

    // Get the input
    const auto* input_tensor(ctx->Input<Tensor>(1));
    const int64_t* input_data(input_tensor->Data<int64_t>());

    // Prepare the output
    Tensor* year_tensor(ctx->Output(0, input_tensor->Shape()));
    Tensor* month_tensor(ctx->Output(1, input_tensor->Shape()));
    Tensor* day_tensor(ctx->Output(2, input_tensor->Shape()));
    Tensor* hour_tensor(ctx->Output(3, input_tensor->Shape()));
    Tensor* minute_tensor(ctx->Output(4, input_tensor->Shape()));
    Tensor* second_tensor(ctx->Output(5, input_tensor->Shape()));
    Tensor* amPm_tensor(ctx->Output(6, input_tensor->Shape()));
    Tensor* hour12_tensor(ctx->Output(7, input_tensor->Shape()));
    Tensor* dayOfWeek_tensor(ctx->Output(8, input_tensor->Shape()));
    Tensor* dayOfQuarter_tensor(ctx->Output(9, input_tensor->Shape()));
    Tensor* dayOfYear_tensor(ctx->Output(10, input_tensor->Shape()));
    Tensor* weekOfMonth_tensor(ctx->Output(11, input_tensor->Shape()));
    Tensor* quarterOfYear_tensor(ctx->Output(12, input_tensor->Shape()));
    Tensor* halfOfYear_tensor(ctx->Output(13, input_tensor->Shape()));
    Tensor* weekIso_tensor(ctx->Output(14, input_tensor->Shape()));
    Tensor* yearIso_tensor(ctx->Output(15, input_tensor->Shape()));
    Tensor* monthLabel_tensor(ctx->Output(16, input_tensor->Shape()));
    Tensor* amPmLabel_tensor(ctx->Output(17, input_tensor->Shape()));
    Tensor* dayOfWeekLabel_tensor(ctx->Output(18, input_tensor->Shape()));
    Tensor* holidayName_tensor(ctx->Output(19, input_tensor->Shape()));
    Tensor* isPaidTimeOff_tensor(ctx->Output(20, input_tensor->Shape()));

    int32_t* year_data(year_tensor->MutableData<int32_t>());
    uint8_t* month_data(month_tensor->MutableData<uint8_t>());
    uint8_t* day_data(day_tensor->MutableData<uint8_t>());
    uint8_t* hour_data(hour_tensor->MutableData<uint8_t>());
    uint8_t* minute_data(minute_tensor->MutableData<uint8_t>());
    uint8_t* second_data(second_tensor->MutableData<uint8_t>());
    uint8_t* amPm_data(amPm_tensor->MutableData<uint8_t>());
    uint8_t* hour12_data(hour12_tensor->MutableData<uint8_t>());
    uint8_t* dayOfWeek_data(dayOfWeek_tensor->MutableData<uint8_t>());
    uint8_t* dayOfQuarter_data(dayOfQuarter_tensor->MutableData<uint8_t>());
    uint16_t* dayOfYear_data(dayOfYear_tensor->MutableData<uint16_t>());
    uint16_t* weekOfMonth_data(weekOfMonth_tensor->MutableData<uint16_t>());
    uint8_t* quarterOfYear_data(quarterOfYear_tensor->MutableData<uint8_t>());
    uint8_t* halfOfYear_data(halfOfYear_tensor->MutableData<uint8_t>());
    uint8_t* weekIso_data(weekIso_tensor->MutableData<uint8_t>());
    int32_t* yearIso_data(yearIso_tensor->MutableData<int32_t>());
    std::string* monthLabel_data(monthLabel_tensor->MutableData<std::string>());
    std::string* amPmLabel_data(amPmLabel_tensor->MutableData<std::string>());
    std::string* dayOfWeekLabel_data(dayOfWeekLabel_tensor->MutableData<std::string>());
    std::string* holidayName_data(holidayName_tensor->MutableData<std::string>());
    uint8_t* isPaidTimeOff_data(isPaidTimeOff_tensor->MutableData<uint8_t>());

    // Execute
    const int64_t length(input_tensor->Shape().Size());

    for (int64_t i = 0; i < length; ++i) {
      auto result(transformer.execute(std::chrono::system_clock::from_time_t(input_data[i])));

      year_data[i] = std::move(result.year);
      month_data[i] = std::move(result.month);
      day_data[i] = std::move(result.day);
      hour_data[i] = std::move(result.hour);
      minute_data[i] = std::move(result.minute);
      second_data[i] = std::move(result.second);
      amPm_data[i] = std::move(result.amPm);
      hour12_data[i] = std::move(result.hour12);
      dayOfWeek_data[i] = std::move(result.dayOfWeek);
      dayOfQuarter_data[i] = std::move(result.dayOfQuarter);
      dayOfYear_data[i] = std::move(result.dayOfYear);
      weekOfMonth_data[i] = std::move(result.weekOfMonth);
      quarterOfYear_data[i] = std::move(result.quarterOfYear);
      halfOfYear_data[i] = std::move(result.halfOfYear);
      weekIso_data[i] = std::move(result.weekIso);
      yearIso_data[i] = std::move(result.yearIso);
      monthLabel_data[i] = std::move(result.monthLabel);
      amPmLabel_data[i] = std::move(result.amPmLabel);
      dayOfWeekLabel_data[i] = std::move(result.dayOfWeekLabel);
      holidayName_data[i] = std::move(result.holidayName);
      isPaidTimeOff_data[i] = std::move(result.isPaidTimeOff);
    }

    return Status::OK();
  }
};

ONNX_OPERATOR_KERNEL_EX(
    DateTimeTransformer,
    kMSFeaturizersDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T0", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    DateTimeTransformer);

}  // namespace featurizers
}  // namespace onnxruntime
