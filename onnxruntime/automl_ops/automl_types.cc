// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"

#include "automl_ops/automl_types.h"
#include "automl_ops/automl_featurizers.h"

namespace dtf = Microsoft::Featurizer::DateTimeFeaturizer;

namespace onnxruntime {

// This temporary to register custom types so ORT is aware of it
// although it still can not serialize such a type.
// These character arrays must be extern so the resulting instantiated template
// is globally unique

extern const char kMsAutoMLDomain[] = "com.microsoft.automl";
extern const char kTimepointName[] = "DateTimeFeaturizer_TimePoint";

// External struct which should be made available to the client
// of the API via an external header.
struct ExternalTimePoint {
  std::int32_t year;
  std::uint8_t month;
  std::uint8_t day;
  std::uint8_t hour;
  std::uint8_t minute;
  std::uint8_t second;
  std::uint8_t dayOfWeek;
  std::uint16_t dayOfYear;
  std::uint8_t quarterOfYear;
  std::uint8_t weekOfMonth;
};

// Specialize for our type so we can convert to external struct
// 
template <>
struct NonTensorTypeConverter<dtf::TimePoint> {
  static void FromContainer(MLDataType dtype, const void* data, size_t data_size, OrtValue& output) {
    ORT_ENFORCE(sizeof(ExternalTimePoint) == data_size, "Expecting an instance of ExternalTimePoint");
    const ExternalTimePoint* dc = reinterpret_cast<const ExternalTimePoint*>(data);
    std::unique_ptr<dtf::TimePoint> tp(new dtf::TimePoint);
    tp->year = dc->year;
    tp->month = dc->month;
    tp->day = dc->day;
    tp->hour = dc->hour;
    tp->minute = dc->minute;
    tp->second = dc->second;
    tp->dayOfWeek = dc->dayOfWeek;
    tp->dayOfYear = dc->dayOfYear;
    tp->quarterOfYear = dc->quarterOfYear;
    tp->weekOfMonth = dc->weekOfMonth;
    output.Init(tp.get(),
                dtype,
                dtype->GetDeleteFunc());
    tp.release();
  }
  static void ToContainer(const OrtValue& input, size_t data_size, void* data) {
    ORT_ENFORCE(sizeof(ExternalTimePoint) == data_size, "Expecting an instance of ExternalTimePoint");
    ExternalTimePoint* dc = reinterpret_cast<ExternalTimePoint*>(data);
    const dtf::TimePoint& tp = input.Get<dtf::TimePoint>();
    dc->year = tp.year;
    dc->month = tp.month;
    dc->day = tp.day;
    dc->hour = tp.hour;
    dc->minute = tp.minute;
    dc->second = tp.second;
    dc->dayOfWeek = tp.dayOfWeek;
    dc->dayOfYear = tp.dayOfYear;
    dc->quarterOfYear = tp.quarterOfYear;
    dc->weekOfMonth = tp.weekOfMonth;
  }
};


// This has to be under onnxruntime to properly specialize a function template
ORT_REGISTER_OPAQUE_TYPE(dtf::TimePoint, kMsAutoMLDomain, kTimepointName);

namespace automl {

#define REGISTER_CUSTOM_PROTO(TYPE, reg_fn)            \
  {                                                    \
    MLDataType mltype = DataTypeImpl::GetType<TYPE>(); \
    reg_fn(mltype);                                    \
  }

void RegisterAutoMLTypes(const std::function<void(MLDataType)>& reg_fn) {
  REGISTER_CUSTOM_PROTO(dtf::TimePoint, reg_fn);
}
#undef REGISTER_CUSTOM_PROTO
} // namespace automl
} // namespace onnxruntime
