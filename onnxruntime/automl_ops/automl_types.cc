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

// We temporarily create a binding here until we standardize it.
class AutoMLTimePoint : public NonTensorType<dtf::TimePoint> {
 public:
  static MLDataType Type() {
    static AutoMLTimePoint inst;
    return &inst;
  }

  bool IsCompatible(const ONNX_NAMESPACE::TypeProto& type_proto) const override {
    const auto& rhs = type_proto;
    return rhs.value_case() == ONNX_NAMESPACE::TypeProto::ValueCase::kTimepointType &&
           rhs.timepoint_type().has_domain() && rhs.timepoint_type().domain() == kMsAutoMLDomain &&
           rhs.timepoint_type().has_name() && rhs.timepoint_type().name() == kTimepointName;
  }

 private:
   AutoMLTimePoint() {
    auto* mutable_tp = this->mutable_type_proto().mutable_timepoint_type();
    mutable_tp->mutable_domain()->assign(kMsAutoMLDomain);
    mutable_tp->mutable_name()->assign(kTimepointName);
  }
};

template<>
MLDataType DataTypeImpl::GetType<dtf::TimePoint>() {
  return AutoMLTimePoint::Type();
}

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
