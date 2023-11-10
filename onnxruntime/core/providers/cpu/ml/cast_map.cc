// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/cast_map.h"
#include <algorithm>
#include "core/common/gsl.h"
using namespace ::onnxruntime::common;

namespace castmap_internal {
template <typename TCastFrom, typename TCastTo>
TCastTo Cast(const TCastFrom& from);

template <>
float Cast<std::string, float>(const std::string& from) {
  return std::stof(from);
}

template <>
int64_t Cast<std::string, int64_t>(const std::string& from) {
  return std::stoll(from);
}

template <>
std::string Cast<std::string, std::string>(const std::string& from) {
  return from;
}

template <>
float Cast<float, float>(const float& from) {
  return from;
}

template <>
int64_t Cast<float, int64_t>(const float& from) {
  return static_cast<int64_t>(from);
}

template <>
std::string Cast<float, std::string>(const float& from) {
  return std::to_string(from);
}
}  // namespace
namespace onnxruntime {
namespace ml {

ONNX_CPU_OPERATOR_ML_KERNEL(
    CastMap,
    1,
    KernelDefBuilder().TypeConstraint("T1",
                                      std::vector<MLDataType>{DataTypeImpl::GetType<std::map<int64_t, std::string>>(),
                                                              DataTypeImpl::GetType<std::map<int64_t, float>>()})
        .TypeConstraint("T2",
                        std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>(),
                                                DataTypeImpl::GetTensorType<int64_t>(),
                                                DataTypeImpl::GetTensorType<std::string>()}),
    CastMap);

Status CastMap::Compute(OpKernelContext* context) const {
  MLDataType input_type = context->InputType(0);

  // input map value is either string or float
  bool float_input = false;

  utils::ContainerChecker c_checker(input_type);
  if (c_checker.IsMapOf<int64_t, float>()) {
    float_input = true;
  } else if (!c_checker.IsMapOf<int64_t, std::string>()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input type of value: ",
                           input_type,
                           " Expected std::map<int64_t, float> or std::map<int64_t, std::string>");
  }

  Status status;
  switch (cast_to_) {
    case (CAST_TO::TO_FLOAT): {
      status = float_input ? ComputeImpl<float, float>(*context, 0.f)
                           : ComputeImpl<std::string, float>(*context, 0.f);
      break;
    }
    case CAST_TO::TO_INT64: {
      status = float_input ? ComputeImpl<float, int64_t>(*context, 0)
                           : ComputeImpl<std::string, int64_t>(*context, 0);
      break;
    }
    case CAST_TO::TO_STRING: {
      status = float_input ? ComputeImpl<float, std::string>(*context, "0.f")
                           : ComputeImpl<std::string, std::string>(*context, "0.f");
      break;
    }
    default:
      return Status(ONNXRUNTIME,
                    INVALID_ARGUMENT,
                    ("Unexpected CAST_TO value of " + std::to_string(static_cast<std::underlying_type<CAST_TO>::type>(cast_to_))));
  }

  return status;
}

template <typename TFrom, typename TTo>
Status CastMap::ComputeImpl(OpKernelContext& context, TTo pad_value) const {
  using namespace castmap_internal;
  using InputMap = std::map<int64_t, TFrom>;

  const auto& X = *context.Input<InputMap>(0);

  int64_t num_dims = map_form_ == PACK_MAP::DENSE ? gsl::narrow_cast<int64_t>(X.size()) : max_map_;

  // create a span for the output
  Tensor* Y = context.Output(0, {1, num_dims});
  auto out = gsl::make_span(Y->MutableData<TTo>(), onnxruntime::narrow<size_t>(Y->Shape().Size()));
  auto out_iter = out.begin();

  // for each item in the entry, use the template specialized Cast function to convert
  if (map_form_ == PACK_MAP::DENSE) {
    // dense map is a straight copy
    std::for_each(X.cbegin(), X.cend(),
                  [&out_iter](const typename InputMap::value_type& entry) {
                    *out_iter = Cast<TFrom, TTo>(entry.second);
                    ++out_iter;
                  });
  } else {
    // sparse map puts pad_value in all entries that aren't present in the input, up to map_max_
    auto cur_input = X.cbegin();
    auto end_input = X.cend();
    auto out_end = out.end();
    int64_t cur_idx = 0;

    ORT_ENFORCE(cur_input == end_input || cur_input->first >= 0,
                "Negative index values are not permitted. First entry in map has index value of ", cur_input->first);

    // for each output value, see if we have an input value, if not use the pad value
    while (out_iter < out_end) {
      if (cur_input != end_input && cur_input->first == cur_idx) {
        *out_iter = Cast<TFrom, TTo>(cur_input->second);
        ++cur_input;
      } else {
        *out_iter = pad_value;
      }

      ++out_iter;
      ++cur_idx;
    }
  }

  return Status::OK();
}

}  // namespace ml
}  // namespace onnxruntime
