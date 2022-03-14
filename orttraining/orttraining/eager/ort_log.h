// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <sstream>

#include <c10/core/Scalar.h>
#include <c10/util/Optional.h>
#include <ATen/core/Tensor.h>
#include <tuple>

namespace torch_ort {
namespace eager {

template<typename T>
class ORTLogHelper {
    public:
        ORTLogHelper(const T& value) : value_{value} {            
        }
       
        const T& GetValue() const {
            return value_;
        }        

    private:
        const T& value_;
};

template<typename T>
inline std::ostream& operator<<(std::ostream& out, const ORTLogHelper<T>& helper)
{
    out << helper.GetValue();
    return out;
}

inline std::ostream& operator<<(std::ostream& out, const ORTLogHelper<at::Device> helper) {
  auto& device = helper.GetValue();
  out << "Device:" << c10::DeviceTypeName(device.type(), true);
  out << ":" << (int)device.index();
  return out;
}

template<typename T>
inline std::ostream& operator<<(std::ostream& out, const ORTLogHelper<c10::optional<T>> helper) {
  auto& optional = helper.GetValue();
  if (optional.has_value()) {
    out << *optional;
  } else {
    out << "None";
  }
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const ORTLogHelper<c10::Scalar> helper) {
  auto& scalar = helper.GetValue();
  out << "Scalar:" << scalar.type();
  switch (scalar.type()) {
    case c10::ScalarType::Double:
      out << "=" << scalar.to<double>();
      break;
    case c10::ScalarType::Long:
      out << "=" << scalar.to<int64_t>();
      break;
    case c10::ScalarType::Bool:
      out << "=" << scalar.to<bool>();
      break;
    default:
      break;
  }
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const ORTLogHelper<at::Tensor> /*tensor*/) {
  out << "Tensor";
  return out;
}

template<typename... Arguments>
inline std::ostream& operator<<(std::ostream& out, const std::tuple<Arguments...>& args) {  
    const std::size_t length = sizeof...(Arguments);
 
    std::apply( 
        [length, &out](auto const&... ps) {
            std::size_t k = 0;
 
            // Variadic expansion used.
            ((out << ORTLogHelper(ps)
                  << (++k == length ? "" : ", ")),
             ...); 
        },
        args);

  return out;
}

#define ORT_LOG_FATAL LOGS_DEFAULT(FATAL)
#define ORT_LOG_ERROR LOGS_DEFAULT(ERROR)
#define ORT_LOG_WARNING LOGS_DEFAULT(WARNING)
#define ORT_LOG_INFO LOGS_DEFAULT(INFO)
#define ORT_LOG_VERBOSE LOGS_DEFAULT(VERBOSE)

#ifdef __clang__
#  define ORT_LOG_FN(...) ORT_LOG_VERBOSE << __PRETTY_FUNCTION__ << '(' << std::tuple(__VA_ARGS__) << ')'
#else
#  define ORT_LOG_FN(...) ORT_LOG_VERBOSE << __PRETTY_FUNCTION__
#endif

} // namespace eager
} // namespace torch_ort