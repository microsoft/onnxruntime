// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <assert.h>
#include <stdint.h>
#include <string>

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/graph/onnx_protobuf.h"


namespace onnxruntime {
namespace utils {

template <typename T>
constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType();

template <>
constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType<float>() {
  return ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
}
template <>
constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType<uint8_t>() {
  return ONNX_NAMESPACE::TensorProto_DataType_UINT8;
};
template <>
constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType<int8_t>() {
  return ONNX_NAMESPACE::TensorProto_DataType_INT8;
};
template <>
constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType<uint16_t>() {
  return ONNX_NAMESPACE::TensorProto_DataType_UINT16;
};
template <>
constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType<int16_t>() {
  return ONNX_NAMESPACE::TensorProto_DataType_INT16;
};
template <>
constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType<int32_t>() {
  return ONNX_NAMESPACE::TensorProto_DataType_INT32;
};
template <>
constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType<int64_t>() {
  return ONNX_NAMESPACE::TensorProto_DataType_INT64;
};
template <>
constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType<std::string>() {
  return ONNX_NAMESPACE::TensorProto_DataType_STRING;
};
template <>
constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType<bool>() {
  return ONNX_NAMESPACE::TensorProto_DataType_BOOL;
};
template <>
constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType<MLFloat16>() {
  return ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
};
template <>
constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType<double>() {
  return ONNX_NAMESPACE::TensorProto_DataType_DOUBLE;
};
template <>
constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType<uint32_t>() {
  return ONNX_NAMESPACE::TensorProto_DataType_UINT32;
};
template <>
constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType<uint64_t>() {
  return ONNX_NAMESPACE::TensorProto_DataType_UINT64;
};
template <>
constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType<BFloat16>() {
  return ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16;
};

// The following primitives are strongly recommended for switching on tensor input datatypes for
// kernel implementations.
//
//  1) If you need to handle all of the primitive tensor contained datatypes, the best choice would be macros
//     DispatchOnTensorType or DispatchOnTensorTypeWithReturn. Use inline wrappers so your function can be invoked as function<T>().
//  2) if you have a few types, use Tensor.IsDataType<T>()/IsDataTypeString() or use utils::IsPrimitiveDataType<T>()
//     if you have a standalone MLDatatType with a sequence of if/else statements.
//  3) For something in between, we suggest to use CallDispatcher pattern.
//
// Invoking DataTypeImpl::GetType<T>() for switching on input types is discouraged and should be avoided.
// Every primitive type carries with it an integer constant that can be used for quick switching on types.

#define DispatchOnTensorType(tensor_type, function, ...)          \
  switch (tensor_type->AsPrimitiveDataType()->GetDataType()) {    \
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:              \
      function<float>(__VA_ARGS__);                               \
      break;                                                      \
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:               \
      function<bool>(__VA_ARGS__);                                \
      break;                                                      \
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:             \
      function<double>(__VA_ARGS__);                              \
      break;                                                      \
    case ONNX_NAMESPACE::TensorProto_DataType_STRING:             \
      function<std::string>(__VA_ARGS__);                         \
      break;                                                      \
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:               \
      function<int8_t>(__VA_ARGS__);                              \
      break;                                                      \
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:              \
      function<uint32_t>(__VA_ARGS__);                            \
      break;                                                      \
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:              \
      function<int16_t>(__VA_ARGS__);                             \
      break;                                                      \
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:             \
      function<uint16_t>(__VA_ARGS__);                            \
      break;                                                      \
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:              \
      function<int32_t>(__VA_ARGS__);                             \
      break;                                                      \
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:             \
      function<uint32_t>(__VA_ARGS__);                            \
      break;                                                      \
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:              \
      function<int64_t>(__VA_ARGS__);                             \
      break;                                                      \
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:             \
      function<uint64_t>(__VA_ARGS__);                            \
      break;                                                      \
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:            \
      function<MLFloat16>(__VA_ARGS__);                           \
      break;                                                      \
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:           \
      function<BFloat16>(__VA_ARGS__);                            \
      break;                                                      \
    default:                                                      \
      ORT_ENFORCE(false, "Unknown tensor type of ", tensor_type); \
  }

#define DispatchOnTensorTypeWithReturn(tensor_type, retval, function, ...) \
  switch (tensor_type->AsPrimitiveDataType()->GetDataType()) {             \
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:                       \
      retval = function<float>(__VA_ARGS__);                               \
      break;                                                               \
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:                        \
      retval = function<bool>(__VA_ARGS__);                                \
      break;                                                               \
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:                      \
      retval = function<double>(__VA_ARGS__);                              \
      break;                                                               \
    case ONNX_NAMESPACE::TensorProto_DataType_STRING:                      \
      retval = function<std::string>(__VA_ARGS__);                         \
      break;                                                               \
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:                        \
      retval = function<int8_t>(__VA_ARGS__);                              \
      break;                                                               \
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:                       \
      retval = function<uint8_t>(__VA_ARGS__);                             \
      break;                                                               \
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:                      \
      retval = function<uint16_t>(__VA_ARGS__);                            \
      break;                                                               \
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:                       \
      retval = function<int16_t>(__VA_ARGS__);                             \
      break;                                                               \
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:                       \
      retval = function<int32_t>(__VA_ARGS__);                             \
      break;                                                               \
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:                      \
      retval = function<uint32_t>(__VA_ARGS__);                            \
      break;                                                               \
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:                       \
      retval = function<int64_t>(__VA_ARGS__);                             \
      break;                                                               \
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:                      \
      retval = function<uint64_t>(__VA_ARGS__);                            \
      break;                                                               \
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:                     \
      retval = function<MLFloat16>(__VA_ARGS__);                           \
      break;                                                               \
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:                    \
      retval = function<BFloat16>(__VA_ARGS__);                            \
      break;                                                               \
    default:                                                               \
      ORT_ENFORCE(false, "Unknown tensor type of ", tensor_type);          \
  }

////////////////////////////////////////////////////////////////////////////////
/// Use the following primitives if you have a few types to switch on so you
//  can write a short sequence of if/else statements.

// This is a frequently used check so we make a separate utility function.
inline bool IsDataTypeString(MLDataType dt_type) {
  auto prim_type = dt_type->AsPrimitiveDataType();
  return (prim_type != nullptr && prim_type->GetDataType() == ONNX_NAMESPACE::TensorProto_DataType_STRING);
}

// Test if MLDataType is a concrete type of PrimitiveDataTypeBase
// and it is T
template <class T>
inline bool IsPrimitiveDataType(MLDataType dt_type) {
  auto prim_type = dt_type->AsPrimitiveDataType();
  return (prim_type != nullptr && prim_type->GetDataType() == ToTensorProtoElementType<T>());
}

// Use after AsPrimitiveDataType() is successful
// Check if PrimitiveDataTypeBase is of type T
template <class T>
inline bool IsPrimitiveDataType(const PrimitiveDataTypeBase* prim_type) {
  assert(prim_type != nullptr);
  return prim_type->GetDataType() == ToTensorProtoElementType<T>();
}

// This implementation contains a workaround for GCC bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=47226
// GCC until very recently does not support template parameter pack expansion within lambda context.
namespace mltype_dispatcher_internal {
// T - type handled by this helper
struct CallableDispatchableHelper {
  int32_t dt_type_;  // Type currently dispatched
  size_t called_;

  explicit CallableDispatchableHelper(int32_t dt_type) : dt_type_(dt_type), called_(0) {}

  // Must return integer to be in a expandable context
  template <class T, class Fn, class... Args>
  int Invoke(Fn&& fn, Args&&... args) {
    if (utils::ToTensorProtoElementType<T>() == dt_type_) {
      std::forward<Fn>(fn)(std::forward<Args>(args)...);
      ++called_;
    }
    return 0;
  }
};

// Default policy is to throw with no return type.
template <class Ret>
struct UnsupportedTypeDefaultPolicy {
  Ret operator()(int32_t dt_type) const {
    ORT_THROW("Unsupported data type: ", dt_type);
  }
};

// Helper with the result type
template <class Ret, class UnsupportedPolicy = UnsupportedTypeDefaultPolicy<Ret>>
struct CallableDispatchableRetHelper {
  int32_t dt_type_;  // Type currently dispatched
  size_t called_;
  Ret result_;

  explicit CallableDispatchableRetHelper(int32_t dt_type) : dt_type_(dt_type), called_(0), result_() {}

  Ret Get() {
    // See if there were multiple invocations.It is a bug.
    ORT_ENFORCE(called_ < 2, "Check for duplicate types in MLTypeCallDispatcherRet");
    // No type was invoked
    if (called_ == 0) {
      result_ = UnsupportedPolicy()(dt_type_);
    }
    return result_;
  }

  // Must return integer to be in a expandable context
  template <class T, class Fn, class... Args>
  int Invoke(Fn&& fn, Args&&... args) {
    if (utils::ToTensorProtoElementType<T>() == dt_type_) {
      result_ = std::forward<Fn>(fn)(std::forward<Args>(args)...);
      ++called_;
    }
    return 0;
  }
};

}  // namespace mltype_dispatcher_internal

// This class helps to efficiently dispatch calls for templated
// kernel implementation functions that has no return value.
// If your implementation function must return a value such as Status
// Use MLTypeCallDispatcherRet class.
//
// The first template parameter is a template<T> struct/class functor
// that must implement operator() with arbitrary number of arguments
// and void return turn. It must return Ret type if you are using MLTypeCallDispatcherRet.
// Fn must be default constructible.
//
// Types is a type list that are supported by this kernel implementation.
// There should be no duplicate types. An exception will be thrown if there
// a duplicate.
//
// The constructor accepts an enum that is obtained from
// input_tensor->DataType()->AsPrimitiveType()->GetDataType().
// Fn will be called only once the type designated by dt_type value.
// If current dt_type is not handled, the Dispatcher will throw an exception.
//
template <template <typename> class Fn, typename... Types>
class MLTypeCallDispatcher {
  int32_t dt_type_;

 public:
  explicit MLTypeCallDispatcher(int32_t dt_type) : dt_type_(dt_type) {}

  template <typename... Args>
  void Invoke(Args&&... args) const {
    mltype_dispatcher_internal::CallableDispatchableHelper helper(dt_type_);
    int results[] = {0, helper.template Invoke<Types>(Fn<Types>(), std::forward<Args>(args)...)...};
    ORT_UNUSED_PARAMETER(results);
    ORT_ENFORCE(helper.called_ < 2, "Check for duplicate types in MLTypeCallDispatcher");
    ORT_ENFORCE(helper.called_ == 1, "Unsupported data type: ", dt_type_);
  }
};

// Version of the MLTypeDispatcher with a return type.
// Return type of Fn must return type convertible to Ret
// The value of the return type will be the return value
// of the function for type T which was specified for execution.
template <class Ret, template <typename> class Fn, typename... Types>
class MLTypeCallDispatcherRet {
  int32_t dt_type_;

 public:
  explicit MLTypeCallDispatcherRet(int32_t dt_type) : dt_type_(dt_type) {}

  template <typename... Args>
  Ret Invoke(Args&&... args) const {
    mltype_dispatcher_internal::CallableDispatchableRetHelper<Ret> helper(dt_type_);
    int results[] = {0, helper.template Invoke<Types>(Fn<Types>(), std::forward<Args>(args)...)...};
    ORT_UNUSED_PARAMETER(results);
    return helper.Get();
  }

  template <class UnsupportedPolicy, typename... Args>
  Ret InvokeWithUnsupportedPolicy(Args&&... args) const {
    mltype_dispatcher_internal::CallableDispatchableRetHelper<Ret, UnsupportedPolicy> helper(dt_type_);
    int results[] = {0, helper.template Invoke<Types>(Fn<Types>(), std::forward<Args>(args)...)...};
    ORT_UNUSED_PARAMETER(results);
    return helper.Get();
  }
};

}  // namespace utils
}  // namespace onnxruntime
