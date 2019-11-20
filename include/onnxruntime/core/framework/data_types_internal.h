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
constexpr ONNX_NAMESPACE::TensorProto_DataType ToTensorProtoElementType() {
  return ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
}

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

namespace data_types_internal {

// Returns nullptr if this is not a sequence
const ONNX_NAMESPACE::TypeProto_Sequence* GetSequenceProto(MLDataType);

// Returns nullptr if this is not a map
const ONNX_NAMESPACE::TypeProto_Map* GetMapProto(MLDataType);

// Returns nullptr if this is not an opaque type
const ONNX_NAMESPACE::TypeProto_Opaque* GetOpaqueProto(MLDataType);

// Returns TypeProto_Sequence of the sequence element
// returns nullptr if the element of this sequence is not a sequence
inline const ONNX_NAMESPACE::TypeProto_Sequence* GetSequenceProto(const ONNX_NAMESPACE::TypeProto_Sequence& seq_proto) {
  const auto& elem_type = seq_proto.elem_type();
  if (elem_type.value_case() == ONNX_NAMESPACE::TypeProto::ValueCase::kSequenceType) {
    return &elem_type.sequence_type();
  }
  return nullptr;
}

inline const ONNX_NAMESPACE::TypeProto_Map* GetMapProto(const ONNX_NAMESPACE::TypeProto_Sequence& seq_proto) {
  if (seq_proto.elem_type().value_case() == ONNX_NAMESPACE::TypeProto::ValueCase::kMapType) {
    return &seq_proto.elem_type().map_type();
  }
  return nullptr;
}

inline const ONNX_NAMESPACE::TypeProto_Sequence* GetSequenceProtoForValue(const ONNX_NAMESPACE::TypeProto_Map& map_proto) {
  if (map_proto.value_type().value_case() == ONNX_NAMESPACE::TypeProto::ValueCase::kSequenceType) {
    return &map_proto.value_type().sequence_type();
  }
  return nullptr;
}

inline const ONNX_NAMESPACE::TypeProto_Map* GetMapProtoForValue(const ONNX_NAMESPACE::TypeProto_Map& map_proto) {
  if (map_proto.value_type().value_case() == ONNX_NAMESPACE::TypeProto::ValueCase::kMapType) {
    return &map_proto.value_type().map_type();
  }
  return nullptr;
}

// Check if the ml_type is a sequence that contains tensors of a
// certain primitive type
inline bool IsSequenceOfPrimitiveType(const ONNX_NAMESPACE::TypeProto_Sequence& seq_proto, int32_t prim_type) {
  const auto& elem_type = seq_proto.elem_type();
  if (elem_type.value_case() == ONNX_NAMESPACE::TypeProto::ValueCase::kTensorType) {
    return elem_type.tensor_type().elem_type() == prim_type;
  }
  return false;
}

inline bool IsMapKeyOfPrimitiveType(const ONNX_NAMESPACE::TypeProto_Map& map_proto, int32_t prim_type) {
  return map_proto.key_type() == prim_type;
}

inline bool IsMapValueOfPrimitiveType(const ONNX_NAMESPACE::TypeProto_Map& map_proto, int32_t val_type) {
  const auto& val_type_proto = map_proto.value_type();
  if (val_type_proto.value_case() == ONNX_NAMESPACE::TypeProto::ValueCase::kTensorType) {
    return val_type_proto.tensor_type().elem_type() == val_type;
  }
  return false;
}

// Default check if this is an Opaque type
// since we do not have pre-assigned c++ representation for those
template <class T>
struct IsSequenceOfType {
  static bool check (const ONNX_NAMESPACE::TypeProto_Sequence& seq_proto) {
    return seq_proto.elem_type().value_case() == ONNX_NAMESPACE::TypeProto::ValueCase::kOpaqueType;
  }
};

template <class T>
struct IsMapValueOfType {
  static bool check (const ONNX_NAMESPACE::TypeProto_Map& map_proto) {
    return map_proto.value_type().value_case() == ONNX_NAMESPACE::TypeProto::ValueCase::kOpaqueType;
  }
};

// Sequence specialization (std::vector)
template <class T>
struct IsMapValueOfType<std::vector<T>> {
  static bool check(const ONNX_NAMESPACE::TypeProto_Map& map_proto) {
    const auto* seq_proto = GetSequenceProtoForValue(map_proto);
    if (val_seq_proto != nullptr) {
      return IsSequenceOfType<std::vector<T>>::check(*seq_proto);
    }
    return false;
  }
};

// Map specialization (std::map)
template <class K, class V>
struct IsMapValueOfType<std::map<K, V>> {
  static bool check(const ONNX_NAMESPACE::TypeProto_Map& map_proto) {
    static_assert(ToTensorProtoElementType<K>() != ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED,
                  "Map Key can not be a non-primitive type");
    const auto* val_map_proto = GetMapProtoForValue(map_proto);
    if (val_map_proto != nullptr) {
      constexpr int32_t key_type = ToTensorProtoElementType<K>();
      if (!IsMapKeyOfPrimitiveType(*val_map_proto, key_type)) {
        return false;
      }
      constexpr int32_t val_type = ToTensorProtoElementType<V>();
      if (val_type != ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED) {
        return IsMapValueOfPrimitiveType(*map_proto, val_type);
      }
      return IsMapValueOfType<V>::check(*val_map_proto);
    }
    return false;
  }
};

// Handles the case where sequence element is also a sequence
template <class T>
struct IsSequenceOfType<std::vector<T>> {
  static bool check(const ONNX_NAMESPACE::TypeProto_Sequence& seq_proto) {
    const auto* nested_seq_proto = GetSequenceProto(seq_proto);
    if (nested_seq_proto != nullptr) {
      constexpr int32_t prim_type = ToTensorProtoElementType<T>();
      if (prim_type != ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED) {
        return IsSequenceOfPrimitiveType(*nested_seq_proto, prim_type);
      }
      // Recurse on the element type of the nested sequence
      return IsSequenceOfType<T>::check(*nested_seq_proto);
    }
    return false;
  }
};

// Handles the case where sequence element is a map
template <class K, class V>
struct IsSequenceOfType<std::map<K, V>> {
  static bool check(const ONNX_NAMESPACE::TypeProto_Sequence& seq_proto) {
    static_assert(ToTensorProtoElementType<K>() != ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED,
                  "Map Key can not be a non-primitive type");
    const auto* map_proto = GetMapProto(seq_proto);
    if (map_proto != nullptr) {
      // Maps can only have primitive types as keys so we can check the key right away
      constexpr int32_t key_type = ToTensorProtoElementType<K>();
      if (!IsMapKeyOfPrimitiveType(*map_proto, key_type)) {
        return false;
      }
      constexpr int32_t val_type = ToTensorProtoElementType<V>();
      if (val_type != ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED) {
        return IsMapValueOfPrimitiveType(*map_proto, val_type);
      }
      return IsMapValueOfType<V>::check(*map_proto);
    }
    return false;
  }
};

}  // namespace data_types_internal

////////////////////////////////////////////////////////////////////
/// Provides generic interface to test whether MLDataType is a Sequence,
/// Map or an Opaque type including arbitrary recursive definitions
/// without querying DataTypeImpl::GetType<T> for all known complex types

// T is a sequence contained element type
// If returns true then we know that the runtime
// representation is std::vector<T>
// T itself can be a runtime representation of another
// sequence, map, opaque type or a tensor
// That is it can be std::vector, std::map
// If T is a primitive type sequence is tested whether it contains
// tensors of that type
// If T is an opaque type, then it is only tested to be opaque but not exactly
// a specific opaque type.
template <class T>
inline bool IsSequenceOf(MLDataType ml_type) {
  const auto* seq_proto = data_types_internal::GetSequenceProto(ml_type);
  if (seq_proto != nullptr) {
    constexpr int32_t prim_type = ToTensorProtoElementType<T>();
    if (prim_type != ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED) {
      return data_types_internal::IsSequenceOfPrimitiveType(*seq_proto, prim_type);
    }
    return data_types_internal::IsSequenceOfType<T>::check(*seq_proto);
  }
  return false;
}

// If the value is an opaque type, it is not tested to be exact
// just the fact it is an opaque
template <class K, class V>
inline bool IsMapOf(MLDataType ml_type) {
  static_assert(ToTensorProtoElementType<K>() != ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED,
                "Map Key can not be a non-primitive type");
  const auto* map_proto = data_types_internal::GetMapProto(ml_type);
  if (map_proto != nullptr) {
    constexpr int32_t key_type = ToTensorProtoElementType<K>();
    if (!data_types_internal::IsMapKeyOfPrimitiveType(*map_proto, key_type)) {
      return false;
    }
    constexpr int32_t val_type = ToTensorProtoElementType<V>();
    if (val_type != ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED) {
      return data_types_internal::IsMapValueOfPrimitiveType(*map_proto, val_type);
    }
    return data_types_internal::IsMapValueOfType<V>::check(*map_proto);
  }
  return false;
}

bool IsOpaqueType(MLDataType ml_type, const char* domain, const char* name);

}  // namespace utils
}  // namespace onnxruntime
