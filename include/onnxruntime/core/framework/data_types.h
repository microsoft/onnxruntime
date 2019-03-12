// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <stdint.h>
#include <type_traits>
#include <map>
#include <unordered_map>

#include "core/common/common.h"
#include "core/common/exceptions.h"

namespace ONNX_NAMESPACE {
class TypeProto;
}  // namespace ONNX_NAMESPACE
namespace onnxruntime {
/// Predefined registered types
//maps
using MapStringToString = std::map<std::string, std::string>;
using MapStringToInt64 = std::map<std::string, int64_t>;
using MapStringToFloat = std::map<std::string, float>;
using MapStringToDouble = std::map<std::string, double>;
using MapInt64ToString = std::map<int64_t, std::string>;
using MapInt64ToInt64 = std::map<int64_t, int64_t>;
using MapInt64ToFloat = std::map<int64_t, float>;
using MapInt64ToDouble = std::map<int64_t, double>;

//vectors/sequences
using VectorString = std::vector<std::string>;
using VectorInt64 = std::vector<int64_t>;
using VectorFloat = std::vector<float>;
using VectorDouble = std::vector<double>;
using VectorMapStringToFloat = std::vector<MapStringToFloat>;
using VectorMapInt64ToFloat = std::vector<MapInt64ToFloat>;

class DataTypeImpl;
class TensorTypeBase;

// MLFloat16
union MLFloat16 {
  uint16_t val;

  explicit MLFloat16(uint16_t x) : val(x) {}
  MLFloat16() : val(0) {}
};

inline bool operator==(const MLFloat16& left, const MLFloat16& right) {
  return left.val == right.val;
}

inline bool operator!=(const MLFloat16& left, const MLFloat16& right) {
  return left.val != right.val;
}

struct ort_endian {
  union q {
    uint16_t v_;
    uint8_t b_[2];
    constexpr explicit q(uint16_t v) noexcept : v_(v) {}
  };
  static constexpr bool is_little() {
    return q(0x200).b_[0] == 0x0;
  }
  static constexpr bool is_big() {
    return q(0x200).b_[0] == 0x2;
  }
};

//BFloat16
struct BFloat16 {
  uint16_t val{0};
  explicit BFloat16() {}
  explicit BFloat16(uint16_t v) : val(v) {}
  explicit BFloat16(float v) {
    uint16_t* dst = reinterpret_cast<uint16_t*>(&v);
    if (ort_endian::is_little()) {
      val = dst[1];
    } else {
      val = dst[0];
    }
  }
  float ToFloat() const {
    float result;
    uint16_t* dst = reinterpret_cast<uint16_t*>(&result);
    if (ort_endian::is_little()) {
      dst[1] = val;
      dst[0] = 0;
    } else {
      dst[0] = val;
      dst[1] = 0;
    }
    return result;
  }
};

inline void BFloat16ToFloat(const BFloat16* blf, float* flt, size_t size) {
  auto src = blf;
  auto d = flt;
  for (; size != 0; ++src, ++d, --size) {
    *d = src->ToFloat();
  }
}

inline void FloatToBFloat16(const float* flt, BFloat16* blf, size_t size) {
  auto src = flt;
  auto d = blf;
  for (; size != 0; ++src, ++d, --size) {
    new (d) BFloat16(*src);
  }
}

inline bool operator==(const BFloat16& left, const BFloat16& right) {
  return left.val == right.val;
}

inline bool operator!=(const BFloat16& left, const BFloat16& right) {
  return left.val != right.val;
}

// DataTypeImpl pointer as unique DataTypeImpl identifier.
using MLDataType = const DataTypeImpl*;
// be used with class MLValue
using DeleteFunc = void (*)(void*);
using CreateFunc = std::function<void*()>;

/**
 * \brief Base class for MLDataType
 *
 */
class DataTypeImpl {
 public:
  virtual ~DataTypeImpl() = default;

  /**
   * \brief this API will be used to check type compatibility at runtime
   *
   * \param type_proto a TypeProto instance that is constructed for a specific type
   *        will be checked against a TypeProto instance contained within a corresponding
   *        MLDataType instance.
   */
  virtual bool IsCompatible(const ONNX_NAMESPACE::TypeProto& type_proto) const = 0;

  virtual size_t Size() const = 0;

  virtual DeleteFunc GetDeleteFunc() const = 0;

  /**
   * \brief Retrieves an instance of TypeProto for
   *        a given MLDataType
   * \returns optional TypeProto. Only ONNX types
              has type proto, non-ONNX types will return nullptr.
   */
  virtual const ONNX_NAMESPACE::TypeProto* GetTypeProto() const = 0;

  virtual bool IsTensorType() const {
    return false;
  }

  // Returns this if this is of tensor-type and null otherwise
  virtual const TensorTypeBase* AsTensorType() const {
    return nullptr;
  }

  // Return the type meta that we are using in the runtime.
  template <typename T, typename... Types>
  static MLDataType GetType();

  // Return the types for a concrete tensor type, like Tensor_Float
  template <typename elemT>
  static MLDataType GetTensorType();

  /**
   * Convert an ONNX TypeProto to onnxruntime DataTypeImpl.
   * However, this conversion is lossy. Don't try to use 'this->GetTypeProto()' converting it back
   * Don't pass the returned value to MLValue::MLValue(...) function
   * \param proto
   */
  static MLDataType TypeFromProto(const ONNX_NAMESPACE::TypeProto& proto);

  static const TensorTypeBase* TensorTypeFromONNXEnum(int type);
  static const char* ToString(MLDataType type);
  // Registers ONNX_NAMESPACE::DataType (internalized string) with
  // MLDataType. DataType is produced by internalizing an instance of
  // TypeProto contained within MLDataType
  static void RegisterDataType(MLDataType);

  static const std::vector<MLDataType>& AllTensorTypes();
  static const std::vector<MLDataType>& AllFixedSizeTensorTypes();
  static const std::vector<MLDataType>& AllNumericTensorTypes();
};

std::ostream& operator<<(std::ostream& out, MLDataType data_type);

/*
 * Type registration helpers
 */
namespace data_types_internal {
/// TensorType helpers
///

// There is a specialization only for one
// type argument.
template <typename... Types>
struct TensorContainedTypeSetter {
  static void SetTensorElementType(ONNX_NAMESPACE::TypeProto&);
  static void SetMapKeyType(ONNX_NAMESPACE::TypeProto&);
};

/// Is a given type on the list of types?
/// Accepts a list of types and the first argument is the type
/// We are checking if it is listed among those that follow
template <typename T, typename... Types>
struct IsAnyOf;

/// Two types remaining, end of the list
template <typename T, typename Tail>
struct IsAnyOf<T, Tail> : public std::is_same<T, Tail> {
};

template <typename T, typename H, typename... Tail>
struct IsAnyOf<T, H, Tail...> {
  static constexpr bool value = (std::is_same<T, H>::value ||
                                 IsAnyOf<T, Tail...>::value);
};

/// Tells if the specified type is one of fundamental types
/// that can be contained within a tensor.
/// We do not have raw fundamental types, rather a subset
/// of fundamental types is contained within tensors.
template <typename T>
struct IsTensorContainedType : public IsAnyOf<T, float, uint8_t, int8_t, uint16_t, int16_t,
                                              int32_t, int64_t, std::string, bool, MLFloat16,
                                              double, uint32_t, uint64_t, BFloat16> {
};

/// This template's Get() returns a corresponding MLDataType
/// It dispatches the call to either GetTensorType<>() or
/// GetType<>()
template <typename T, bool TensorContainedType>
struct GetMLDataType;

template <typename T>
struct GetMLDataType<T, true> {
  static MLDataType Get() {
    return DataTypeImpl::GetTensorType<T>();
  }
};

template <typename T>
struct GetMLDataType<T, false> {
  static MLDataType Get() {
    return DataTypeImpl::GetType<T>();
  }
};

/// MapTypes helper API
/// K should always be one of the primitive data types
/// V can be either a primitive type (in which case it is a tensor)
/// or other preregistered types

void CopyMutableMapValue(const ONNX_NAMESPACE::TypeProto&,
                         ONNX_NAMESPACE::TypeProto&);

template <typename K, typename V>
struct SetMapTypes {
  static void Set(ONNX_NAMESPACE::TypeProto& proto) {
    TensorContainedTypeSetter<K>::SetMapKeyType(proto);
    MLDataType dt = GetMLDataType<V, IsTensorContainedType<V>::value>::Get();
    const auto* value_proto = dt->GetTypeProto();
    ORT_ENFORCE(value_proto != nullptr, typeid(V).name(),
                " expected to be a registered ONNX type");
    CopyMutableMapValue(*value_proto, proto);
  }
};

/// Sequence helpers
///
// Element type is a primitive type so we set it to a tensor<elemT>
void CopyMutableSeqElement(const ONNX_NAMESPACE::TypeProto&,
                           ONNX_NAMESPACE::TypeProto&);

template <typename T>
struct SetSequenceType {
  static void Set(ONNX_NAMESPACE::TypeProto& proto) {
    MLDataType dt = GetMLDataType<T, IsTensorContainedType<T>::value>::Get();
    const auto* elem_proto = dt->GetTypeProto();
    ORT_ENFORCE(elem_proto != nullptr, typeid(T).name(),
                " expected to be a registered ONNX type");
    CopyMutableSeqElement(*elem_proto, proto);
  }
};

/// OpaqueTypes helpers
///
void AssignOpaqueDomainName(const char* domain, const char* name,
                            ONNX_NAMESPACE::TypeProto& proto);

}  // namespace data_types_internal

/// All tensors base
class TensorTypeBase : public DataTypeImpl {
 public:
  static MLDataType Type();

  /// We first compare type_proto pointers and then
  /// if they do not match try to account for the case
  /// where TypeProto was created ad-hoc and not queried from MLDataType
  bool IsCompatible(const ONNX_NAMESPACE::TypeProto& type_proto) const override;

  bool IsTensorType() const override {
    return true;
  }

  const TensorTypeBase* AsTensorType() const override {
    return this;
  }

  size_t Size() const override;

  DeleteFunc GetDeleteFunc() const override;

  const ONNX_NAMESPACE::TypeProto* GetTypeProto() const override;

  virtual MLDataType GetElementType() const {
    // should never reach here.
    ORT_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }

  TensorTypeBase(const TensorTypeBase&) = delete;
  TensorTypeBase& operator=(const TensorTypeBase&) = delete;

 protected:
  ONNX_NAMESPACE::TypeProto& mutable_type_proto();

  TensorTypeBase();
  ~TensorTypeBase() override;

 private:
  struct Impl;
  Impl* impl_;
};

/**
 * \brief Tensor type. This type does not have a C++ type associated with
 * it at registration time except the element type. One of the types mentioned
 * above at IsTensorContainedType<> list is acceptable.
 *
 * \details
 *        Usage:
 *        ORT_REGISTER_TENSOR(ELEMENT_TYPE)
 *        Currently all of the Tensors irrespective of the dimensions are mapped to Tensor<type>
 *        type. IsCompatible() currently ignores shape.
 */

template <typename elemT>
class TensorType : public TensorTypeBase {
 public:
  static_assert(data_types_internal::IsTensorContainedType<elemT>::value,
                "Requires one of the tensor fundamental types");

  static MLDataType Type();

  /// Tensors only can contain basic data types
  /// that have been previously registered with ONNXRuntime
  MLDataType GetElementType() const override {
    return DataTypeImpl::GetType<elemT>();
  }

 private:
  TensorType() {
    using namespace data_types_internal;
    TensorContainedTypeSetter<elemT>::SetTensorElementType(this->mutable_type_proto());
  }
};

/**
 * \brief Base type for all non-tensors, maps, sequences and opaques
 */
class NonTensorTypeBase : public DataTypeImpl {
 public:
  size_t Size() const override = 0;

  DeleteFunc GetDeleteFunc() const override = 0;

  virtual CreateFunc GetCreateFunc() const = 0;

  const ONNX_NAMESPACE::TypeProto* GetTypeProto() const override;

  NonTensorTypeBase(const NonTensorTypeBase&) = delete;
  NonTensorTypeBase& operator=(const NonTensorTypeBase&) = delete;

 protected:
  NonTensorTypeBase();
  ~NonTensorTypeBase() override;

  ONNX_NAMESPACE::TypeProto& mutable_type_proto();

  bool IsMapCompatible(const ONNX_NAMESPACE::TypeProto& type_proto) const;

  bool IsSequenceCompatible(const ONNX_NAMESPACE::TypeProto& type_proto) const;

  bool IsOpaqueCompatible(const ONNX_NAMESPACE::TypeProto& type_proto) const;

 private:
  struct Impl;
  Impl* impl_;
};

// This is where T is the actual CPPRuntimeType
template <typename T>
class NonTensorType : public NonTensorTypeBase {
 private:
  static void Delete(void* p) {
    delete static_cast<T*>(p);
  }

 public:
  size_t Size() const override {
    return sizeof(T);
  }

  DeleteFunc GetDeleteFunc() const override {
    return &Delete;
  }

  CreateFunc GetCreateFunc() const override {
    return []() { return new T(); };
  }

 protected:
  NonTensorType() = default;
};

/**
 * \brief MapType. Use this type to register
 * mapping types.
 *
 * \param T - cpp type that you wish to register as runtime MapType
 *
 * \details Usage: ORT_REGISTER_MAP(C++Type)
 *          The type is required to have mapped_type and
 *          key_type defined
 */
template <typename CPPType>
class MapType : public NonTensorType<CPPType> {
 public:
  static_assert(data_types_internal::IsTensorContainedType<typename CPPType::key_type>::value,
                "Requires one of the tensor fundamental types as key");

  static MLDataType Type();

  bool IsCompatible(const ONNX_NAMESPACE::TypeProto& type_proto) const override {
    return this->IsMapCompatible(type_proto);
  }

 private:
  MapType() {
    using namespace data_types_internal;
    SetMapTypes<typename CPPType::key_type, typename CPPType::mapped_type>::Set(this->mutable_type_proto());
  }
};

/**
 * \brief SequenceType. Use to register sequences.
 *
 *  \param T - CPP type that you wish to register as Sequence
 *             runtime type.
 *
 * \details Usage: ORT_REGISTER_SEQ(C++Type)
 *          The type is required to have value_type defined
 */
template <typename CPPType>
class SequenceType : public NonTensorType<CPPType> {
 public:
  static MLDataType Type();

  bool IsCompatible(const ONNX_NAMESPACE::TypeProto& type_proto) const override {
    return this->IsSequenceCompatible(type_proto);
  }

 private:
  SequenceType() {
    data_types_internal::SetSequenceType<typename CPPType::value_type>::Set(this->mutable_type_proto());
  }
};

/**
 * \brief OpaqueType
 *
 * \param T - cpp runtume that implements the Opaque type
 *
 * \param const char D[] - domain must be extern to be unique
 *
 * \param const char N[] - name must be extern to be unique
 *
 * \details Only one CPP type can be associated with a particular
 *          OpaqueType registration
 *
 */
template <typename T, const char D[], const char N[]>
class OpaqueType : public NonTensorType<T> {
 public:
  static MLDataType Type();

  bool IsCompatible(const ONNX_NAMESPACE::TypeProto& type_proto) const override {
    return this->IsOpaqueCompatible(type_proto);
  }

 private:
  OpaqueType() {
    data_types_internal::AssignOpaqueDomainName(D, N, this->mutable_type_proto());
  }
};

template <typename T>
class NonOnnxType : public DataTypeImpl {
 private:
  static void Delete(void* p) {
    delete static_cast<T*>(p);
  }

 public:
  bool IsCompatible(const ONNX_NAMESPACE::TypeProto&) const override {
    return false;
  }

  static MLDataType Type();

  size_t Size() const override {
    return sizeof(T);
  }

  DeleteFunc GetDeleteFunc() const override {
    return &Delete;
  }

  const ONNX_NAMESPACE::TypeProto* GetTypeProto() const final {
    return nullptr;
  }

 private:
  NonOnnxType() = default;
};

// Explicit specialization of base class template function
// is only possible within the enclosing namespace scope,
// thus a simple way to pre-instantiate a given template
// at a registration time does not currently work and the macro
// is needed.
#define ORT_REGISTER_TENSOR_TYPE(ELEM_TYPE)             \
  template <>                                           \
  MLDataType TensorType<ELEM_TYPE>::Type() {            \
    static TensorType<ELEM_TYPE> tensor_type;           \
    return &tensor_type;                                \
  }                                                     \
  template <>                                           \
  MLDataType DataTypeImpl::GetTensorType<ELEM_TYPE>() { \
    return TensorType<ELEM_TYPE>::Type();               \
  }

#define ORT_REGISTER_MAP(TYPE)               \
  template <>                                \
  MLDataType MapType<TYPE>::Type() {         \
    static MapType<TYPE> map_type;           \
    return &map_type;                        \
  }                                          \
  template <>                                \
  MLDataType DataTypeImpl::GetType<TYPE>() { \
    return MapType<TYPE>::Type();            \
  }

#define ORT_REGISTER_SEQ(TYPE)               \
  template <>                                \
  MLDataType SequenceType<TYPE>::Type() {    \
    static SequenceType<TYPE> sequence_type; \
    return &sequence_type;                   \
  }                                          \
  template <>                                \
  MLDataType DataTypeImpl::GetType<TYPE>() { \
    return SequenceType<TYPE>::Type();       \
  }

#define ORT_REGISTER_NON_ONNX_TYPE(TYPE)     \
  template <>                                \
  MLDataType NonOnnxType<TYPE>::Type() {     \
    static NonOnnxType<TYPE> non_onnx_type;  \
    return &non_onnx_type;                   \
  }                                          \
  template <>                                \
  MLDataType DataTypeImpl::GetType<TYPE>() { \
    return NonOnnxType<TYPE>::Type();        \
  }

#define ORT_REGISTER_OPAQUE_TYPE(CPPType, Domain, Name)   \
  template <>                                             \
  MLDataType OpaqueType<CPPType, Domain, Name>::Type() {  \
    static OpaqueType<CPPType, Domain, Name> opaque_type; \
    return &opaque_type;                                  \
  }                                                       \
  template <>                                             \
  MLDataType DataTypeImpl::GetType<CPPType>() {           \
    return OpaqueType<CPPType, Domain, Name>::Type();     \
  }
}  // namespace onnxruntime
