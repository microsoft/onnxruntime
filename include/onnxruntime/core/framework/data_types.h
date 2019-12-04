// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <map>
#include <unordered_map>

#include "core/common/common.h"
#include "core/common/exceptions.h"
#include "core/framework/endian.h"

struct OrtValue;

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
using VectorMapStringToFloat = std::vector<MapStringToFloat>;
using VectorMapInt64ToFloat = std::vector<MapInt64ToFloat>;

class DataTypeImpl;
class TensorTypeBase;
class SparseTensorTypeBase;
class NonTensorTypeBase;
class PrimitiveDataTypeBase;

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

inline bool operator<(const MLFloat16& left, const MLFloat16& right) {
  return left.val < right.val;
}

//BFloat16
struct BFloat16 {
  uint16_t val{0};
  explicit BFloat16() = default;
  explicit BFloat16(uint16_t v) : val(v) {}
  explicit BFloat16(float v) {
    if (endian::native == endian::little) {
      std::memcpy(&val, reinterpret_cast<char*>(&v) + sizeof(uint16_t), sizeof(uint16_t));
    } else {
      std::memcpy(&val, &v, sizeof(uint16_t));
    }
  }

  float ToFloat() const {
    float result;
    char* const first = reinterpret_cast<char*>(&result);
    char* const second = first + sizeof(uint16_t);
    if (endian::native == endian::little) {
      std::memset(first, 0, sizeof(uint16_t));
      std::memcpy(second, &val, sizeof(uint16_t));
    } else {
      std::memcpy(first, &val, sizeof(uint16_t));
      std::memset(second, 0, sizeof(uint16_t));
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

inline bool operator<(const BFloat16& left, const BFloat16& right) {
  return left.val < right.val;
}

// DataTypeImpl pointer as unique DataTypeImpl identifier.
using MLDataType = const DataTypeImpl*;
// be used with class MLValue
using DeleteFunc = void (*)(void*);
using CreateFunc = void* (*)();

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

  virtual bool IsSparseTensorType() const {
    return false;
  }

  // Returns this if this is of tensor-type and null otherwise
  virtual const TensorTypeBase* AsTensorType() const {
    return nullptr;
  }

  // Returns this if this is of sparse-tensor-type and null otherwise
  virtual const SparseTensorTypeBase* AsSparseTensorType() const {
    return nullptr;
  }

  virtual const NonTensorTypeBase* AsNonTensorTypeBase() const {
    return nullptr;
  }

  // Returns this if this is one of the primitive data types (specialization of PrimitiveDataTypeBase)
  // and null otherwise
  virtual const PrimitiveDataTypeBase* AsPrimitiveDataType() const {
    return nullptr;
  }

  // Return the type meta that we are using in the runtime.
  template <typename T>
  static MLDataType GetType();

  // Return the types for a concrete tensor type, like Tensor_Float
  template <typename elemT>
  static MLDataType GetTensorType();

  template <typename elemT>
  static MLDataType GetSequenceTensorType();

  // Return the MLDataType for a concrete sparse tensor type.
  template <typename elemT>
  static MLDataType GetSparseTensorType();

  /**
   * Convert an ONNX TypeProto to onnxruntime DataTypeImpl.
   * However, this conversion is lossy. Don't try to use 'this->GetTypeProto()' converting it back.
   * Even though GetTypeProto() will not have the original information, it will still have enough to correctly
   * map to MLDataType.
   * \param proto
   */
  static MLDataType TypeFromProto(const ONNX_NAMESPACE::TypeProto& proto);

  static const TensorTypeBase* TensorTypeFromONNXEnum(int type);
  static const SparseTensorTypeBase* SparseTensorTypeFromONNXEnum(int type);
  static const NonTensorTypeBase* SequenceTensorTypeFromONNXEnum(int type);

  static const char* ToString(MLDataType type);
  // Registers ONNX_NAMESPACE::DataType (internalized string) with
  // MLDataType. DataType is produced by internalizing an instance of
  // TypeProto contained within MLDataType
  static void RegisterDataType(MLDataType);
  static MLDataType GetDataType(const std::string&);

  static const std::vector<MLDataType>& AllTensorTypes();
  static const std::vector<MLDataType>& AllSequenceTensorTypes();
  static const std::vector<MLDataType>& AllFixedSizeTensorTypes();
  static const std::vector<MLDataType>& AllNumericTensorTypes();
  static const std::vector<MLDataType>& AllIEEEFloatTensorTypes();
  static const std::vector<MLDataType>& AllFixedSizeTensorExceptHalfTypes();
  static const std::vector<MLDataType>& AllIEEEFloatTensorExceptHalfTypes();
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
struct TensorElementTypeSetter {
  static void SetTensorElementType(ONNX_NAMESPACE::TypeProto&);
  static void SetMapKeyType(ONNX_NAMESPACE::TypeProto&);
  static int32_t GetElementType();
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

/// Use "IsSparseTensorContainedType<T>::value" to test if a type T
/// is permitted as the element-type of a sparse-tensor.

template <typename T>
struct IsSparseTensorContainedType : public IsAnyOf<T, float, uint8_t, int8_t, uint16_t, int16_t,
                                                    int32_t, int64_t, bool, MLFloat16,
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
    TensorElementTypeSetter<K>::SetMapKeyType(proto);
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
    TensorElementTypeSetter<elemT>::SetTensorElementType(this->mutable_type_proto());
  }
};

/// Common base-class for all sparse-tensors (with different element types).
class SparseTensorTypeBase : public DataTypeImpl {
 public:
  static MLDataType Type();

  bool IsSparseTensorType() const override {
    return true;
  }

  const SparseTensorTypeBase* AsSparseTensorType() const override {
    return this;
  }

  bool IsCompatible(const ONNX_NAMESPACE::TypeProto& type_proto) const override;

  size_t Size() const override;

  DeleteFunc GetDeleteFunc() const override;

  const ONNX_NAMESPACE::TypeProto* GetTypeProto() const override;

  virtual MLDataType GetElementType() const {
    // should never reach here.
    ORT_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }

  SparseTensorTypeBase(const SparseTensorTypeBase&) = delete;
  SparseTensorTypeBase& operator=(const SparseTensorTypeBase&) = delete;

 protected:
  ONNX_NAMESPACE::TypeProto& mutable_type_proto();

  SparseTensorTypeBase();
  ~SparseTensorTypeBase() override;

 private:
  struct Impl;
  Impl* impl_;
};

template <typename elemT>
class SparseTensorType : public SparseTensorTypeBase {
 public:
  static_assert(data_types_internal::IsSparseTensorContainedType<elemT>::value,
                "Requires one of the sparse-tensor fundamental types");

  static MLDataType Type();

  /// Return a MLDataType representing the element-type
  MLDataType GetElementType() const override {
    return DataTypeImpl::GetType<elemT>();
  }

 private:
  SparseTensorType() {
    using namespace data_types_internal;
    TensorElementTypeSetter<elemT>::SetSparseTensorElementType(mutable_type_proto());
  }
};

/**
  * \brief Provide a specialization for your C++ Non-tensor type
  *        so your implementation FromDataTypeContainer/ToDataTypeContainer
  *        functions correctly. Otherwise you get a default implementation
  *        which may not be what you need/want.
  *
  * This class is used to create OrtValue, fetch data from OrtValue via
  * C/C++ APIs
  */
template <class T>
struct NonTensorTypeConverter {
  static void FromContainer(MLDataType /*dtype*/, const void* /*data*/, size_t /*data_size*/, OrtValue& /*output*/) {
    ORT_THROW("Not implemented");
  }
  static void ToContainer(const OrtValue& /*input*/, size_t /*data_size*/, void* /*data*/) {
    ORT_THROW("Not implemented");
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

  const NonTensorTypeBase* AsNonTensorTypeBase() const override {
    return this;
  }

  // \brief Override for Non-tensor types to initialize non-tensor CPP
  // data representation from data. The caller of the interface
  // should have a shared definition of the data which is used to initialize
  // CPP data representation. This is used from C API.
  //
  // \param data - pointer to a data container structure non_tensor type specific
  // \param data_size - size of the data container structure, used for rudimentary checks
  // \param output - reference to a default constructed non-tensor type
  // \returns OrtValue
  // \throw if there is an error
  virtual void FromDataContainer(const void* data, size_t data_size, OrtValue& output) const;

  // \brief Override for Non-tensor types to fetch data from the internal CPP data representation
  // The caller of the interface should have a shared definition of the data which is used to initialize
  // CPP data representation. This is used from C API.
  //
  // \param input - OrtValue containing data
  // \param data_size - size of the structure that is being passed for receiving data, used for
  //                    validation
  // \param data - pointer to receiving data structure
  virtual void ToDataContainer(const OrtValue& input, size_t data_size, void* data) const;

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
    return []() -> void* { return new T(); };
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

  void FromDataContainer(const void* data, size_t data_size, OrtValue& output) const override {
    NonTensorTypeConverter<T>::FromContainer(this, data, data_size, output);
  }

  void ToDataContainer(const OrtValue& input, size_t data_size, void* data) const override {
    NonTensorTypeConverter<T>::ToContainer(input, data_size, data);
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

class PrimitiveDataTypeBase : public DataTypeImpl {
 public:
  bool IsCompatible(const ONNX_NAMESPACE::TypeProto&) const override {
    return false;
  }

  const PrimitiveDataTypeBase* AsPrimitiveDataType() const override final {
    return this;
  }

  const ONNX_NAMESPACE::TypeProto* GetTypeProto() const final {
    return nullptr;
  }

  int32_t GetDataType() const {
    return data_type_;
  }

 protected:
  PrimitiveDataTypeBase() = default;

  void SetDataType(int32_t data_type) {
    data_type_ = data_type;
  }

 private:
  int32_t data_type_;
};

template <typename T>
class PrimitiveDataType : public PrimitiveDataTypeBase {
 private:
  static void Delete(void* p) {
    delete static_cast<T*>(p);
  }

 public:
  static MLDataType Type();

  size_t Size() const override {
    return sizeof(T);
  }

  DeleteFunc GetDeleteFunc() const override {
    return &Delete;
  }

 private:
  PrimitiveDataType() {
    this->SetDataType(data_types_internal::TensorElementTypeSetter<T>::GetElementType());
  }
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

#define ORT_REGISTER_SPARSE_TENSOR_TYPE(ELEM_TYPE)            \
  template <>                                                 \
  MLDataType SparseTensorType<ELEM_TYPE>::Type() {            \
    static SparseTensorType<ELEM_TYPE> tensor_type;           \
    return &tensor_type;                                      \
  }                                                           \
  template <>                                                 \
  MLDataType DataTypeImpl::GetSparseTensorType<ELEM_TYPE>() { \
    return SparseTensorType<ELEM_TYPE>::Type();               \
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

#define ORT_REGISTER_SEQ_TENSOR_TYPE(ELEM_TYPE)                 \
  template <>                                                   \
  MLDataType SequenceTensorType<ELEM_TYPE>::Type() {            \
    static SequenceTensorType<ELEM_TYPE> sequence_tensor_type;  \
    return &sequence_tensor_type;                               \
  }                                                             \
  template <>                                                   \
  MLDataType DataTypeImpl::GetSequenceTensorType<ELEM_TYPE>() { \
    return SequenceTensorType<ELEM_TYPE>::Type();               \
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

#define ORT_REGISTER_PRIM_TYPE(TYPE)               \
  template <>                                      \
  MLDataType PrimitiveDataType<TYPE>::Type() {     \
    static PrimitiveDataType<TYPE> prim_data_type; \
    return &prim_data_type;                        \
  }                                                \
  template <>                                      \
  MLDataType DataTypeImpl::GetType<TYPE>() {       \
    return PrimitiveDataType<TYPE>::Type();        \
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
