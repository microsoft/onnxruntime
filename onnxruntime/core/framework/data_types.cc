// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/data_types.h"
#include "core/framework/tensor.h"
#include "core/framework/TensorSeq.h"
#include "core/framework/sparse_tensor.h"
#include "core/framework/data_types_internal.h"
#include "core/graph/onnx_protobuf.h"

#ifdef MICROSOFT_AUTOML
#include "automl_ops/automl_types.h"
#endif

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include "onnx/defs/data_type_utils.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
// Return the MLDataType used for a generic Tensor
template <>
MLDataType DataTypeImpl::GetType<Tensor>() {
  return TensorTypeBase::Type();
}

}  // namespace onnxruntime

// This conflics with the above GetType<>() specialization
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {

// Return the MLDataType used for a generic SparseTensor
template <>
MLDataType DataTypeImpl::GetType<SparseTensor>() {
  return SparseTensorTypeBase::Type();
}

static bool IsTensorTypeScalar(const ONNX_NAMESPACE::TypeProto_Tensor& tensor_type_proto) {
  int sz = tensor_type_proto.shape().dim_size();
  return sz == 0 || sz == 1;
}

namespace data_types_internal {

template <typename T>
struct TensorElementTypeSetter<T> {
  static void SetTensorElementType(ONNX_NAMESPACE::TypeProto& proto) {
    proto.mutable_tensor_type()->set_elem_type(utils::ToTensorProtoElementType<T>());
  }
  static void SetSparseTensorElementType(ONNX_NAMESPACE::TypeProto& proto) {
    proto.mutable_sparse_tensor_type()->set_elem_type(utils::ToTensorProtoElementType<T>());
  }
  static void SetMapKeyType(ONNX_NAMESPACE::TypeProto& proto) {
    proto.mutable_map_type()->set_key_type(utils::ToTensorProtoElementType<T>());
  }
  constexpr static int32_t GetElementType () {
    return utils::ToTensorProtoElementType<T>();
  }
};

// Pre-instantiate
template struct
    TensorElementTypeSetter<float>;
template struct
    TensorElementTypeSetter<uint8_t>;
template struct
    TensorElementTypeSetter<int8_t>;
template struct
    TensorElementTypeSetter<uint16_t>;
template struct
    TensorElementTypeSetter<int16_t>;
template struct
    TensorElementTypeSetter<int32_t>;
template struct
    TensorElementTypeSetter<int64_t>;
template struct
    TensorElementTypeSetter<std::string>;
template struct
    TensorElementTypeSetter<bool>;
template struct
    TensorElementTypeSetter<MLFloat16>;
template struct
    TensorElementTypeSetter<double>;
template struct
    TensorElementTypeSetter<uint32_t>;
template struct
    TensorElementTypeSetter<uint64_t>;
template struct
    TensorElementTypeSetter<BFloat16>;

void CopyMutableMapValue(const ONNX_NAMESPACE::TypeProto& value_proto,
                         ONNX_NAMESPACE::TypeProto& map_proto) {
  map_proto.mutable_map_type()->mutable_value_type()->CopyFrom(value_proto);
}

void CopyMutableSeqElement(const ONNX_NAMESPACE::TypeProto& elem_proto,
                           ONNX_NAMESPACE::TypeProto& proto) {
  proto.mutable_sequence_type()->mutable_elem_type()->CopyFrom(elem_proto);
}

void AssignOpaqueDomainName(const char* domain, const char* name,
                            ONNX_NAMESPACE::TypeProto& proto) {
  auto* mutable_opaque = proto.mutable_opaque_type();
  mutable_opaque->mutable_domain()->assign(domain);
  mutable_opaque->mutable_name()->assign(name);
}

bool IsCompatible(const ONNX_NAMESPACE::TypeProto_Tensor& tensor_proto,
                  const ONNX_NAMESPACE::TypeProto_Tensor& type_proto);

bool IsCompatible(const ONNX_NAMESPACE::TypeProto_SparseTensor& tensor_proto,
                  const ONNX_NAMESPACE::TypeProto_SparseTensor& type_proto);

bool IsCompatible(const ONNX_NAMESPACE::TypeProto_Map& map_proto,
                  const ONNX_NAMESPACE::TypeProto_Map& type_proto);

bool IsCompatible(const ONNX_NAMESPACE::TypeProto_Sequence& sequence_proto,
                  const ONNX_NAMESPACE::TypeProto_Sequence& type_proto);

bool IsCompatible(const ONNX_NAMESPACE::TypeProto_Opaque& opaque_proto,
                  const ONNX_NAMESPACE::TypeProto_Opaque& type_proto);

bool IsCompatible(const ONNX_NAMESPACE::TypeProto_Tensor& tensor_proto,
                  const ONNX_NAMESPACE::TypeProto_Tensor& type_proto) {
  return type_proto.elem_type() == tensor_proto.elem_type();
  /* Currently all Tensors with all kinds of shapes
     are mapped into the same MLDataType (same element type)
     so we omit shape from IsCompatible consideration
     */
}

bool IsCompatible(const ONNX_NAMESPACE::TypeProto_Map& map_proto,
                  const ONNX_NAMESPACE::TypeProto_Map& type_proto) {
  const auto& lhs = map_proto;
  const auto& rhs = type_proto;
  bool result = true;
  if (lhs.key_type() == rhs.key_type() &&
      lhs.value_type().value_case() == rhs.value_type().value_case()) {
    switch (lhs.value_type().value_case()) {
      case TypeProto::ValueCase::kTensorType:
        result = IsCompatible(lhs.value_type().tensor_type(), rhs.value_type().tensor_type());
        break;
      case TypeProto::ValueCase::kSequenceType:
        result = IsCompatible(lhs.value_type().sequence_type(), rhs.value_type().sequence_type());
        break;
      case TypeProto::ValueCase::kMapType:
        result = IsCompatible(lhs.value_type().map_type(), rhs.value_type().map_type());
        break;
      case TypeProto::ValueCase::kOpaqueType:
        result = IsCompatible(lhs.value_type().opaque_type(), rhs.value_type().opaque_type());
        break;
      case TypeProto::ValueCase::kSparseTensorType:
        result = IsCompatible(lhs.value_type().sparse_tensor_type(), rhs.value_type().sparse_tensor_type());
        break;
      default:
        ORT_ENFORCE(false);
        break;
    }
  } else {
    result = false;
  }
  return result;
}

bool IsCompatible(const ONNX_NAMESPACE::TypeProto_Sequence& sequence_proto,
                  const ONNX_NAMESPACE::TypeProto_Sequence& type_proto) {
  bool result = true;
  const auto& lhs = sequence_proto;
  const auto& rhs = type_proto;
  if (lhs.elem_type().value_case() == rhs.elem_type().value_case()) {
    switch (lhs.elem_type().value_case()) {
      case TypeProto::ValueCase::kTensorType:
        result = IsCompatible(lhs.elem_type().tensor_type(), rhs.elem_type().tensor_type());
        break;
      case TypeProto::ValueCase::kSequenceType:
        result = IsCompatible(lhs.elem_type().sequence_type(), rhs.elem_type().sequence_type());
        break;
      case TypeProto::ValueCase::kMapType:
        result = IsCompatible(lhs.elem_type().map_type(), rhs.elem_type().map_type());
        break;
      case TypeProto::ValueCase::kOpaqueType:
        result = IsCompatible(lhs.elem_type().opaque_type(), rhs.elem_type().opaque_type());
        break;
      case TypeProto::ValueCase::kSparseTensorType:
        result = IsCompatible(lhs.elem_type().sparse_tensor_type(), rhs.elem_type().sparse_tensor_type());
        break;
      default:
        ORT_ENFORCE(false);
        break;
    }
  } else {
    result = false;
  }
  return result;
}
bool IsCompatible(const ONNX_NAMESPACE::TypeProto_Opaque& opaque_proto, const ONNX_NAMESPACE::TypeProto_Opaque& type_proto) {
  const auto& lhs = opaque_proto;
  const auto& rhs = type_proto;
  bool lhs_domain = utils::HasDomain(lhs);
  bool rhs_domain = utils::HasDomain(rhs);

  if ((lhs_domain != rhs_domain) ||
      (lhs_domain && rhs_domain && lhs.domain() != lhs.domain())) {
    return false;
  }

  bool lhs_name = utils::HasName(lhs);
  bool rhs_name = utils::HasName(rhs);

  return !((lhs_name != rhs_name) ||
           (lhs_name && rhs_name && lhs.name() != rhs.name()));
}

bool IsCompatible(const ONNX_NAMESPACE::TypeProto_SparseTensor& tensor_proto,
                  const ONNX_NAMESPACE::TypeProto_SparseTensor& type_proto) {
  return type_proto.elem_type() == tensor_proto.elem_type();
}

void RegisterAllProtos(const std::function<void(MLDataType)>& /*reg_fn*/);

class DataTypeRegistry {
  std::unordered_map<DataType, MLDataType> mapping_;

  DataTypeRegistry() {
    RegisterAllProtos([this](MLDataType mltype) { RegisterDataType(mltype); });
#ifdef MICROSOFT_AUTOML
    automl::RegisterAutoMLTypes([this](MLDataType mltype) { RegisterDataType(mltype); });
#endif
  }

  ~DataTypeRegistry() = default;

 public:
  DataTypeRegistry(const DataTypeRegistry&) = delete;
  DataTypeRegistry& operator=(const DataTypeRegistry&) = delete;

  static DataTypeRegistry& instance() {
    static DataTypeRegistry inst;
    return inst;
  }

  void RegisterDataType(MLDataType mltype) {
    using namespace ONNX_NAMESPACE;
    const auto* proto = mltype->GetTypeProto();
    ORT_ENFORCE(proto != nullptr, "Only ONNX MLDataType can be registered");
    DataType type = Utils::DataTypeUtils::ToType(*proto);
    auto p = mapping_.insert(std::make_pair(type, mltype));
    ORT_ENFORCE(p.second, "We do not expect duplicate registration of types for: ", type);
  }

  MLDataType GetMLDataType(const ONNX_NAMESPACE::TypeProto& proto) const {
    using namespace ONNX_NAMESPACE;
    DataType type = Utils::DataTypeUtils::ToType(proto);
    auto p = mapping_.find(type);
    if (p != mapping_.end()) {
      return p->second;
    }
    return nullptr;
  }

  MLDataType GetMLDataType(const std::string& data_type) const {
    using namespace ONNX_NAMESPACE;
    DataType dtype = Utils::DataTypeUtils::ToType(data_type);
    if (dtype == nullptr) {
      return nullptr;
    }
    auto hit = mapping_.find(dtype);
    if (hit == mapping_.end()) {
      return nullptr;
    }
    return hit->second;
  }
};

struct TypeProtoImpl {
  const TypeProto* GetProto() const {
    return &proto_;
  }
  TypeProto& mutable_type_proto() {
    return proto_;
  }

  TypeProto proto_;
};

}  // namespace data_types_internal

/// TensorTypeBase
struct TensorTypeBase::Impl : public data_types_internal::TypeProtoImpl {
};

const ONNX_NAMESPACE::TypeProto* TensorTypeBase::GetTypeProto() const {
  return impl_->GetProto();
}

TensorTypeBase::TensorTypeBase() : impl_(new Impl()) {}
TensorTypeBase::~TensorTypeBase() {
  delete impl_;
}

size_t TensorTypeBase::Size() const {
  return sizeof(Tensor);
}

template <typename T>
static void Delete(void* p) {
  delete static_cast<T*>(p);
}

DeleteFunc TensorTypeBase::GetDeleteFunc() const {
  return &Delete<Tensor>;
}

ONNX_NAMESPACE::TypeProto& TensorTypeBase::mutable_type_proto() {
  return impl_->mutable_type_proto();
}

bool TensorTypeBase::IsCompatible(const ONNX_NAMESPACE::TypeProto& type_proto) const {
  const auto* thisProto = GetTypeProto();
  ORT_ENFORCE(thisProto->value_case() == TypeProto::ValueCase::kTensorType);
  ORT_ENFORCE(utils::HasElemType(thisProto->tensor_type()));

  if (&type_proto == thisProto) {
    return true;
  }

  if (type_proto.value_case() != TypeProto::ValueCase::kTensorType) {
    return false;
  }

  return data_types_internal::IsCompatible(thisProto->tensor_type(), type_proto.tensor_type());
}

MLDataType TensorTypeBase::Type() {
  static TensorTypeBase tensor_base;
  return &tensor_base;
}

/// SparseTensor

struct SparseTensorTypeBase::Impl : public data_types_internal::TypeProtoImpl {
};

SparseTensorTypeBase::SparseTensorTypeBase() : impl_(new Impl()) {}
SparseTensorTypeBase::~SparseTensorTypeBase() {
  delete impl_;
}

bool SparseTensorTypeBase::IsCompatible(const ONNX_NAMESPACE::TypeProto& type_proto) const {
  const auto* thisProto = GetTypeProto();
  if (&type_proto == thisProto) {
    return true;
  }
  if (type_proto.value_case() != TypeProto::ValueCase::kSparseTensorType) {
    return false;
  }

  ORT_ENFORCE(thisProto->value_case() == TypeProto::ValueCase::kSparseTensorType);
  ORT_ENFORCE(utils::HasElemType(thisProto->sparse_tensor_type()));

  return data_types_internal::IsCompatible(thisProto->sparse_tensor_type(), type_proto.sparse_tensor_type());
}

size_t SparseTensorTypeBase::Size() const {
  return sizeof(SparseTensor);
}

DeleteFunc SparseTensorTypeBase::GetDeleteFunc() const {
  return &Delete<SparseTensor>;
}

const ONNX_NAMESPACE::TypeProto* SparseTensorTypeBase::GetTypeProto() const {
  return impl_->GetProto();
}

ONNX_NAMESPACE::TypeProto& SparseTensorTypeBase::mutable_type_proto() {
  return impl_->mutable_type_proto();
}

MLDataType SparseTensorTypeBase::Type() {
  static SparseTensorTypeBase sparse_tensor_base;
  return &sparse_tensor_base;
}

/// NoTensorTypeBase
struct NonTensorTypeBase::Impl : public data_types_internal::TypeProtoImpl {};

NonTensorTypeBase::NonTensorTypeBase() : impl_(new Impl()) {
}

NonTensorTypeBase::~NonTensorTypeBase() {
  delete impl_;
}

ONNX_NAMESPACE::TypeProto& NonTensorTypeBase::mutable_type_proto() {
  return impl_->mutable_type_proto();
}

const ONNX_NAMESPACE::TypeProto* NonTensorTypeBase::GetTypeProto() const {
  return impl_->GetProto();
}

bool NonTensorTypeBase::IsMapCompatible(const ONNX_NAMESPACE::TypeProto& type_proto) const {
  const auto* thisProto = impl_->GetProto();
  if (&type_proto == thisProto) {
    return true;
  }
  if (type_proto.value_case() != TypeProto::ValueCase::kMapType) {
    return false;
  }
  ORT_ENFORCE(thisProto->value_case() == TypeProto::ValueCase::kMapType);
  ORT_ENFORCE(utils::HasKeyType(thisProto->map_type()));
  ORT_ENFORCE(utils::HasKeyType(thisProto->map_type()));
  return data_types_internal::IsCompatible(thisProto->map_type(), type_proto.map_type());
}

bool NonTensorTypeBase::IsSequenceCompatible(const ONNX_NAMESPACE::TypeProto& type_proto) const {
  const auto* thisProto = impl_->GetProto();
  if (&type_proto == thisProto) {
    return true;
  }
  if (type_proto.value_case() != TypeProto::ValueCase::kSequenceType) {
    return false;
  }
  ORT_ENFORCE(thisProto->value_case() == TypeProto::ValueCase::kSequenceType);
  ORT_ENFORCE(utils::HasElemType(thisProto->sequence_type()));
  return data_types_internal::IsCompatible(thisProto->sequence_type(), type_proto.sequence_type());
}

bool NonTensorTypeBase::IsOpaqueCompatible(const ONNX_NAMESPACE::TypeProto& type_proto) const {
  const auto* thisProto = impl_->GetProto();
  if (&type_proto == thisProto) {
    return true;
  }
  if (type_proto.value_case() != TypeProto::ValueCase::kOpaqueType) {
    return false;
  }
  ORT_ENFORCE(thisProto->value_case() == TypeProto::ValueCase::kOpaqueType);
  return data_types_internal::IsCompatible(thisProto->opaque_type(), type_proto.opaque_type());
}

// The below two APIs must be implemented in the derived types to be used
void NonTensorTypeBase::FromDataContainer(const void* /* data */, size_t /*data_size*/, OrtValue& /* output */) const {
  ORT_ENFORCE(false, "Not implemented");
}

void NonTensorTypeBase::ToDataContainer(const OrtValue& /* input */, size_t /*data_size */, void* /* data */) const {
  ORT_ENFORCE(false, "Not implemented");
}

ORT_REGISTER_TENSOR_TYPE(int32_t);
ORT_REGISTER_TENSOR_TYPE(float);
ORT_REGISTER_TENSOR_TYPE(bool);
ORT_REGISTER_TENSOR_TYPE(std::string);
ORT_REGISTER_TENSOR_TYPE(int8_t);
ORT_REGISTER_TENSOR_TYPE(uint8_t);
ORT_REGISTER_TENSOR_TYPE(uint16_t);
ORT_REGISTER_TENSOR_TYPE(int16_t);
ORT_REGISTER_TENSOR_TYPE(int64_t);
ORT_REGISTER_TENSOR_TYPE(double);
ORT_REGISTER_TENSOR_TYPE(uint32_t);
ORT_REGISTER_TENSOR_TYPE(uint64_t);
ORT_REGISTER_TENSOR_TYPE(MLFloat16);
ORT_REGISTER_TENSOR_TYPE(BFloat16);

ORT_REGISTER_SPARSE_TENSOR_TYPE(int32_t);
ORT_REGISTER_SPARSE_TENSOR_TYPE(float);
ORT_REGISTER_SPARSE_TENSOR_TYPE(bool);
// ORT_REGISTER_SPARSE_TENSOR_TYPE(std::string);
ORT_REGISTER_SPARSE_TENSOR_TYPE(int8_t);
ORT_REGISTER_SPARSE_TENSOR_TYPE(uint8_t);
ORT_REGISTER_SPARSE_TENSOR_TYPE(uint16_t);
ORT_REGISTER_SPARSE_TENSOR_TYPE(int16_t);
ORT_REGISTER_SPARSE_TENSOR_TYPE(int64_t);
ORT_REGISTER_SPARSE_TENSOR_TYPE(double);
ORT_REGISTER_SPARSE_TENSOR_TYPE(uint32_t);
ORT_REGISTER_SPARSE_TENSOR_TYPE(uint64_t);
ORT_REGISTER_SPARSE_TENSOR_TYPE(MLFloat16);
ORT_REGISTER_SPARSE_TENSOR_TYPE(BFloat16);

ORT_REGISTER_MAP(MapStringToString);
ORT_REGISTER_MAP(MapStringToInt64);
ORT_REGISTER_MAP(MapStringToFloat);
ORT_REGISTER_MAP(MapStringToDouble);
ORT_REGISTER_MAP(MapInt64ToString);
ORT_REGISTER_MAP(MapInt64ToInt64);
ORT_REGISTER_MAP(MapInt64ToFloat);
ORT_REGISTER_MAP(MapInt64ToDouble);

// Register sequence of tensor types
ORT_REGISTER_SEQ(TensorSeq)  // required to ensure GetType<TensorSeq> works
ORT_REGISTER_SEQ_TENSOR_TYPE(int32_t);
ORT_REGISTER_SEQ_TENSOR_TYPE(float);
ORT_REGISTER_SEQ_TENSOR_TYPE(bool);
ORT_REGISTER_SEQ_TENSOR_TYPE(std::string);
ORT_REGISTER_SEQ_TENSOR_TYPE(int8_t);
ORT_REGISTER_SEQ_TENSOR_TYPE(uint8_t);
ORT_REGISTER_SEQ_TENSOR_TYPE(uint16_t);
ORT_REGISTER_SEQ_TENSOR_TYPE(int16_t);
ORT_REGISTER_SEQ_TENSOR_TYPE(int64_t);
ORT_REGISTER_SEQ_TENSOR_TYPE(double);
ORT_REGISTER_SEQ_TENSOR_TYPE(uint32_t);
ORT_REGISTER_SEQ_TENSOR_TYPE(uint64_t);
ORT_REGISTER_SEQ_TENSOR_TYPE(MLFloat16);
ORT_REGISTER_SEQ_TENSOR_TYPE(BFloat16);

ORT_REGISTER_SEQ(VectorMapStringToFloat);
ORT_REGISTER_SEQ(VectorMapInt64ToFloat);

// Used for Tensor Proto registrations
#define REGISTER_TENSOR_PROTO(TYPE, reg_fn)                  \
  {                                                          \
    MLDataType mltype = DataTypeImpl::GetTensorType<TYPE>(); \
    reg_fn(mltype);                                          \
  }

#define REGISTER_SEQ_TENSOR_PROTO(TYPE, reg_fn)                      \
  {                                                                  \
    MLDataType mltype = DataTypeImpl::GetSequenceTensorType<TYPE>(); \
    reg_fn(mltype);                                                  \
  }

#define REGISTER_SPARSE_TENSOR_PROTO(TYPE, reg_fn)                 \
  {                                                                \
    MLDataType mltype = DataTypeImpl::GetSparseTensorType<TYPE>(); \
    reg_fn(mltype);                                                \
  }

#define REGISTER_ONNX_PROTO(TYPE, reg_fn)              \
  {                                                    \
    MLDataType mltype = DataTypeImpl::GetType<TYPE>(); \
    reg_fn(mltype);                                    \
  }

namespace data_types_internal {

void RegisterAllProtos(const std::function<void(MLDataType)>& reg_fn) {
  REGISTER_TENSOR_PROTO(int32_t, reg_fn);
  REGISTER_TENSOR_PROTO(float, reg_fn);
  REGISTER_TENSOR_PROTO(bool, reg_fn);
  REGISTER_TENSOR_PROTO(std::string, reg_fn);
  REGISTER_TENSOR_PROTO(int8_t, reg_fn);
  REGISTER_TENSOR_PROTO(uint8_t, reg_fn);
  REGISTER_TENSOR_PROTO(uint16_t, reg_fn);
  REGISTER_TENSOR_PROTO(int16_t, reg_fn);
  REGISTER_TENSOR_PROTO(int64_t, reg_fn);
  REGISTER_TENSOR_PROTO(double, reg_fn);
  REGISTER_TENSOR_PROTO(uint32_t, reg_fn);
  REGISTER_TENSOR_PROTO(uint64_t, reg_fn);
  REGISTER_TENSOR_PROTO(MLFloat16, reg_fn);
  REGISTER_TENSOR_PROTO(BFloat16, reg_fn);

  REGISTER_SPARSE_TENSOR_PROTO(int32_t, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(float, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(bool, reg_fn);
  // REGISTER_SPARSE_TENSOR_PROTO(std::string, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(int8_t, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(uint8_t, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(uint16_t, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(int16_t, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(int64_t, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(double, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(uint32_t, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(uint64_t, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(MLFloat16, reg_fn);
  REGISTER_SPARSE_TENSOR_PROTO(BFloat16, reg_fn);

  REGISTER_ONNX_PROTO(MapStringToString, reg_fn);
  REGISTER_ONNX_PROTO(MapStringToInt64, reg_fn);
  REGISTER_ONNX_PROTO(MapStringToFloat, reg_fn);
  REGISTER_ONNX_PROTO(MapStringToDouble, reg_fn);
  REGISTER_ONNX_PROTO(MapInt64ToString, reg_fn);
  REGISTER_ONNX_PROTO(MapInt64ToInt64, reg_fn);
  REGISTER_ONNX_PROTO(MapInt64ToFloat, reg_fn);
  REGISTER_ONNX_PROTO(MapInt64ToDouble, reg_fn);

  REGISTER_SEQ_TENSOR_PROTO(int32_t, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(float, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(bool, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(std::string, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(int8_t, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(uint8_t, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(uint16_t, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(int16_t, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(int64_t, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(double, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(uint32_t, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(uint64_t, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(MLFloat16, reg_fn);
  REGISTER_SEQ_TENSOR_PROTO(BFloat16, reg_fn);

  REGISTER_ONNX_PROTO(VectorMapStringToFloat, reg_fn);
  REGISTER_ONNX_PROTO(VectorMapInt64ToFloat, reg_fn);
}
}  // namespace data_types_internal

void DataTypeImpl::RegisterDataType(MLDataType mltype) {
  data_types_internal::DataTypeRegistry::instance().RegisterDataType(mltype);
}

MLDataType DataTypeImpl::GetDataType(const std::string& data_type) {
  return data_types_internal::DataTypeRegistry::instance().GetMLDataType(data_type);
}

const char* DataTypeImpl::ToString(MLDataType type) {
  if (type == DataTypeImpl::GetTensorType<float>()) {
    return "tensor(float)";
  }
  if (type == DataTypeImpl::GetTensorType<bool>()) {
    return "tensor(bool)";
  }

  if (type == DataTypeImpl::GetTensorType<int32_t>()) {
    return "tensor(int32)";
  }

  if (type == DataTypeImpl::GetTensorType<double>()) {
    return "tensor(double)";
  }

  if (type == DataTypeImpl::GetTensorType<std::string>()) {
    return "tensor(string)";
  }

  if (type == DataTypeImpl::GetTensorType<uint8_t>()) {
    return "tensor(uint8)";
  }

  if (type == DataTypeImpl::GetTensorType<uint16_t>()) {
    return "tensor(uint16)";
  }

  if (type == DataTypeImpl::GetTensorType<int16_t>()) {
    return "tensor(int16)";
  }

  if (type == DataTypeImpl::GetTensorType<int64_t>()) {
    return "tensor(int64)";
  }

  if (type == DataTypeImpl::GetTensorType<uint32_t>()) {
    return "tensor(uint32)";
  }

  if (type == DataTypeImpl::GetTensorType<uint64_t>()) {
    return "tensor(uint64)";
  }

  if (type == DataTypeImpl::GetTensorType<MLFloat16>()) {
    return "tensor(MLFloat16)";
  }
  if (type == DataTypeImpl::GetTensorType<BFloat16>()) {
    return "tensor(bfloat16)";
  }
  return "unknown";
}
const TensorTypeBase* DataTypeImpl::TensorTypeFromONNXEnum(int type) {
  switch (type) {
    case TensorProto_DataType_FLOAT:
      return DataTypeImpl::GetTensorType<float>()->AsTensorType();
    case TensorProto_DataType_BOOL:
      return DataTypeImpl::GetTensorType<bool>()->AsTensorType();
    case TensorProto_DataType_INT32:
      return DataTypeImpl::GetTensorType<int32_t>()->AsTensorType();
    case TensorProto_DataType_DOUBLE:
      return DataTypeImpl::GetTensorType<double>()->AsTensorType();
    case TensorProto_DataType_STRING:
      return DataTypeImpl::GetTensorType<std::string>()->AsTensorType();
    case TensorProto_DataType_UINT8:
      return DataTypeImpl::GetTensorType<uint8_t>()->AsTensorType();
    case TensorProto_DataType_UINT16:
      return DataTypeImpl::GetTensorType<uint16_t>()->AsTensorType();
    case TensorProto_DataType_INT8:
      return DataTypeImpl::GetTensorType<int8_t>()->AsTensorType();
    case TensorProto_DataType_INT16:
      return DataTypeImpl::GetTensorType<int16_t>()->AsTensorType();
    case TensorProto_DataType_INT64:
      return DataTypeImpl::GetTensorType<int64_t>()->AsTensorType();
    case TensorProto_DataType_UINT32:
      return DataTypeImpl::GetTensorType<uint32_t>()->AsTensorType();
    case TensorProto_DataType_UINT64:
      return DataTypeImpl::GetTensorType<uint64_t>()->AsTensorType();
    case TensorProto_DataType_FLOAT16:
      return DataTypeImpl::GetTensorType<MLFloat16>()->AsTensorType();
    case TensorProto_DataType_BFLOAT16:
      return DataTypeImpl::GetTensorType<BFloat16>()->AsTensorType();
    default:
      ORT_NOT_IMPLEMENTED("tensor type ", type, " is not supported");
  }
}

const NonTensorTypeBase* DataTypeImpl::SequenceTensorTypeFromONNXEnum(int type) {
  switch (type) {
    case TensorProto_DataType_FLOAT:
      return DataTypeImpl::GetSequenceTensorType<float>()->AsNonTensorTypeBase();
    case TensorProto_DataType_BOOL:
      return DataTypeImpl::GetSequenceTensorType<bool>()->AsNonTensorTypeBase();
    case TensorProto_DataType_INT32:
      return DataTypeImpl::GetSequenceTensorType<int32_t>()->AsNonTensorTypeBase();
    case TensorProto_DataType_DOUBLE:
      return DataTypeImpl::GetSequenceTensorType<double>()->AsNonTensorTypeBase();
    case TensorProto_DataType_STRING:
      return DataTypeImpl::GetSequenceTensorType<std::string>()->AsNonTensorTypeBase();
    case TensorProto_DataType_UINT8:
      return DataTypeImpl::GetSequenceTensorType<uint8_t>()->AsNonTensorTypeBase();
    case TensorProto_DataType_UINT16:
      return DataTypeImpl::GetSequenceTensorType<uint16_t>()->AsNonTensorTypeBase();
    case TensorProto_DataType_INT8:
      return DataTypeImpl::GetSequenceTensorType<int8_t>()->AsNonTensorTypeBase();
    case TensorProto_DataType_INT16:
      return DataTypeImpl::GetSequenceTensorType<int16_t>()->AsNonTensorTypeBase();
    case TensorProto_DataType_INT64:
      return DataTypeImpl::GetSequenceTensorType<int64_t>()->AsNonTensorTypeBase();
    case TensorProto_DataType_UINT32:
      return DataTypeImpl::GetSequenceTensorType<uint32_t>()->AsNonTensorTypeBase();
    case TensorProto_DataType_UINT64:
      return DataTypeImpl::GetSequenceTensorType<uint64_t>()->AsNonTensorTypeBase();
    case TensorProto_DataType_FLOAT16:
      return DataTypeImpl::GetSequenceTensorType<MLFloat16>()->AsNonTensorTypeBase();
    case TensorProto_DataType_BFLOAT16:
      return DataTypeImpl::GetSequenceTensorType<BFloat16>()->AsNonTensorTypeBase();
    default:
      ORT_NOT_IMPLEMENTED("tensor type ", type, " is not supported");
  }
}

const SparseTensorTypeBase* DataTypeImpl::SparseTensorTypeFromONNXEnum(int type) {
  switch (type) {
    case TensorProto_DataType_FLOAT:
      return reinterpret_cast<const SparseTensorTypeBase*>(DataTypeImpl::GetSparseTensorType<float>());
    case TensorProto_DataType_BOOL:
      return reinterpret_cast<const SparseTensorTypeBase*>(DataTypeImpl::GetSparseTensorType<bool>());
    case TensorProto_DataType_INT32:
      return reinterpret_cast<const SparseTensorTypeBase*>(DataTypeImpl::GetSparseTensorType<int32_t>());
    case TensorProto_DataType_DOUBLE:
      return reinterpret_cast<const SparseTensorTypeBase*>(DataTypeImpl::GetSparseTensorType<double>());
    // case TensorProto_DataType_STRING:
    // return reinterpret_cast<const SparseTensorTypeBase*>(DataTypeImpl::GetSparseTensorType<std::string>());
    case TensorProto_DataType_UINT8:
      return reinterpret_cast<const SparseTensorTypeBase*>(DataTypeImpl::GetSparseTensorType<uint8_t>());
    case TensorProto_DataType_UINT16:
      return reinterpret_cast<const SparseTensorTypeBase*>(DataTypeImpl::GetSparseTensorType<uint16_t>());
    case TensorProto_DataType_INT8:
      return reinterpret_cast<const SparseTensorTypeBase*>(DataTypeImpl::GetSparseTensorType<int8_t>());
    case TensorProto_DataType_INT16:
      return reinterpret_cast<const SparseTensorTypeBase*>(DataTypeImpl::GetSparseTensorType<int16_t>());
    case TensorProto_DataType_INT64:
      return reinterpret_cast<const SparseTensorTypeBase*>(DataTypeImpl::GetSparseTensorType<int64_t>());
    case TensorProto_DataType_UINT32:
      return reinterpret_cast<const SparseTensorTypeBase*>(DataTypeImpl::GetSparseTensorType<uint32_t>());
    case TensorProto_DataType_UINT64:
      return reinterpret_cast<const SparseTensorTypeBase*>(DataTypeImpl::GetSparseTensorType<uint64_t>());
    case TensorProto_DataType_FLOAT16:
      return reinterpret_cast<const SparseTensorTypeBase*>(DataTypeImpl::GetSparseTensorType<MLFloat16>());
    case TensorProto_DataType_BFLOAT16:
      return reinterpret_cast<const SparseTensorTypeBase*>(DataTypeImpl::GetSparseTensorType<BFloat16>());
    default:
      ORT_NOT_IMPLEMENTED("sparse tensor type ", type, " is not supported");
  }
}

MLDataType DataTypeImpl::TypeFromProto(const ONNX_NAMESPACE::TypeProto& proto) {
  const auto& registry = data_types_internal::DataTypeRegistry::instance();

  switch (proto.value_case()) {
    case TypeProto::ValueCase::kTensorType: {
      const auto& tensor_type = proto.tensor_type();
      ORT_ENFORCE(utils::HasElemType(tensor_type));
      return TensorTypeFromONNXEnum(tensor_type.elem_type());
    } break;  // kTensorType
    case TypeProto::ValueCase::kSparseTensorType: {
      const auto& sparse_tensor_type = proto.sparse_tensor_type();
      ORT_ENFORCE(utils::HasElemType(sparse_tensor_type));
      return SparseTensorTypeFromONNXEnum(sparse_tensor_type.elem_type());
    } break;  // kSparseTensorType
    case TypeProto::ValueCase::kMapType: {
      const auto& maptype = proto.map_type();
      auto keytype = maptype.key_type();
      const auto& value_type = maptype.value_type();

      if (value_type.value_case() == TypeProto::ValueCase::kTensorType &&
          IsTensorTypeScalar(value_type.tensor_type())) {
        auto value_elem_type = value_type.tensor_type().elem_type();
        switch (value_elem_type) {
          case TensorProto_DataType_STRING: {
            switch (keytype) {
              case TensorProto_DataType_STRING:
                return DataTypeImpl::GetType<MapStringToString>();
              case TensorProto_DataType_INT64:
                return DataTypeImpl::GetType<MapInt64ToString>();
              default:
                break;
            }
          } break;
          case TensorProto_DataType_INT64:
            switch (keytype) {
              case TensorProto_DataType_STRING:
                return DataTypeImpl::GetType<MapStringToInt64>();
              case TensorProto_DataType_INT64:
                return DataTypeImpl::GetType<MapInt64ToInt64>();
              default:
                break;
            }
            break;
          case TensorProto_DataType_FLOAT:
            switch (keytype) {
              case TensorProto_DataType_STRING:
                return DataTypeImpl::GetType<MapStringToFloat>();
              case TensorProto_DataType_INT64:
                return DataTypeImpl::GetType<MapInt64ToFloat>();
              default:
                break;
            }
            break;
          case TensorProto_DataType_DOUBLE:
            switch (keytype) {
              case TensorProto_DataType_STRING:
                return DataTypeImpl::GetType<MapStringToDouble>();
              case TensorProto_DataType_INT64:
                return DataTypeImpl::GetType<MapInt64ToDouble>();
              default:
                break;
            }
            break;
          default:
            break;
        }
        MLDataType type = registry.GetMLDataType(proto);
        ORT_ENFORCE(type != nullptr, "Map with key type: ", keytype, " value type: ", value_elem_type, " is not registered");
        return type;
      }  // not if(scalar tensor) pre-reg types
      MLDataType type = registry.GetMLDataType(proto);
      if (type == nullptr) {
        DataType str_type = ONNX_NAMESPACE::Utils::DataTypeUtils::ToType(proto);
        ORT_NOT_IMPLEMENTED("type: ", *str_type, " is not registered");
      }
      return type;

    } break;  // kMapType
    case TypeProto::ValueCase::kSequenceType: {
      auto& seq_type = proto.sequence_type();
      auto& val_type = seq_type.elem_type();

      switch (val_type.value_case()) {
        case TypeProto::ValueCase::kMapType: {
          auto& maptype = val_type.map_type();
          auto keytype = maptype.key_type();
          auto& value_type = maptype.value_type();

          if (value_type.value_case() == TypeProto::ValueCase::kTensorType &&
              IsTensorTypeScalar(value_type.tensor_type())) {
            auto value_elem_type = value_type.tensor_type().elem_type();
            switch (value_elem_type) {
              case TensorProto_DataType_FLOAT: {
                switch (keytype) {
                  case TensorProto_DataType_STRING:
                    return DataTypeImpl::GetType<VectorMapStringToFloat>();
                  case TensorProto_DataType_INT64:
                    return DataTypeImpl::GetType<VectorMapInt64ToFloat>();
                  default:
                    break;
                }
              }
              default:
                break;
            }
          }
        }  // MapType
        break;
        case TypeProto::ValueCase::kTensorType: {
          return DataTypeImpl::GetType<TensorSeq>();
        }  // kTensorType
        break;
        default:
          break;
      }  // Sequence value case
    }    // kSequenceType
    break;
    default:
      break;
  }  // proto.value_case()
  MLDataType type = registry.GetMLDataType(proto);
  if (type == nullptr) {
    DataType str_type = ONNX_NAMESPACE::Utils::DataTypeUtils::ToType(proto);
    ORT_NOT_IMPLEMENTED("type: ", *str_type, " is not currently registered or supported");
  }
  return type;
}

//Below are the types the we need to execute the runtime
//They are not compatible with TypeProto in ONNX.
ORT_REGISTER_PRIM_TYPE(int32_t);
ORT_REGISTER_PRIM_TYPE(float);
ORT_REGISTER_PRIM_TYPE(bool);
ORT_REGISTER_PRIM_TYPE(std::string);
ORT_REGISTER_PRIM_TYPE(int8_t);
ORT_REGISTER_PRIM_TYPE(uint8_t);
ORT_REGISTER_PRIM_TYPE(uint16_t);
ORT_REGISTER_PRIM_TYPE(int16_t);
ORT_REGISTER_PRIM_TYPE(int64_t);
ORT_REGISTER_PRIM_TYPE(double);
ORT_REGISTER_PRIM_TYPE(uint32_t);
ORT_REGISTER_PRIM_TYPE(uint64_t);
ORT_REGISTER_PRIM_TYPE(MLFloat16);
ORT_REGISTER_PRIM_TYPE(BFloat16);

const std::vector<MLDataType>& DataTypeImpl::AllFixedSizeTensorExceptHalfTypes() {
  static std::vector<MLDataType> all_fixed_size_tensor_types =
      {DataTypeImpl::GetTensorType<float>(),
       DataTypeImpl::GetTensorType<double>(),
       DataTypeImpl::GetTensorType<int64_t>(),
       DataTypeImpl::GetTensorType<uint64_t>(),
       DataTypeImpl::GetTensorType<int32_t>(),
       DataTypeImpl::GetTensorType<uint32_t>(),
       DataTypeImpl::GetTensorType<int16_t>(),
       DataTypeImpl::GetTensorType<uint16_t>(),
       DataTypeImpl::GetTensorType<int8_t>(),
       DataTypeImpl::GetTensorType<uint8_t>(),
       DataTypeImpl::GetTensorType<bool>()};

  return all_fixed_size_tensor_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllIEEEFloatTensorExceptHalfTypes() {
  static std::vector<MLDataType> all_IEEE_float_tensor_except_half_types =
      {DataTypeImpl::GetTensorType<float>(),
       DataTypeImpl::GetTensorType<double>()};

  return all_IEEE_float_tensor_except_half_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllIEEEFloatTensorTypes() {
  static std::vector<MLDataType> all_IEEE_float_tensor_types =
      {DataTypeImpl::GetTensorType<float>(),
       DataTypeImpl::GetTensorType<double>(),
       DataTypeImpl::GetTensorType<MLFloat16>()};

  return all_IEEE_float_tensor_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllFixedSizeTensorTypes() {
  static std::vector<MLDataType> all_fixed_size_tensor_types =
      {DataTypeImpl::GetTensorType<float>(),
       DataTypeImpl::GetTensorType<double>(),
       DataTypeImpl::GetTensorType<int64_t>(),
       DataTypeImpl::GetTensorType<uint64_t>(),
       DataTypeImpl::GetTensorType<int32_t>(),
       DataTypeImpl::GetTensorType<uint32_t>(),
       DataTypeImpl::GetTensorType<int16_t>(),
       DataTypeImpl::GetTensorType<uint16_t>(),
       DataTypeImpl::GetTensorType<int8_t>(),
       DataTypeImpl::GetTensorType<uint8_t>(),
       DataTypeImpl::GetTensorType<MLFloat16>(),
       DataTypeImpl::GetTensorType<BFloat16>(),
       DataTypeImpl::GetTensorType<bool>()};

  return all_fixed_size_tensor_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllTensorTypes() {
  static std::vector<MLDataType> all_tensor_types =
      {DataTypeImpl::GetTensorType<float>(),
       DataTypeImpl::GetTensorType<double>(),
       DataTypeImpl::GetTensorType<int64_t>(),
       DataTypeImpl::GetTensorType<uint64_t>(),
       DataTypeImpl::GetTensorType<int32_t>(),
       DataTypeImpl::GetTensorType<uint32_t>(),
       DataTypeImpl::GetTensorType<int16_t>(),
       DataTypeImpl::GetTensorType<uint16_t>(),
       DataTypeImpl::GetTensorType<int8_t>(),
       DataTypeImpl::GetTensorType<uint8_t>(),
       DataTypeImpl::GetTensorType<MLFloat16>(),
       DataTypeImpl::GetTensorType<BFloat16>(),
       DataTypeImpl::GetTensorType<bool>(),
       DataTypeImpl::GetTensorType<std::string>()};

  return all_tensor_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllSequenceTensorTypes() {
  static std::vector<MLDataType> all_sequence_tensor_types =
      {DataTypeImpl::GetSequenceTensorType<float>(),
       DataTypeImpl::GetSequenceTensorType<double>(),
       DataTypeImpl::GetSequenceTensorType<int64_t>(),
       DataTypeImpl::GetSequenceTensorType<uint64_t>(),
       DataTypeImpl::GetSequenceTensorType<int32_t>(),
       DataTypeImpl::GetSequenceTensorType<uint32_t>(),
       DataTypeImpl::GetSequenceTensorType<int16_t>(),
       DataTypeImpl::GetSequenceTensorType<uint16_t>(),
       DataTypeImpl::GetSequenceTensorType<int8_t>(),
       DataTypeImpl::GetSequenceTensorType<uint8_t>(),
       DataTypeImpl::GetSequenceTensorType<MLFloat16>(),
       DataTypeImpl::GetSequenceTensorType<BFloat16>(),
       DataTypeImpl::GetSequenceTensorType<bool>(),
       DataTypeImpl::GetSequenceTensorType<std::string>()};

  return all_sequence_tensor_types;
}

const std::vector<MLDataType>& DataTypeImpl::AllNumericTensorTypes() {
  static std::vector<MLDataType> all_numeric_size_tensor_types =
      {DataTypeImpl::GetTensorType<float>(),
       DataTypeImpl::GetTensorType<double>(),
       DataTypeImpl::GetTensorType<int64_t>(),
       DataTypeImpl::GetTensorType<uint64_t>(),
       DataTypeImpl::GetTensorType<int32_t>(),
       DataTypeImpl::GetTensorType<uint32_t>(),
       DataTypeImpl::GetTensorType<int16_t>(),
       DataTypeImpl::GetTensorType<uint16_t>(),
       DataTypeImpl::GetTensorType<int8_t>(),
       DataTypeImpl::GetTensorType<uint8_t>(),
       DataTypeImpl::GetTensorType<MLFloat16>(),
       DataTypeImpl::GetTensorType<BFloat16>()};

  return all_numeric_size_tensor_types;
}

// helper to stream. expected to only be used for error output, so any typeid lookup
// cost should be fine. alternative would be to add a static string field to DataTypeImpl
// that we set in the register macro to the type name, and output that instead.
std::ostream& operator<<(std::ostream& out, const DataTypeImpl* data_type) {
  if (data_type == nullptr)
    return out << "(null)";

  return out << typeid(*data_type).name();
}

}  // namespace onnxruntime
