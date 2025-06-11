// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tensorprotoutils.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>

#include "callback.h"
#include "core/common/make_string.h"
#include "core/common/safeint.h"
#include "core/common/status.h"
#include "core/framework/allocator.h"
#include "core/framework/data_types.h"
#include "core/framework/endian.h"
#include "core/framework/endian_utils.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "mem_buffer.h"

struct OrtStatus {
  OrtErrorCode code;
  char msg[1];  // a null-terminated string
};

namespace onnxruntime {
namespace test {

std::vector<int64_t> GetTensorShapeFromTensorProto(const onnx::TensorProto& tensor_proto) {
  const auto& dims = tensor_proto.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    tensor_shape_vec[i] = dims[i];
  }

  return tensor_shape_vec;
}

static bool CalcMemSizeForArrayWithAlignment(size_t nmemb, size_t size, size_t alignment, size_t* out) {
  bool ok = true;

  ORT_TRY {
    SafeInt<size_t> alloc_size(size);
    if (alignment == 0) {
      *out = alloc_size * nmemb;
    } else {
      size_t alignment_mask = alignment - 1;
      *out = (alloc_size * nmemb + alignment_mask) & ~static_cast<size_t>(alignment_mask);
    }
  }
  ORT_CATCH(const OnnxRuntimeException&) {
    // overflow in calculating the size thrown by SafeInt.
    ok = false;
  }

  return ok;
}

// This function doesn't support string tensors
template <typename T>
static void UnpackTensorWithRawData(const void* raw_data, size_t raw_data_length, size_t expected_size,
                                    /*out*/ T* p_data) {
  size_t expected_size_in_bytes;
  if (!CalcMemSizeForArrayWithAlignment(expected_size, sizeof(T), 0, &expected_size_in_bytes)) {
    ORT_CXX_API_THROW("size overflow", OrtErrorCode::ORT_FAIL);
  }
  if (raw_data_length != expected_size_in_bytes)
    ORT_CXX_API_THROW(MakeString("UnpackTensor: the pre-allocated size does not match the raw data size, expected ",
                                 expected_size_in_bytes, ", got ", raw_data_length),
                      OrtErrorCode::ORT_FAIL);

  /* Convert Endianness */
  if constexpr (endian::native != endian::little && sizeof(T) > 1) {
    utils::SwapByteOrderCopy(sizeof(T), gsl::make_span(reinterpret_cast<const unsigned char*>(raw_data), raw_data_length),
                             gsl::make_span(reinterpret_cast<unsigned char*>(p_data), raw_data_length));
  } else {
    memcpy(p_data, raw_data, raw_data_length);
  }
}

template <>
void UnpackTensorWithRawData<Int4x2>(const void* raw_data, size_t raw_data_len, size_t expected_num_elements,
                                     /*out*/ Int4x2* p_data) {
  static_assert(std::is_trivially_copyable<Int4x2>::value, "T must be trivially copyable");

  if (p_data == nullptr) {
    ORT_CXX_API_THROW("nullptr == p_data", OrtErrorCode::ORT_FAIL);
  }

  size_t num_packed_pairs = (expected_num_elements + 1) / 2;

  if (num_packed_pairs != raw_data_len) {
    ORT_CXX_API_THROW("Unexpected number of packed int4 pairs", OrtErrorCode::ORT_FAIL);
  }

  gsl::span<const Int4x2> src_span = gsl::make_span(reinterpret_cast<const Int4x2*>(raw_data), num_packed_pairs);
  gsl::span<Int4x2> dst_span = gsl::make_span(p_data, num_packed_pairs);

  std::memcpy(dst_span.data(), src_span.data(), num_packed_pairs);
}

template <>
void UnpackTensorWithRawData<UInt4x2>(const void* raw_data, size_t raw_data_len, size_t expected_num_elements,
                                      /*out*/ UInt4x2* p_data) {
  static_assert(std::is_trivially_copyable<UInt4x2>::value, "T must be trivially copyable");

  if (p_data == nullptr) {
    ORT_CXX_API_THROW("nullptr == p_data", OrtErrorCode::ORT_FAIL);
  }

  size_t num_packed_pairs = (expected_num_elements + 1) / 2;

  if (num_packed_pairs != raw_data_len) {
    ORT_CXX_API_THROW("Unexpected number of packed int4 pairs", OrtErrorCode::ORT_FAIL);
  }

  gsl::span<const UInt4x2> src_span = gsl::make_span(reinterpret_cast<const UInt4x2*>(raw_data), num_packed_pairs);
  gsl::span<UInt4x2> dst_span = gsl::make_span(p_data, num_packed_pairs);

  std::memcpy(dst_span.data(), src_span.data(), num_packed_pairs);
}

// This macro doesn't work for Float16/bool/string tensors
#define DEFINE_UNPACK_TENSOR(T, Type, field_name, field_size)                                             \
  template <>                                                                                             \
  void UnpackTensor(const onnx::TensorProto& tensor, const void* raw_data, size_t raw_data_len,           \
                    /*out*/ T* p_data, size_t expected_size) {                                            \
    if (nullptr == p_data) {                                                                              \
      const size_t size = raw_data != nullptr ? raw_data_len : tensor.field_size();                       \
      if (size == 0) return;                                                                              \
      ORT_CXX_API_THROW("", OrtErrorCode::ORT_INVALID_ARGUMENT);                                          \
    }                                                                                                     \
    if (nullptr == p_data || Type != tensor.data_type()) {                                                \
      ORT_CXX_API_THROW("", OrtErrorCode::ORT_INVALID_ARGUMENT);                                          \
    }                                                                                                     \
    if (raw_data != nullptr) {                                                                            \
      UnpackTensorWithRawData(raw_data, raw_data_len, expected_size, p_data);                             \
      return;                                                                                             \
    }                                                                                                     \
    if (static_cast<size_t>(tensor.field_size()) != expected_size)                                        \
      ORT_CXX_API_THROW(MakeString("corrupted protobuf data: tensor shape size(", expected_size,          \
                                   ") does not match the data size(", tensor.field_size(), ") in proto"), \
                        OrtErrorCode::ORT_FAIL);                                                          \
    auto& data = tensor.field_name();                                                                     \
    for (auto data_iter = data.cbegin(); data_iter != data.cend(); ++data_iter)                           \
      *p_data++ = onnxruntime::narrow<T>(*data_iter);                                                     \
    return;                                                                                               \
  }

// TODO: complex64 complex128
DEFINE_UNPACK_TENSOR(float, onnx::TensorProto_DataType_FLOAT, float_data, float_data_size)
DEFINE_UNPACK_TENSOR(double, onnx::TensorProto_DataType_DOUBLE, double_data, double_data_size);
DEFINE_UNPACK_TENSOR(uint8_t, onnx::TensorProto_DataType_UINT8, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR(int8_t, onnx::TensorProto_DataType_INT8, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR(int16_t, onnx::TensorProto_DataType_INT16, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR(uint16_t, onnx::TensorProto_DataType_UINT16, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR(int32_t, onnx::TensorProto_DataType_INT32, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR(int64_t, onnx::TensorProto_DataType_INT64, int64_data, int64_data_size)
DEFINE_UNPACK_TENSOR(uint64_t, onnx::TensorProto_DataType_UINT64, uint64_data, uint64_data_size)
DEFINE_UNPACK_TENSOR(uint32_t, onnx::TensorProto_DataType_UINT32, uint64_data, uint64_data_size)

// doesn't support raw data
template <>
void UnpackTensor(const onnx::TensorProto& tensor, const void* /*raw_data*/, size_t /*raw_data_len*/,
                  /*out*/ std::string* p_data, size_t expected_size) {
  if (nullptr == p_data) {
    if (tensor.string_data_size() == 0) return;
    ORT_CXX_API_THROW("", OrtErrorCode::ORT_INVALID_ARGUMENT);
  }
  if (onnx::TensorProto_DataType_STRING != tensor.data_type()) {
    ORT_CXX_API_THROW("", OrtErrorCode::ORT_INVALID_ARGUMENT);
  }

  if (static_cast<size_t>(tensor.string_data_size()) != expected_size)
    ORT_CXX_API_THROW(
        "UnpackTensor: the pre-allocate size does not match the size in proto", OrtErrorCode::ORT_FAIL);

  auto& string_data = tensor.string_data();
  for (const auto& iter : string_data) {
    *p_data++ = iter;
  }

  return;
}
template <>
void UnpackTensor(const onnx::TensorProto& tensor, const void* raw_data, size_t raw_data_len,
                  /*out*/ bool* p_data, size_t expected_size) {
  if (nullptr == p_data) {
    const size_t size = raw_data != nullptr ? raw_data_len : tensor.int32_data_size();
    if (size == 0) return;
    ORT_CXX_API_THROW("", OrtErrorCode::ORT_INVALID_ARGUMENT);
  }
  if (onnx::TensorProto_DataType_BOOL != tensor.data_type()) {
    ORT_CXX_API_THROW("", OrtErrorCode::ORT_INVALID_ARGUMENT);
  }

  if (raw_data != nullptr) {
    return UnpackTensorWithRawData(raw_data, raw_data_len, expected_size, p_data);
  }

  if (static_cast<size_t>(tensor.int32_data_size()) != expected_size)
    ORT_CXX_API_THROW(
        "UnpackTensor: the pre-allocate size does not match the size in proto", OrtErrorCode::ORT_FAIL);
  for (int iter : tensor.int32_data()) {
    *p_data++ = static_cast<bool>(iter);
  }

  return;
}
template <>
void UnpackTensor(const onnx::TensorProto& tensor, const void* raw_data, size_t raw_data_len,
                  /*out*/ MLFloat16* p_data, size_t expected_size) {
  if (nullptr == p_data) {
    const size_t size = raw_data != nullptr ? raw_data_len : tensor.int32_data_size();
    if (size == 0) return;
    ORT_CXX_API_THROW("", OrtErrorCode::ORT_INVALID_ARGUMENT);
  }
  if (onnx::TensorProto_DataType_FLOAT16 != tensor.data_type()) {
    ORT_CXX_API_THROW("", OrtErrorCode::ORT_INVALID_ARGUMENT);
  }

  if (raw_data != nullptr) {
    return UnpackTensorWithRawData(raw_data, raw_data_len, expected_size, p_data);
  }

  if (static_cast<size_t>(tensor.int32_data_size()) != expected_size)
    ORT_CXX_API_THROW(
        "UnpackTensor: the pre-allocate size does not match the size in proto", OrtErrorCode::ORT_FAIL);

  constexpr int max_value = std::numeric_limits<uint16_t>::max();
  for (int i = 0; i < static_cast<int>(expected_size); i++) {
    int v = tensor.int32_data()[i];
    if (v < 0 || v > max_value) {
      ORT_CXX_API_THROW(
          "data overflow", OrtErrorCode::ORT_FAIL);
    }
    p_data[i] = MLFloat16::FromBits(static_cast<uint16_t>(v));
  }

  return;
}

template <>
void UnpackTensor(const onnx::TensorProto& tensor, const void* raw_data, size_t raw_data_len,
                  /*out*/ BFloat16* p_data, size_t expected_size) {
  if (nullptr == p_data) {
    const size_t size = raw_data != nullptr ? raw_data_len : tensor.int32_data_size();
    if (size == 0)
      return;

    ORT_CXX_API_THROW("", OrtErrorCode::ORT_INVALID_ARGUMENT);
  }
  if (onnx::TensorProto_DataType_BFLOAT16 != tensor.data_type()) {
    ORT_CXX_API_THROW("", OrtErrorCode::ORT_INVALID_ARGUMENT);
  }

  if (raw_data != nullptr) {
    return UnpackTensorWithRawData(raw_data, raw_data_len, expected_size, p_data);
  }

  if (static_cast<size_t>(tensor.int32_data_size()) != expected_size)
    ORT_CXX_API_THROW(
        "UnpackTensor: the pre-allocate size does not match the size in proto", OrtErrorCode::ORT_FAIL);

  constexpr int max_value = std::numeric_limits<uint16_t>::max();
  for (int i = 0; i < static_cast<int>(expected_size); i++) {
    int v = tensor.int32_data()[i];
    if (v < 0 || v > max_value) {
      ORT_CXX_API_THROW(
          "data overflow", OrtErrorCode::ORT_FAIL);
    }
    p_data[i] = BFloat16(static_cast<uint16_t>(v));
  }

  return;
}

#define DEFINE_UNPACK_TENSOR_FLOAT8(TYPE, ONNX_TYPE)                                                       \
  template <>                                                                                              \
  void UnpackTensor(const onnx::TensorProto& tensor, const void* raw_data, size_t raw_data_len,            \
                    /*out*/ TYPE* p_data, size_t expected_size) {                                          \
    if (nullptr == p_data) {                                                                               \
      const size_t size = raw_data != nullptr ? raw_data_len : tensor.int32_data_size();                   \
      if (size == 0)                                                                                       \
        return;                                                                                            \
      ORT_CXX_API_THROW("", OrtErrorCode::ORT_INVALID_ARGUMENT);                                           \
    }                                                                                                      \
    if (onnx::ONNX_TYPE != tensor.data_type()) {                                                           \
      ORT_CXX_API_THROW("", OrtErrorCode::ORT_INVALID_ARGUMENT);                                           \
    }                                                                                                      \
    if (raw_data != nullptr) {                                                                             \
      return UnpackTensorWithRawData(raw_data, raw_data_len, expected_size, p_data);                       \
    }                                                                                                      \
    if (static_cast<size_t>(tensor.int32_data_size()) != expected_size)                                    \
      ORT_CXX_API_THROW(                                                                                   \
          "UnpackTensor: the pre-allocate size does not match the size in proto", OrtErrorCode::ORT_FAIL); \
    constexpr int max_value = std::numeric_limits<uint8_t>::max();                                         \
    for (int i = 0; i < static_cast<int>(expected_size); i++) {                                            \
      int v = tensor.int32_data()[i];                                                                      \
      if (v < 0 || v > max_value) {                                                                        \
        ORT_CXX_API_THROW(                                                                                 \
            "data overflow", OrtErrorCode::ORT_FAIL);                                                      \
      }                                                                                                    \
      p_data[i] = TYPE(static_cast<uint8_t>(v), TYPE::FromBits());                                         \
    }                                                                                                      \
    return;                                                                                                \
  }

#if !defined(DISABLE_FLOAT8_TYPES)
DEFINE_UNPACK_TENSOR_FLOAT8(Float8E4M3FN, TensorProto_DataType_FLOAT8E4M3FN)
DEFINE_UNPACK_TENSOR_FLOAT8(Float8E4M3FNUZ, TensorProto_DataType_FLOAT8E4M3FNUZ)
DEFINE_UNPACK_TENSOR_FLOAT8(Float8E5M2, TensorProto_DataType_FLOAT8E5M2)
DEFINE_UNPACK_TENSOR_FLOAT8(Float8E5M2FNUZ, TensorProto_DataType_FLOAT8E5M2FNUZ)
#endif

#define DEFINE_UNPACK_TENSOR_INT4(INT4_TYPE, ONNX_TYPE)                                                   \
  template <>                                                                                             \
  void UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len, \
                    /*out*/ INT4_TYPE* p_data, size_t expected_num_elems) {                               \
    if (nullptr == p_data) {                                                                              \
      const size_t size = raw_data != nullptr ? raw_data_len : tensor.int32_data_size();                  \
      if (size == 0) {                                                                                    \
        return;                                                                                           \
      }                                                                                                   \
      ORT_CXX_API_THROW("p_data == nullptr, but size != 0", OrtErrorCode::ORT_INVALID_ARGUMENT);          \
    }                                                                                                     \
    if (ONNX_NAMESPACE::ONNX_TYPE != tensor.data_type()) {                                                \
      ORT_CXX_API_THROW("TensorProto data type is not INT4", OrtErrorCode::ORT_INVALID_ARGUMENT);         \
    }                                                                                                     \
                                                                                                          \
    size_t expected_int4_pairs = (expected_num_elems + 1) / 2;                                            \
                                                                                                          \
    if (raw_data != nullptr) {                                                                            \
      UnpackTensorWithRawData(raw_data, raw_data_len, expected_num_elems, p_data);                        \
      return;                                                                                             \
    }                                                                                                     \
                                                                                                          \
    if (static_cast<size_t>(tensor.int32_data_size()) != expected_int4_pairs) {                           \
      ORT_CXX_API_THROW("UnpackTensor: the pre-allocated size does not match the size in proto",          \
                        OrtErrorCode::ORT_FAIL);                                                          \
    }                                                                                                     \
                                                                                                          \
    for (int i = 0; i < static_cast<int>(tensor.int32_data_size()); i++) {                                \
      p_data[i] = INT4_TYPE(static_cast<std::byte>(tensor.int32_data()[i]));                              \
    }                                                                                                     \
  }

DEFINE_UNPACK_TENSOR_INT4(Int4x2, TensorProto_DataType_INT4)
DEFINE_UNPACK_TENSOR_INT4(UInt4x2, TensorProto_DataType_UINT4)

#define CASE_PROTO_TRACE(X, Y)                                                \
  case onnx::TensorProto_DataType::TensorProto_DataType_##X:                  \
    if (!CalcMemSizeForArrayWithAlignment(size, sizeof(Y), alignment, out)) { \
      ORT_CXX_API_THROW("Invalid TensorProto", OrtErrorCode::ORT_FAIL);       \
    }                                                                         \
    break;

#define CASE_PROTO_TRACE_INT4(X)                                                \
  case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_##X:          \
    if (!CalcMemSizeForArrayWithAlignment((size + 1) / 2, 1, alignment, out)) { \
      ORT_CXX_API_THROW("Invalid TensorProto", OrtErrorCode::ORT_FAIL);         \
    }                                                                           \
    break;

template <size_t alignment>
Status GetSizeInBytesFromTensorProto(const ONNX_NAMESPACE::TensorProto& tensor_proto, size_t* out) {
  const auto& dims = tensor_proto.dims();
  size_t size = 1;
  for (google::protobuf::int64 dim : dims) {
    if (dim < 0 || static_cast<uint64_t>(dim) >= std::numeric_limits<size_t>::max()) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Invalid TensorProto");
    }
    if (!CalcMemSizeForArrayWithAlignment(size, static_cast<size_t>(dim), 0, &size)) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Invalid TensorProto");
    }
  }
  switch (tensor_proto.data_type()) {
    CASE_PROTO_TRACE(FLOAT, float);
    CASE_PROTO_TRACE(DOUBLE, double);
    CASE_PROTO_TRACE(BOOL, bool);
    CASE_PROTO_TRACE(INT8, int8_t);
    CASE_PROTO_TRACE(INT16, int16_t);
    CASE_PROTO_TRACE(INT32, int32_t);
    CASE_PROTO_TRACE(INT64, int64_t);
    CASE_PROTO_TRACE(UINT8, uint8_t);
    CASE_PROTO_TRACE(UINT16, uint16_t);
    CASE_PROTO_TRACE(UINT32, uint32_t);
    CASE_PROTO_TRACE(UINT64, uint64_t);
    CASE_PROTO_TRACE(FLOAT16, MLFloat16);
    CASE_PROTO_TRACE(BFLOAT16, BFloat16);
#if !defined(DISABLE_FLOAT8_TYPES)
    CASE_PROTO_TRACE(FLOAT8E4M3FN, Float8E4M3FN);
    CASE_PROTO_TRACE(FLOAT8E4M3FNUZ, Float8E4M3FNUZ);
    CASE_PROTO_TRACE(FLOAT8E5M2, Float8E5M2);
    CASE_PROTO_TRACE(FLOAT8E5M2FNUZ, Float8E5M2FNUZ);
#endif
    CASE_PROTO_TRACE(STRING, std::string);
    CASE_PROTO_TRACE_INT4(UINT4);
    CASE_PROTO_TRACE_INT4(INT4);
    default:
      return Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED);
  }
  return Status::OK();
}

struct UnInitializeParam {
  void* preallocated;
  size_t preallocated_size;
  ONNXTensorElementDataType ele_type;
};

OrtStatus* OrtInitializeBufferForTensor(void* input, size_t input_len,
                                        ONNXTensorElementDataType type) {
  OrtStatus* status = nullptr;
  ORT_TRY {
    if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING || input == nullptr) return nullptr;
    size_t tensor_size = input_len / sizeof(std::string);
    std::string* ptr = reinterpret_cast<std::string*>(input);
    for (size_t i = 0, n = tensor_size; i < n; ++i) {
      new (ptr + i) std::string();
    }
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = Ort::GetApi().CreateStatus(ORT_RUNTIME_EXCEPTION, ex.what());
    });
  }

  return status;
}

ORT_API(void, OrtUninitializeBuffer, _In_opt_ void* input, size_t input_len, enum ONNXTensorElementDataType type);
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 26409)
#endif
static void UnInitTensor(void* param) noexcept {
  UnInitializeParam* p = reinterpret_cast<UnInitializeParam*>(param);
  OrtUninitializeBuffer(p->preallocated, p->preallocated_size, p->ele_type);
  delete p;
}

ORT_API(void, OrtUninitializeBuffer, _In_opt_ void* input, size_t input_len, enum ONNXTensorElementDataType type) {
  if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING || input == nullptr) return;
  size_t tensor_size = input_len / sizeof(std::string);
  std::string* ptr = reinterpret_cast<std::string*>(input);
  using std::string;
  for (size_t i = 0, n = tensor_size; i < n; ++i) {
    ptr[i].~string();
  }
}

#define CASE_PROTO(X, Y)                                                                      \
  case onnx::TensorProto_DataType::TensorProto_DataType_##X:                                  \
    ::onnxruntime::test::UnpackTensor<Y>(tensor_proto, raw_data, raw_data_len,                \
                                         (Y*)preallocated, static_cast<size_t>(tensor_size)); \
    break;

#define CASE_TYPE(X)                   \
  case onnx::TensorProto_DataType_##X: \
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_##X;

ONNXTensorElementDataType CApiElementTypeFromProtoType(int type) {
  switch (type) {
    CASE_TYPE(FLOAT)
    CASE_TYPE(UINT8)
    CASE_TYPE(INT8)
    CASE_TYPE(UINT16)
    CASE_TYPE(INT16)
    CASE_TYPE(INT32)
    CASE_TYPE(INT64)
    CASE_TYPE(STRING)
    CASE_TYPE(BOOL)
    CASE_TYPE(FLOAT16)
    CASE_TYPE(DOUBLE)
    CASE_TYPE(UINT32)
    CASE_TYPE(UINT64)
    CASE_TYPE(COMPLEX64)
    CASE_TYPE(COMPLEX128)
    CASE_TYPE(BFLOAT16)
    CASE_TYPE(FLOAT8E4M3FN)
    CASE_TYPE(FLOAT8E4M3FNUZ)
    CASE_TYPE(FLOAT8E5M2)
    CASE_TYPE(FLOAT8E5M2FNUZ)
    CASE_TYPE(UINT4)
    CASE_TYPE(INT4)
    default:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
}

ONNXTensorElementDataType GetTensorElementType(const onnx::TensorProto& tensor_proto) {
  return CApiElementTypeFromProtoType(tensor_proto.data_type());
}

Status TensorProtoToMLValue(const onnx::TensorProto& tensor_proto, const MemBuffer& m, Ort::Value& value,
                            OrtCallback& deleter) {
  const OrtMemoryInfo& allocator = m.GetAllocInfo();
  ONNXTensorElementDataType ele_type = test::GetTensorElementType(tensor_proto);
  const void* raw_data = nullptr;
  size_t raw_data_len = 0;
  void* tensor_data;
  {
    if (tensor_proto.data_location() == onnx::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Server doesn't support external data.");
    } else if (tensor_proto.has_raw_data()) {
      if (ele_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING)
        return Status(common::ONNXRUNTIME, common::FAIL, "String tensor cannot have raw data.");
      raw_data = tensor_proto.raw_data().data();
      raw_data_len = tensor_proto.raw_data().size();
    }
    {
      void* preallocated = m.GetBuffer();
      size_t preallocated_size = m.GetLen();
      int64_t tensor_size = 1;
      {
        for (auto i : tensor_proto.dims()) {
          if (i < 0) return Status(common::ONNXRUNTIME, common::FAIL, "Tensor can't contain negative dims");
          tensor_size *= i;
        }
      }
      // tensor_size could be zero. see test_slice_start_out_of_bounds\test_data_set_0\output_0.pb
      if (static_cast<uint64_t>(tensor_size) > SIZE_MAX) {
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Size overflow");
      }
      size_t size_to_allocate = 0;
      ORT_RETURN_IF_ERROR(GetSizeInBytesFromTensorProto<0>(tensor_proto, &size_to_allocate));

      if (preallocated && preallocated_size < size_to_allocate)
        return Status(common::ONNXRUNTIME, common::FAIL, MakeString("The buffer planner is not consistent with tensor buffer size, expected ", size_to_allocate, ", got ", preallocated_size));
      switch (tensor_proto.data_type()) {
        CASE_PROTO(FLOAT, float);
        CASE_PROTO(DOUBLE, double);
        CASE_PROTO(BOOL, bool);
        CASE_PROTO(INT8, int8_t);
        CASE_PROTO(INT16, int16_t);
        CASE_PROTO(INT32, int32_t);
        CASE_PROTO(INT64, int64_t);
        CASE_PROTO(UINT8, uint8_t);
        CASE_PROTO(UINT16, uint16_t);
        CASE_PROTO(UINT32, uint32_t);
        CASE_PROTO(UINT64, uint64_t);
        CASE_PROTO(FLOAT16, MLFloat16);
        CASE_PROTO(BFLOAT16, BFloat16);
#if !defined(DISABLE_FLOAT8_TYPES)
        CASE_PROTO(FLOAT8E4M3FN, Float8E4M3FN);
        CASE_PROTO(FLOAT8E4M3FNUZ, Float8E4M3FNUZ);
        CASE_PROTO(FLOAT8E5M2, Float8E5M2);
        CASE_PROTO(FLOAT8E5M2FNUZ, Float8E5M2FNUZ);
#endif
        CASE_PROTO(INT4, Int4x2);
        CASE_PROTO(UINT4, UInt4x2);
        case onnx::TensorProto_DataType::TensorProto_DataType_STRING:
          if (preallocated != nullptr) {
            OrtStatus* status = OrtInitializeBufferForTensor(preallocated, preallocated_size, ele_type);
            if (status != nullptr) {
              Ort::GetApi().ReleaseStatus(status);
              return Status(common::ONNXRUNTIME, common::FAIL, "initialize preallocated buffer failed");
            }
            deleter.f = UnInitTensor;
            deleter.param = new UnInitializeParam{preallocated, preallocated_size, ele_type};
          }
          ::onnxruntime::test::UnpackTensor<std::string>(tensor_proto, raw_data, raw_data_len,
                                                         (std::string*)preallocated, static_cast<size_t>(tensor_size));
          break;
        default: {
          std::ostringstream ostr;
          ostr << "Initialized tensor with unexpected type: " << tensor_proto.data_type();
          return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, ostr.str());
        }
      }
      tensor_data = preallocated;
    }
  }
  std::vector<int64_t> tensor_shape_vec = GetTensorShapeFromTensorProto(tensor_proto);
  // Note: We permit an empty tensor_shape_vec, and treat it as a scalar (a tensor of size 1).
  value = Ort::Value::CreateTensor(&allocator, tensor_data, m.GetLen(), tensor_shape_vec.data(), tensor_shape_vec.size(), (ONNXTensorElementDataType)tensor_proto.data_type());
  return Status::OK();
}

Status MLValueToTensorProto(Ort::Value& value, onnx::TensorProto& tensor_proto) {
  Ort::TensorTypeAndShapeInfo tensor_info = value.GetTensorTypeAndShapeInfo();
  size_t out_dims = tensor_info.GetDimensionsCount();
  std::vector<int64_t> out_shape = tensor_info.GetShape();
  for (size_t j = 0; j != out_dims; ++j) {
    if (out_shape[j] < 0) return Status(common::ONNXRUNTIME, common::FAIL, "Tensor can't contain negative dims");
    tensor_proto.add_dims(out_shape[j]);
  }
  size_t tensor_size = tensor_info.GetElementCount();
  if (static_cast<uint64_t>(tensor_size) > SIZE_MAX) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Size overflow");
  }

  ONNXTensorElementDataType tensor_elem_data_type = tensor_info.GetElementType();
  int tensor_elem_bytes;
  int tensor_proto_dtype;
  switch (tensor_elem_data_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      tensor_proto_dtype = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT;
      tensor_elem_bytes = sizeof(float);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      tensor_proto_dtype = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8;
      tensor_elem_bytes = sizeof(uint8_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      tensor_proto_dtype = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8;
      tensor_elem_bytes = sizeof(int8_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      tensor_proto_dtype = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT16;
      tensor_elem_bytes = sizeof(uint16_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      tensor_proto_dtype = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16;
      tensor_elem_bytes = sizeof(int16_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      tensor_proto_dtype = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32;
      tensor_elem_bytes = sizeof(int32_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      tensor_proto_dtype = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64;
      tensor_elem_bytes = sizeof(int64_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      tensor_proto_dtype = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL;
      tensor_elem_bytes = sizeof(bool);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      tensor_proto_dtype = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16;
      tensor_elem_bytes = 2;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      tensor_proto_dtype = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE;
      tensor_elem_bytes = sizeof(double);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      tensor_proto_dtype = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT32;
      tensor_elem_bytes = sizeof(uint32_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      tensor_proto_dtype = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT64;
      tensor_elem_bytes = sizeof(uint64_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      tensor_proto_dtype = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_COMPLEX64;
      tensor_elem_bytes = 8;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      tensor_proto_dtype = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_COMPLEX128;
      tensor_elem_bytes = 16;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      tensor_proto_dtype = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BFLOAT16;
      tensor_elem_bytes = 2;
      break;
#if !defined(DISABLE_FLOAT8_TYPES)
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
      tensor_proto_dtype = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FN;
      tensor_elem_bytes = 1;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
      tensor_proto_dtype = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FNUZ;
      tensor_elem_bytes = 1;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
      tensor_proto_dtype = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT8E5M2;
      tensor_elem_bytes = 1;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
      tensor_proto_dtype = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT8E5M2FNUZ;
      tensor_elem_bytes = 1;
      break;
#endif
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4:
      tensor_proto_dtype = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT4;
      tensor_elem_bytes = 1;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:
      tensor_proto_dtype = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT4;
      tensor_elem_bytes = 1;
      break;
    default: {
      // In this function, we do not support
      // ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING and ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
      std::ostringstream ostr;
      ostr << "Initialized tensor with unexpected type: " << tensor_elem_data_type;
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, ostr.str());
    }
  }
  tensor_proto.set_data_type(tensor_proto_dtype);
  void* output_buffer = value.GetTensorMutableRawData();
  tensor_proto.set_raw_data(
      output_buffer, tensor_size * tensor_elem_bytes);
  return Status::OK();
}

template Status GetSizeInBytesFromTensorProto<kAllocAlignment>(const onnx::TensorProto& tensor_proto, size_t* out);
template Status GetSizeInBytesFromTensorProto<0>(const onnx::TensorProto& tensor_proto, size_t* out);
}  // namespace test
}  // namespace onnxruntime
