// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tensorprotoutils.h"

#include <memory>
#include <algorithm>
#include <limits>
#include <sstream>
#include "onnx-ml.pb.h"
#include "onnxruntime_cxx_api.h"

namespace onnxruntime {


//From core common
inline void MakeStringInternal(std::ostringstream& /*ss*/) noexcept {
}

template <typename T>
inline void MakeStringInternal(std::ostringstream& ss, const T& t) noexcept {
  ss << t;
}

template <typename T, typename... Args>
inline void MakeStringInternal(std::ostringstream& ss, const T& t, const Args&... args) noexcept {
  ::onnxruntime::MakeStringInternal(ss, t);
  ::onnxruntime::MakeStringInternal(ss, args...);
}

template <typename... Args>
std::string MakeString(const Args&... args) {
  std::ostringstream ss;
  ::onnxruntime::MakeStringInternal(ss, args...);
  return std::string(ss.str());
}

// Specializations for already-a-string types.
template <>
inline std::string MakeString(const std::string& str) {
  return str;
}
inline std::string MakeString(const char* p_str) {
  return p_str;
}



namespace server {
constexpr bool IsLittleEndianOrder() noexcept {
#if defined(_WIN32)
  return true;
#elif defined(__GNUC__) || defined(__clang__)
  return __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;
#else
#error server::IsLittleEndianOrder() is not implemented in this environment.
#endif
}

std::vector<int64_t> GetTensorShapeFromTensorProto(const onnx::TensorProto& tensor_proto) {
  const auto& dims = tensor_proto.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    tensor_shape_vec[i] = dims[i];
  }

  return tensor_shape_vec;
}


template <size_t alignment>
static bool CalcMemSizeForArrayWithAlignment(size_t nmemb, size_t size, size_t* out) noexcept {
  static constexpr size_t max_allowed = (static_cast<size_t>(1) << (static_cast<size_t>(std::numeric_limits<size_t>::digits >> 1))) - alignment;
  static constexpr size_t max_size = std::numeric_limits<size_t>::max() - alignment;
  static constexpr size_t alignment_mask = alignment - 1;
  //Indeed, we only need to check if max_size / nmemb < size
  //max_allowed is for avoiding unnecessary DIV.
  if (nmemb >= max_allowed && max_size / nmemb < size) {
    return false;
  }
  if (size >= max_allowed &&
      nmemb > 0 && max_size / nmemb < size) {
    return false;
  }
  if (alignment == 0)
    *out = size * nmemb;
  else
    *out = (size * nmemb + alignment_mask) & ~static_cast<size_t>(alignment_mask);
  return true;
}

static bool CalcMemSizeForArray(size_t nmemb, size_t size, size_t* out) noexcept {
  return CalcMemSizeForArrayWithAlignment<0>(nmemb, size, out);
}

// This function doesn't support string tensors
template <typename T>
static void UnpackTensorWithRawData(const void* raw_data, size_t raw_data_length, size_t expected_size,
                                    /*out*/ T* p_data) {
  // allow this low level routine to be somewhat unsafe. assuming it's thoroughly tested and valid
  {
    size_t expected_size_in_bytes;
    if (!CalcMemSizeForArray(expected_size, sizeof(T), &expected_size_in_bytes)) {
      throw Ort::Exception("size overflow", OrtErrorCode::ORT_FAIL);
    }
    if (raw_data_length != expected_size_in_bytes)
      throw Ort::Exception(MakeString("UnpackTensor: the pre-allocated size does not match the raw data size, expected ",
                                      expected_size_in_bytes, ", got ", raw_data_length),
                           OrtErrorCode::ORT_FAIL);
    if constexpr (IsLittleEndianOrder()) {
      memcpy(p_data, raw_data, raw_data_length);
    } else {
      const size_t type_size = sizeof(T);
      const char* buff = reinterpret_cast<const char*>(raw_data);
      for (size_t i = 0; i < raw_data_length; i += type_size, buff += type_size) {
        T result;
        const char* temp_bytes = reinterpret_cast<char*>(&result);
        for (size_t j = 0; j < type_size; ++j) {
          memcpy((void*)&temp_bytes[j], (void*)&buff[type_size - 1 - i], 1);
        }
        p_data[i] = result;
      }
    }
  }
}

// This macro doesn't work for Float16/bool/string tensors
#define DEFINE_UNPACK_TENSOR(T, Type, field_name, field_size)                                                \
  template <>                                                                                                \
  void UnpackTensor(const onnx::TensorProto& tensor, const void* raw_data, size_t raw_data_len,              \
                    /*out*/ T* p_data, int64_t expected_size) {                                              \
    if (nullptr == p_data) {                                                                                 \
      const size_t size = raw_data != nullptr ? raw_data_len : tensor.field_size();                          \
      if (size == 0) return;                                                                                 \
      throw Ort::Exception("", OrtErrorCode::ORT_INVALID_ARGUMENT);                                          \
    }                                                                                                        \
    if (nullptr == p_data || Type != tensor.data_type()) {                                                   \
      throw Ort::Exception("", OrtErrorCode::ORT_INVALID_ARGUMENT);                                          \
    }                                                                                                        \
    if (raw_data != nullptr) {                                                                               \
      UnpackTensorWithRawData(raw_data, raw_data_len, expected_size, p_data);                                \
      return;                                                                                                \
    }                                                                                                        \
    if (tensor.field_size() != expected_size)                                                                \
      throw Ort::Exception(MakeString("corrupted protobuf data: tensor shape size(", expected_size,          \
                                      ") does not match the data size(", tensor.field_size(), ") in proto"), \
                           OrtErrorCode::ORT_FAIL);                                                          \
    auto& data = tensor.field_name();                                                                        \
    for (auto data_iter = data.cbegin(); data_iter != data.cend(); ++data_iter)                              \
      *p_data++ = *reinterpret_cast<const T*>(data_iter);                                                    \
    return;                                                                                                  \
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
                  /*out*/ std::string* p_data, int64_t expected_size) {
  if (nullptr == p_data) {
    if (tensor.string_data_size() == 0) return;
    throw Ort::Exception("", OrtErrorCode::ORT_INVALID_ARGUMENT);
  }
  if (onnx::TensorProto_DataType_STRING != tensor.data_type()) {
    throw Ort::Exception("", OrtErrorCode::ORT_INVALID_ARGUMENT);
  }

  if (tensor.string_data_size() != expected_size)
    throw Ort::Exception(
        "UnpackTensor: the pre-allocate size does not match the size in proto", OrtErrorCode::ORT_FAIL);

  auto& string_data = tensor.string_data();
  for (const auto& iter : string_data) {
    *p_data++ = iter;
  }

  return;
}
template <>
void UnpackTensor(const onnx::TensorProto& tensor, const void* raw_data, size_t raw_data_len,
                  /*out*/ bool* p_data, int64_t expected_size) {
  if (nullptr == p_data) {
    const size_t size = raw_data != nullptr ? raw_data_len : tensor.int32_data_size();
    if (size == 0) return;
    throw Ort::Exception("", OrtErrorCode::ORT_INVALID_ARGUMENT);
  }
  if (onnx::TensorProto_DataType_BOOL != tensor.data_type()) {
    throw Ort::Exception("", OrtErrorCode::ORT_INVALID_ARGUMENT);
  }

  if (raw_data != nullptr) {
    return UnpackTensorWithRawData(raw_data, raw_data_len, expected_size, p_data);
  }

  if (tensor.int32_data_size() != expected_size)
    throw Ort::Exception(
        "UnpackTensor: the pre-allocate size does not match the size in proto", OrtErrorCode::ORT_FAIL);
  for (int iter : tensor.int32_data()) {
    *p_data++ = static_cast<bool>(iter);
  }

  return;
}


#define CASE_PROTO_TRACE(X, Y)                                                            \
  case onnx::TensorProto_DataType::TensorProto_DataType_##X:                              \
    if (!CalcMemSizeForArrayWithAlignment<alignment>(size, sizeof(Y), out)) { \
      throw Ort::Exception("Invalid TensorProto", OrtErrorCode::ORT_FAIL);                \
    }                                                                                     \
    break;

template <size_t alignment>
void GetSizeInBytesFromTensorProto(const onnx::TensorProto& tensor_proto, size_t* out) {
  const auto& dims = tensor_proto.dims();
  size_t size = 1;
  for (google::protobuf::int64 dim : dims) {
    if (dim < 0 || static_cast<uint64_t>(dim) >= std::numeric_limits<size_t>::max()) {
      throw Ort::Exception("Invalid TensorProto", OrtErrorCode::ORT_FAIL);
    }
    if (!CalcMemSizeForArray(size, static_cast<size_t>(dim), &size)) {
      throw Ort::Exception("Invalid TensorProto", OrtErrorCode::ORT_FAIL);
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
    CASE_PROTO_TRACE(STRING, std::string);
    default:
      throw Ort::Exception("", OrtErrorCode::ORT_NOT_IMPLEMENTED);
  }
  return;
}

struct UnInitializeParam {
  void* preallocated;
  size_t preallocated_size;
  ONNXTensorElementDataType ele_type;
};

void OrtInitializeBufferForTensor(void* input, size_t input_len,
                                  ONNXTensorElementDataType type) {
  try {
    if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING || input == nullptr) return;
    size_t tensor_size = input_len / sizeof(std::string);
    std::string* ptr = reinterpret_cast<std::string*>(input);
    for (size_t i = 0, n = tensor_size; i < n; ++i) {
      new (ptr + i) std::string();
    }
  } catch (std::exception& ex) {
    throw Ort::Exception(ex.what(), OrtErrorCode::ORT_RUNTIME_EXCEPTION);
  }
  return;
}

#define CASE_PROTO(X, Y)                                                                                         \
  case onnx::TensorProto_DataType::TensorProto_DataType_##X:                                                     \
    ::onnxruntime::server::UnpackTensor<Y>(tensor_proto, raw_data, raw_data_len, (Y*)preallocated, tensor_size); \
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
    default:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
}

ONNXTensorElementDataType GetTensorElementType(const onnx::TensorProto& tensor_proto) {
  return CApiElementTypeFromProtoType(tensor_proto.data_type());
}

void TensorProtoToMLValue(const onnx::TensorProto& tensor_proto, const MemBuffer& m, Ort::Value& value) {
  const OrtMemoryInfo& allocator = m.GetAllocInfo();
  ONNXTensorElementDataType ele_type = server::GetTensorElementType(tensor_proto);
  const void* raw_data = nullptr;
  size_t raw_data_len = 0;
  void* tensor_data;
  {
    if (tensor_proto.data_location() == onnx::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL) {
      throw Ort::Exception("Server doesn't support external data.", OrtErrorCode::ORT_INVALID_ARGUMENT);
    } else if (tensor_proto.has_raw_data()) {
      if (ele_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING)
        throw Ort::Exception("String tensor cannot have raw data.", OrtErrorCode::ORT_FAIL);
      raw_data = tensor_proto.raw_data().data();
      raw_data_len = tensor_proto.raw_data().size();
    }
    {
      void* preallocated = m.GetBuffer();
      size_t preallocated_size = m.GetLen();
      int64_t tensor_size = 1;
      {
        for (auto i : tensor_proto.dims()) {
          if (i < 0) throw Ort::Exception("Tensor can't contain negative dims", OrtErrorCode::ORT_FAIL);
          tensor_size *= i;
        }
      }
      // tensor_size could be zero. see test_slice_start_out_of_bounds\test_data_set_0\output_0.pb
      if (static_cast<uint64_t>(tensor_size) > SIZE_MAX) {
        throw Ort::Exception("Size overflow", OrtErrorCode::ORT_INVALID_ARGUMENT);
      }
      size_t size_to_allocate;
      GetSizeInBytesFromTensorProto<0>(tensor_proto, &size_to_allocate);

      if (preallocated && preallocated_size < size_to_allocate)
        throw Ort::Exception(MakeString(
                                 "The buffer planner is not consistent with tensor buffer size, expected ",
                                 size_to_allocate, ", got ", preallocated_size),
                             OrtErrorCode::ORT_FAIL);
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
        case onnx::TensorProto_DataType::TensorProto_DataType_STRING:
          if (preallocated != nullptr) {
            OrtInitializeBufferForTensor(preallocated, preallocated_size, ele_type);
          }
          ::onnxruntime::server::UnpackTensor<std::string>(tensor_proto, raw_data, raw_data_len,
                                                           (std::string*)preallocated, tensor_size);
          break;
        default: {
          std::ostringstream ostr;
          ostr << "Initialized tensor with unexpected type: " << tensor_proto.data_type();
          throw Ort::Exception(ostr.str(), OrtErrorCode::ORT_INVALID_ARGUMENT);
        }
      }
      tensor_data = preallocated;
    }
  }
  std::vector<int64_t> tensor_shape_vec = GetTensorShapeFromTensorProto(tensor_proto);
  // Note: We permit an empty tensor_shape_vec, and treat it as a scalar (a tensor of size 1).
  value = Ort::Value::CreateTensor(&allocator, tensor_data, m.GetLen(), tensor_shape_vec.data(), tensor_shape_vec.size(), (ONNXTensorElementDataType)tensor_proto.data_type());
  return;
}
template void GetSizeInBytesFromTensorProto<256>(const onnx::TensorProto& tensor_proto,
                                                 size_t* out);
template void GetSizeInBytesFromTensorProto<0>(const onnx::TensorProto& tensor_proto, size_t* out);
}  // namespace server
}  // namespace onnxruntime
