// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"

#include <memory>
#include <algorithm>
#include <limits>
#include <gsl/gsl>

#include "core/common/logging/logging.h"
#include "core/graph/onnx_protobuf.h"
#include "core/framework/endian_utils.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "core/framework/ort_value_pattern_planner.h"
#include "core/framework/allocator.h"
#include "core/framework/callback.h"
#include "core/framework/data_types.h"
#include "core/platform/path_lib.h"
#include "core/session/ort_apis.h"
#include "onnx/defs/tensor_proto_util.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;

// Provide template specializations for onnxruntime-specific types.
namespace ONNX_NAMESPACE {
template <>
TensorProto ToTensor<onnxruntime::MLFloat16>(const onnxruntime::MLFloat16& value) {
  TensorProto t;
  t.set_data_type(TensorProto_DataType_FLOAT16);
  t.add_int32_data(value.val);
  return t;
}

template <>
TensorProto ToTensor<onnxruntime::MLFloat16>(const std::vector<onnxruntime::MLFloat16>& values) {
  TensorProto t;
  t.clear_int32_data();
  t.set_data_type(TensorProto_DataType_FLOAT16);
  for (const onnxruntime::MLFloat16& val : values) {
    t.add_int32_data(val.val);
  }
  return t;
}

bool operator==(const ONNX_NAMESPACE::TensorShapeProto_Dimension& l,
                const ONNX_NAMESPACE::TensorShapeProto_Dimension& r) {
  if (l.has_dim_value()) {
    return r.has_dim_value() && l.dim_value() == r.dim_value();
  } else if (l.has_dim_param()) {
    return r.has_dim_param() && l.dim_param() == r.dim_param() && !l.dim_param().empty();
  } else {
    // l is unknown - has neither dim_value nor dim_param
  }

  return false;
}
}  // namespace ONNX_NAMESPACE

namespace {

std::vector<int64_t> GetTensorShapeFromTensorProto(const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  const auto& dims = tensor_proto.dims();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    tensor_shape_vec[i] = dims[i];
  }

  return tensor_shape_vec;
}

// This function doesn't support string tensors
template <typename T>
static Status UnpackTensorWithRawData(const void* raw_data, size_t raw_data_length, size_t expected_size,
                                      /*out*/ T* p_data) {
  size_t expected_size_in_bytes;
  if (!onnxruntime::IAllocator::CalcMemSizeForArray(expected_size, sizeof(T), &expected_size_in_bytes)) {
    return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "size overflow");
  }
  if (raw_data_length != expected_size_in_bytes)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "UnpackTensor: the pre-allocated size does not match the raw data size, expected ",
                           expected_size_in_bytes, ", got ", raw_data_length);

  const char* const raw_data_bytes = reinterpret_cast<const char*>(raw_data);
  ORT_RETURN_IF_ERROR(onnxruntime::utils::ReadLittleEndian(
      gsl::make_span(raw_data_bytes, raw_data_length), gsl::make_span(p_data, expected_size)));
  return Status::OK();
}
}  // namespace

namespace onnxruntime {
namespace utils {

// This macro doesn't work for Float16/bool/string tensors
#define DEFINE_UNPACK_TENSOR(T, Type, field_name, field_size)                                                      \
  template <>                                                                                                      \
  Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len,        \
                      /*out*/ T* p_data, size_t expected_size) {                                                   \
    if (nullptr == p_data) {                                                                                       \
      const size_t size = raw_data != nullptr ? raw_data_len : tensor.field_size();                                \
      if (size == 0) return Status::OK();                                                                          \
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);                                                \
    }                                                                                                              \
    if (nullptr == p_data || Type != tensor.data_type()) {                                                         \
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);                                                \
    }                                                                                                              \
    if (raw_data != nullptr) {                                                                                     \
      return UnpackTensorWithRawData(raw_data, raw_data_len, expected_size, p_data);                               \
    }                                                                                                              \
    if (static_cast<size_t>(tensor.field_size()) != expected_size)                                                 \
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "corrupted protobuf data: tensor shape size(",         \
                             expected_size, ") does not match the data size(", tensor.field_size(), ") in proto"); \
    auto& data = tensor.field_name();                                                                              \
    for (auto data_iter = data.cbegin(); data_iter != data.cend(); ++data_iter)                                    \
      *p_data++ = *reinterpret_cast<const T*>(data_iter);                                                          \
    return Status::OK();                                                                                           \
  }

// TODO: complex64 complex128
DEFINE_UNPACK_TENSOR(float, ONNX_NAMESPACE::TensorProto_DataType_FLOAT, float_data, float_data_size)
DEFINE_UNPACK_TENSOR(double, ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, double_data, double_data_size);
DEFINE_UNPACK_TENSOR(uint8_t, ONNX_NAMESPACE::TensorProto_DataType_UINT8, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR(int8_t, ONNX_NAMESPACE::TensorProto_DataType_INT8, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR(int16_t, ONNX_NAMESPACE::TensorProto_DataType_INT16, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR(uint16_t, ONNX_NAMESPACE::TensorProto_DataType_UINT16, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR(int32_t, ONNX_NAMESPACE::TensorProto_DataType_INT32, int32_data, int32_data_size)
DEFINE_UNPACK_TENSOR(int64_t, ONNX_NAMESPACE::TensorProto_DataType_INT64, int64_data, int64_data_size)
DEFINE_UNPACK_TENSOR(uint64_t, ONNX_NAMESPACE::TensorProto_DataType_UINT64, uint64_data, uint64_data_size)
DEFINE_UNPACK_TENSOR(uint32_t, ONNX_NAMESPACE::TensorProto_DataType_UINT32, uint64_data, uint64_data_size)

// doesn't support raw data
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* /*raw_data*/, size_t /*raw_data_len*/,
                    /*out*/ std::string* p_data, size_t expected_size) {
  if (nullptr == p_data) {
    if (tensor.string_data_size() == 0) return Status::OK();
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }
  if (ONNX_NAMESPACE::TensorProto_DataType_STRING != tensor.data_type()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }

  if (static_cast<size_t>(tensor.string_data_size()) != expected_size)
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "UnpackTensor: the pre-allocate size does not match the size in proto");

  auto& string_data = tensor.string_data();
  for (const auto& iter : string_data) {
    *p_data++ = iter;
  }

  return Status::OK();
}
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len,
                    /*out*/ bool* p_data, size_t expected_size) {
  if (nullptr == p_data) {
    const size_t size = raw_data != nullptr ? raw_data_len : tensor.int32_data_size();
    if (size == 0) return Status::OK();
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }
  if (ONNX_NAMESPACE::TensorProto_DataType_BOOL != tensor.data_type()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }

  if (raw_data != nullptr) {
    return UnpackTensorWithRawData(raw_data, raw_data_len, expected_size, p_data);
  }

  if (static_cast<size_t>(tensor.int32_data_size()) != expected_size)
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "UnpackTensor: the pre-allocate size does not match the size in proto");
  for (int iter : tensor.int32_data()) {
    *p_data++ = static_cast<bool>(iter);
  }

  return Status::OK();
}
template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len,
                    /*out*/ MLFloat16* p_data, size_t expected_size) {
  if (nullptr == p_data) {
    const size_t size = raw_data != nullptr ? raw_data_len : tensor.int32_data_size();
    if (size == 0) return Status::OK();
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }
  if (ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 != tensor.data_type()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }

  if (raw_data != nullptr) {
    return UnpackTensorWithRawData(raw_data, raw_data_len, expected_size, p_data);
  }

  if (static_cast<size_t>(tensor.int32_data_size()) != expected_size)
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "UnpackTensor: the pre-allocate size does not match the size in proto");

  constexpr int max_value = std::numeric_limits<uint16_t>::max();
  for (int i = 0; i < static_cast<int>(expected_size); i++) {
    int v = tensor.int32_data()[i];
    if (v < 0 || v > max_value) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "data overflow");
    }
    p_data[i] = MLFloat16(static_cast<uint16_t>(v));
  }

  return Status::OK();
}

template <>
Status UnpackTensor(const ONNX_NAMESPACE::TensorProto& tensor, const void* raw_data, size_t raw_data_len,
                    /*out*/ BFloat16* p_data, size_t expected_size) {
  if (nullptr == p_data) {
    const size_t size = raw_data != nullptr ? raw_data_len : tensor.int32_data_size();
    if (size == 0)
      return Status::OK();

    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }
  if (ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16 != tensor.data_type()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT);
  }

  if (raw_data != nullptr) {
    return UnpackTensorWithRawData(raw_data, raw_data_len, expected_size, p_data);
  }

  if (static_cast<size_t>(tensor.int32_data_size()) != expected_size)
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "UnpackTensor: the pre-allocate size does not match the size in proto");

  constexpr int max_value = std::numeric_limits<uint16_t>::max();
  for (int i = 0; i < static_cast<int>(expected_size); i++) {
    int v = tensor.int32_data()[i];
    if (v < 0 || v > max_value) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "data overflow");
    }
    p_data[i] = BFloat16(static_cast<uint16_t>(v));
  }

  return Status::OK();
}

#define CASE_PROTO_TRACE(X, Y)                                                                     \
  case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_##X:                             \
    if (!IAllocator::CalcMemSizeForArrayWithAlignment<alignment>(size, sizeof(Y), out)) {          \
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Invalid TensorProto"); \
    }                                                                                              \
    break;

template <size_t alignment>
common::Status GetSizeInBytesFromTensorProto(const ONNX_NAMESPACE::TensorProto& tensor_proto, size_t* out) {
  const auto& dims = tensor_proto.dims();
  size_t size = 1;
  for (google::protobuf::int64 dim : dims) {
    if (dim < 0 || static_cast<uint64_t>(dim) >= std::numeric_limits<size_t>::max()) {
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Invalid TensorProto");
    }
    if (!IAllocator::CalcMemSizeForArray(size, static_cast<size_t>(dim), &size)) {
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Invalid TensorProto");
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
    CASE_PROTO_TRACE(STRING, std::string);
    default:
      return common::Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED);
  }
  return Status::OK();
}

TensorShape GetTensorShapeFromTensorShapeProto(const ONNX_NAMESPACE::TensorShapeProto& tensor_shape_proto) {
  const auto& dims = tensor_shape_proto.dim();
  std::vector<int64_t> tensor_shape_vec(static_cast<size_t>(dims.size()));
  for (int i = 0; i < dims.size(); ++i) {
    tensor_shape_vec[i] = HasDimValue(dims[i]) ? dims[i].dim_value()
                                               : -1; /* symbolic dimensions are represented as -1 in onnxruntime*/
  }
  return TensorShape(std::move(tensor_shape_vec));
}

struct UnInitializeParam {
  void* preallocated;
  size_t preallocated_size;
  ONNXTensorElementDataType ele_type;
};

ORT_API_STATUS_IMPL(OrtInitializeBufferForTensor, _In_opt_ void* input, size_t input_len,
                    enum ONNXTensorElementDataType type) {
  try {
    if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING || input == nullptr) return nullptr;
    size_t tensor_size = input_len / sizeof(std::string);
    std::string* ptr = reinterpret_cast<std::string*>(input);
    for (size_t i = 0, n = tensor_size; i < n; ++i) {
      new (ptr + i) std::string();
    }
  } catch (const std::exception& ex) {
    return OrtApis::CreateStatus(ORT_RUNTIME_EXCEPTION, ex.what());
  }
  return nullptr;
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

static void UnInitTensor(void* param) noexcept {
  UnInitializeParam* p = reinterpret_cast<UnInitializeParam*>(param);
  OrtUninitializeBuffer(p->preallocated, p->preallocated_size, p->ele_type);
  delete p;
}

#define CASE_PROTO(X, Y)                                                                                            \
  case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_##X:                                              \
    ORT_RETURN_IF_ERROR(                                                                                            \
        UnpackTensor<Y>(tensor_proto, raw_data, raw_data_len, (Y*)preallocated, static_cast<size_t>(tensor_size))); \
    break;

class AutoDelete {
 public:
  OrtCallback d{nullptr, nullptr};
  AutoDelete() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(AutoDelete);
  ~AutoDelete() {
    if (d.f != nullptr) {
      d.f(d.param);
    }
  }
};

static void DeleteCharArray(void* param) noexcept {
  auto arr = reinterpret_cast<char*>(param);
  delete[] arr;
}

static Status GetFileContent(
    const Env& env, const ORTCHAR_T* file_path, FileOffsetType offset, size_t length,
    void*& raw_buffer, OrtCallback& deleter) {
  // query length if it is 0
  if (length == 0) {
    ORT_RETURN_IF_ERROR(env.GetFileLength(file_path, length));
  }

  // first, try to map into memory
  {
    Env::MappedMemoryPtr mapped_memory{};
    auto status = env.MapFileIntoMemory(file_path, offset, length, mapped_memory);
    if (status.IsOK()) {
      deleter = mapped_memory.get_deleter().callback;
      raw_buffer = mapped_memory.release();
      return Status::OK();
    }
  }

  // if that fails, try to copy
  auto buffer = onnxruntime::make_unique<char[]>(length);
  ORT_RETURN_IF_ERROR(env.ReadFileIntoBuffer(
      file_path, offset, length, gsl::make_span(buffer.get(), length)));

  deleter = OrtCallback{DeleteCharArray, buffer.get()};
  raw_buffer = buffer.release();
  return Status::OK();
}

static void MoveOrtCallback(OrtCallback& from, OrtCallback& to) {
  to.f = from.f;
  to.param = from.param;
  from.f = nullptr;
  from.param = nullptr;
}
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 6239)
#endif
Status TensorProtoToMLValue(const Env& env, const ORTCHAR_T* tensor_proto_path,
                            const ONNX_NAMESPACE::TensorProto& tensor_proto, const MemBuffer& m, OrtValue& value,
                            OrtCallback& deleter) {
  const OrtMemoryInfo& allocator = m.GetAllocInfo();
  ONNXTensorElementDataType ele_type = utils::GetTensorElementType(tensor_proto);
  deleter.f = nullptr;
  deleter.param = nullptr;
  void* raw_data = nullptr;
  size_t raw_data_len = 0;
  const DataTypeImpl* const type = DataTypeImpl::TensorTypeFromONNXEnum(tensor_proto.data_type())->GetElementType();
  AutoDelete deleter_for_file_data;
  void* tensor_data;
  {
    if (tensor_proto.data_location() == TensorProto_DataLocation_EXTERNAL) {
      if (ele_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING)
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "string tensor can not have raw data");

      std::unique_ptr<ExternalDataInfo> external_data_info;
      ORT_RETURN_IF_ERROR(ExternalDataInfo::Create(tensor_proto.external_data(), external_data_info));
      std::basic_string<ORTCHAR_T> full_path;
      if (tensor_proto_path != nullptr) {
        ORT_RETURN_IF_ERROR(GetDirNameFromFilePath(tensor_proto_path, full_path));
        full_path = ConcatPathComponent<ORTCHAR_T>(full_path, external_data_info->GetRelPath());
      } else {
        full_path = external_data_info->GetRelPath();
      }
      raw_data_len = external_data_info->GetLength();
      // load the file
      ORT_RETURN_IF_ERROR(GetFileContent(
          env, full_path.c_str(), external_data_info->GetOffset(), raw_data_len,
          raw_data, deleter_for_file_data.d));
    } else if (utils::HasRawData(tensor_proto)) {
      if (ele_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING)
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "string tensor can not have raw data");
      raw_data = const_cast<char*>(tensor_proto.raw_data().data());
      // TODO The line above has const-correctness issues. Below is a possible fix which copies the tensor_proto data
      //      into a writeable buffer. However, it requires extra memory which may exceed the limit for certain tests.
      //auto buffer = onnxruntime::make_unique<char[]>(tensor_proto.raw_data().size());
      //std::memcpy(buffer.get(), tensor_proto.raw_data().data(), tensor_proto.raw_data().size());
      //deleter_for_file_data.d = OrtCallback{DeleteCharArray, buffer.get()};
      //raw_data = buffer.release();
      raw_data_len = tensor_proto.raw_data().size();
    }
    if (endian::native == endian::little && raw_data != nullptr && deleter_for_file_data.d.f != nullptr) {
      tensor_data = raw_data;
      MoveOrtCallback(deleter_for_file_data.d, deleter);
    } else {
      void* preallocated = m.GetBuffer();
      size_t preallocated_size = m.GetLen();
      int64_t tensor_size = 1;
      {
        for (auto i : tensor_proto.dims()) {
          if (i < 0)
            return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "tensor can't contain negative dims");
          tensor_size *= i;
        }
      }
      // tensor_size could be zero. see test_slice_start_out_of_bounds\test_data_set_0\output_0.pb
      if (static_cast<uint64_t>(tensor_size) > SIZE_MAX) {
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "size overflow");
      }
      size_t size_to_allocate;
      if (!IAllocator::CalcMemSizeForArrayWithAlignment<0>(static_cast<size_t>(tensor_size), type->Size(),
                                                           &size_to_allocate)) {
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "size overflow");
      }

      if (preallocated && preallocated_size < size_to_allocate)
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "The buffer planner is not consistent with tensor buffer size, expected ",
                               size_to_allocate, ", got ", preallocated_size);
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
        case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_STRING:
          if (preallocated != nullptr) {
            OrtStatus* status = OrtInitializeBufferForTensor(preallocated, preallocated_size, ele_type);
            if (status != nullptr) {
              OrtApis::ReleaseStatus(status);
              return Status(common::ONNXRUNTIME, common::FAIL, "initialize preallocated buffer failed");
            }

            deleter.f = UnInitTensor;
            deleter.param = new UnInitializeParam{preallocated, preallocated_size, ele_type};
          }
          ORT_RETURN_IF_ERROR(UnpackTensor<std::string>(tensor_proto, raw_data, raw_data_len,
                                                        (std::string*)preallocated, static_cast<size_t>(tensor_size)));
          break;
        default: {
          std::ostringstream ostr;
          ostr << "Initialized tensor with unexpected type: " << tensor_proto.data_type();
          return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, ostr.str());
        }
      }
      tensor_data = preallocated;
    }
  }
  std::vector<int64_t> tensor_shape_vec = GetTensorShapeFromTensorProto(tensor_proto);
  // Note: We permit an empty tensor_shape_vec, and treat it as a scalar (a tensor of size 1).
  TensorShape tensor_shape{tensor_shape_vec};

  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  value.Init(new Tensor(type, tensor_shape, tensor_data, allocator), ml_tensor,
             ml_tensor->GetDeleteFunc());
  return Status::OK();
}
#ifdef _MSC_VER
#pragma warning(pop)
#pragma warning(disable : 6239)
#endif
#define CASE_TYPE(X)                             \
  case ONNX_NAMESPACE::TensorProto_DataType_##X: \
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

ONNXTensorElementDataType GetTensorElementType(const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  return CApiElementTypeFromProtoType(tensor_proto.data_type());
}

ONNX_NAMESPACE::TensorProto TensorToTensorProto(const Tensor& tensor, const std::string& tensor_proto_name) {
  // Given we are using the raw_data field in the protobuf, this will work only for little-endian format.
  ORT_ENFORCE(endian::native == endian::little);

  // Set name, dimensions, type, and data of the TensorProto.
  ONNX_NAMESPACE::TensorProto tensor_proto;

  tensor_proto.set_name(tensor_proto_name);

  for (auto& dim : tensor.Shape().GetDims()) {
    tensor_proto.add_dims(dim);
  }

  tensor_proto.set_data_type(tensor.GetElementType());
  if (tensor.IsDataTypeString()) {
    auto* mutable_string_data = tensor_proto.mutable_string_data();
    auto f = tensor.Data<std::string>();
    auto end = f + tensor.Shape().Size();
    for (; f < end; ++f) {
      *mutable_string_data->Add() = *f;
    }
  } else {
    tensor_proto.set_raw_data(tensor.DataRaw(), tensor.SizeInBytes());
  }

  return tensor_proto;
}

common::Status ConstantNodeProtoToTensorProto(const ONNX_NAMESPACE::NodeProto& node,
                                              ONNX_NAMESPACE::TensorProto& tensor) {
  const AttributeProto& constant_attribute = node.attribute(0);

  switch (constant_attribute.type()) {
    case AttributeProto_AttributeType_TENSOR:
      tensor = constant_attribute.t();
      break;
    case AttributeProto_AttributeType_FLOAT:
      tensor.set_data_type(TensorProto_DataType_FLOAT);
      tensor.add_float_data(constant_attribute.f());
      break;
    case AttributeProto_AttributeType_FLOATS:
      tensor.set_data_type(TensorProto_DataType_FLOAT);
      *tensor.mutable_float_data() = constant_attribute.floats();
      break;
    case AttributeProto_AttributeType_INT:
      tensor.set_data_type(TensorProto_DataType_INT64);
      tensor.add_int64_data(constant_attribute.i());
      break;
    case AttributeProto_AttributeType_INTS:
      tensor.set_data_type(TensorProto_DataType_INT64);
      *tensor.mutable_int64_data() = constant_attribute.ints();
      break;
    case AttributeProto_AttributeType_STRING:
      tensor.set_data_type(TensorProto_DataType_STRING);
      tensor.add_string_data(constant_attribute.s());
      break;
    case AttributeProto_AttributeType_STRINGS: {
      tensor.set_data_type(TensorProto_DataType_STRING);
      *tensor.mutable_string_data() = constant_attribute.strings();
      break;
    }
    case AttributeProto_AttributeType_SPARSE_TENSOR: {
      auto& s = constant_attribute.sparse_tensor();
      ORT_RETURN_IF_ERROR(SparseTensorProtoToDenseTensorProto(s, tensor));
      break;
    }
    default:
      ORT_THROW("Unsupported attribute value type of ", constant_attribute.type(),
                " in 'Constant' node '", node.name(), "'");
  }

  // set name last in case attribute type was tensor (would copy over name)
  *(tensor.mutable_name()) = node.output(0);

  return Status::OK();
}

template <typename T>
static Status CopySparseData(size_t n_sparse_elements,
                             const ONNX_NAMESPACE::TensorProto& indices,
                             gsl::span<const int64_t> dims,
                             std::function<void(size_t from_idx, size_t to_idx)> copier) {
  Status status = Status::OK();
  TensorShape indices_shape(indices.dims().data(), indices.dims().size());

  auto indices_data = gsl::make_span<const int64_t>(indices.int64_data().data(), static_cast<size_t>(indices_shape.Size()));

  if (indices_shape.NumDimensions() == 1) {
    // flattened indexes
    for (size_t i = 0; i < n_sparse_elements; ++i) {
      copier(i, static_cast<size_t>(indices_data[i]));
    }
  } else if (indices_shape.NumDimensions() == 2) {
    // entries in format {NNZ, rank}
    size_t rank = static_cast<size_t>(indices_shape[1]);
    ORT_ENFORCE(rank == dims.size() && rank > 0);
    const int64_t* cur_index = indices_data.data();
    std::vector<size_t> multipliers;
    multipliers.resize(rank);

    // calculate sum of inner dimension elements for each dimension.
    // e.g. if shape {2,3,4}, the result should be {3*4, 4, 1}
    multipliers[rank - 1] = 1;
    for (int32_t r = static_cast<int32_t>(rank) - 2; r >= 0; --r) {
      multipliers[r] = static_cast<size_t>(dims[r + 1]) * multipliers[r + 1];
    }

    // calculate the offset for the entry
    // e.g. if shape was {2,3,4} and entry was (1, 0, 2) the offset is 14
    // as there are 2 rows, each with 12 entries per row
    for (size_t i = 0; i < n_sparse_elements; ++i) {
      size_t idx = 0;
      for (size_t j = 0; j < rank; ++j) {
        idx += static_cast<size_t>(cur_index[j]) * multipliers[j];
      }

      copier(i, idx);
      cur_index += rank;
    }

    ORT_ENFORCE(cur_index == &*indices_data.cend());
  } else {
    status = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH, "Invalid SparseTensor indices. Should be rank 0 or 1. Got:",
                             indices_shape);
  }

  return status;
}

common::Status SparseTensorProtoToDenseTensorProto(const ONNX_NAMESPACE::SparseTensorProto& sparse,
                                                   ONNX_NAMESPACE::TensorProto& dense) {
  Status status = Status::OK();

  const auto& sparse_values = sparse.values();
  auto type = sparse_values.data_type();
  dense.set_data_type(type);

  SafeInt<size_t> n_sparse_elements = 1;
  for (auto dim : sparse_values.dims()) {
    n_sparse_elements *= dim;
  }

  SafeInt<size_t> n_dense_elements = 1;
  for (auto dim : sparse.dims()) {
    n_dense_elements *= dim;
    dense.add_dims(dim);
  }

  const auto& indices = sparse.indices();
  auto dims = gsl::make_span<const int64_t>(dense.dims().data(), dense.dims().size());

  // need to read in sparse data first as it could be in a type specific field, in raw data, or in external data
  size_t sparse_bytes = 0;
  ORT_RETURN_IF_ERROR(GetSizeInBytesFromTensorProto<0>(sparse_values, &sparse_bytes));

  if (type != TensorProto_DataType_STRING) {
    std::vector<unsigned char> sparse_data_storage(sparse_bytes, 0);
    void* sparse_data = sparse_data_storage.data();

    size_t element_size = 0;

    // setup buffer for output
    switch (type) {
      case TensorProto_DataType_FLOAT: {
        element_size = sizeof(float);
        UnpackTensor<float>(sparse_values, static_cast<float*>(sparse_data), n_sparse_elements);
        break;
      }
      case TensorProto_DataType_INT64: {
        element_size = sizeof(int64_t);
        UnpackTensor<int64_t>(sparse_values, static_cast<int64_t*>(sparse_data), n_sparse_elements);
        break;
      }
      case TensorProto_DataType_INT32: {
        element_size = sizeof(int32_t);
        UnpackTensor<int32_t>(sparse_values, static_cast<int32_t*>(sparse_data), n_sparse_elements);
        break;
      }
      case TensorProto_DataType_DOUBLE: {
        element_size = sizeof(double);
        UnpackTensor<double>(sparse_values, static_cast<double*>(sparse_data), n_sparse_elements);
        break;
      }
      case TensorProto_DataType_UINT32: {
        element_size = sizeof(uint32_t);
        UnpackTensor<uint32_t>(sparse_values, static_cast<uint32_t*>(sparse_data), n_sparse_elements);
        break;
      }
      case TensorProto_DataType_UINT64: {
        element_size = sizeof(uint64_t);
        UnpackTensor<uint64_t>(sparse_values, static_cast<uint64_t*>(sparse_data), n_sparse_elements);
        break;
      }
      default:
        status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported sparse tensor data type of ", type);
    }

    // by putting the data into a std::string we can avoid a copy as set_raw_data can do a std::move
    // into the TensorProto. however to actually write to the buffer we have created in the std::string we need
    // this somewhat dirty hack to get a mutable pointer. we could alternatively use &dense_data_storage.front()
    // but using const_cast makes it more obvious we're doing something ugly.
    std::string dense_data_storage(n_dense_elements * element_size, 0);
    void* dense_data = const_cast<char*>(dense_data_storage.data());

    switch (element_size) {
      case 4: {
        auto dense_data_span = gsl::make_span<uint32_t>(static_cast<uint32_t*>(dense_data), n_dense_elements);
        status = CopySparseData<uint32_t>(
            n_sparse_elements,
            indices, dims,
            [sparse_data, dense_data_span](size_t from_idx, size_t to_idx) {
              dense_data_span[to_idx] = static_cast<const uint32_t*>(sparse_data)[from_idx];
            });

        break;
      }
      case 8: {
        auto dense_data_span = gsl::make_span<uint64_t>(static_cast<uint64_t*>(dense_data), n_dense_elements);
        status = CopySparseData<uint64_t>(
            n_sparse_elements,
            indices, dims,
            [sparse_data, dense_data_span](size_t from_idx, size_t to_idx) {
              dense_data_span[to_idx] = static_cast<const uint64_t*>(sparse_data)[from_idx];
            });

        break;
      }
    }

    dense.set_raw_data(std::move(dense_data_storage));

  } else {
    // strings need to be handled differently as they can't use raw data (as per ONNX rules)
    std::vector<std::string> sparse_data(n_sparse_elements);
    UnpackTensor<std::string>(sparse_values, sparse_data.data(), n_sparse_elements);

    // RepeatedPtrField<std::string> doesn't have a Resize method so manually add elements
    auto dense_strings = dense.mutable_string_data();
    dense_strings->Reserve(n_dense_elements);
    for (int64_t j = 0; j < n_dense_elements; ++j) {
      dense_strings->Add("");
    }

    status = CopySparseData<std::string>(
        n_sparse_elements,
        indices, dims,
        [&sparse_values, &dense_strings](size_t from_idx, size_t to_idx) {
          const std::string& input = sparse_values.string_data()[SafeInt<int32_t>(from_idx)];
          *dense_strings->Mutable(SafeInt<int32_t>(to_idx)) = input;
        });
  }

  return status;
}

template common::Status GetSizeInBytesFromTensorProto<256>(const ONNX_NAMESPACE::TensorProto& tensor_proto,
                                                           size_t* out);
template common::Status GetSizeInBytesFromTensorProto<0>(const ONNX_NAMESPACE::TensorProto& tensor_proto, size_t* out);
}  // namespace utils
}  // namespace onnxruntime
