#pragma once

#define ORT_API_MANUAL_INIT
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#include "onnx_extended_helpers.h"

namespace ortops {

////////////////////////
// errors and exceptions
////////////////////////

template <typename T> struct CTypeToOnnxType {
  ONNXTensorElementDataType onnx_type() const;
};

template <> struct CTypeToOnnxType<float> {
  inline ONNXTensorElementDataType onnx_type() const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
};

template <> struct CTypeToOnnxType<int64_t> {
  inline ONNXTensorElementDataType onnx_type() const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  }
};

template <> struct CTypeToOnnxType<double> {
  inline ONNXTensorElementDataType onnx_type() const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
  }
};

inline void _ThrowOnError_(OrtStatus *ort_status, const char *filename, int line,
                           const OrtApi &api) {
  if (ort_status) {
    OrtErrorCode code = api.GetErrorCode(ort_status);
    if (code == ORT_OK) {
      api.ReleaseStatus(ort_status);
    } else {
      std::string message(api.GetErrorMessage(ort_status));
      api.ReleaseStatus(ort_status);
      if (code != ORT_OK) {
        throw std::runtime_error(onnx_extended_helpers::MakeString(
            "error: onnxruntime(", code, "), ", message, "\n    ", filename, ":", line));
      }
    }
  }
}

#define ThrowOnError(api, ort_status) _ThrowOnError_(ort_status, __FILE__, __LINE__, api)

////////
// types
////////

inline bool is_float8(ONNXTensorElementDataType type) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080 && ORT_VERSION >= 1160
  return (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN) ||
         (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ) ||
         (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2) ||
         (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ);
#else
  return type >= 17;
#endif
}

////////////////
// kernel inputs
////////////////

inline std::string KernelInfoGetInputName(const OrtApi &api, const OrtKernelInfo *info,
                                          int index) {
  std::size_t size;
  OrtStatus *status = api.KernelInfo_GetInputName(info, index, nullptr, &size);
  if (status != nullptr) {
    OrtErrorCode code = api.GetErrorCode(status);
    if (code == ORT_FAIL) {
      api.ReleaseStatus(status);
      return std::string();
    } else {
      ThrowOnError(api, status);
    }
    api.ReleaseStatus(status);
  }
  std::string str_out;
  str_out.resize(size);
  ThrowOnError(api, api.KernelInfo_GetInputName(info, index, &str_out[0], &size));
  str_out.resize(size - 1); // remove the terminating character '\0'
  return str_out;
}

////////////////////
// kernel attributes
////////////////////

class AttOrtValue {
public:
  ONNXTensorElementDataType elem_type;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
  inline void clear() {
    bytes.clear();
    shape.clear();
    elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
  inline bool empty() const { return bytes.empty(); }
};

inline std::string KernelInfoGetOptionalAttributeString(const OrtApi &api,
                                                        const OrtKernelInfo *info,
                                                        const char *name,
                                                        const std::string &default_value) {
  std::size_t size = 0;
  std::string str_out;

  OrtStatus *status = api.KernelInfoGetAttribute_string(info, name, nullptr, &size);

  if (status != nullptr) {
    OrtErrorCode code = api.GetErrorCode(status);
    if (code == ORT_FAIL) {
      api.ReleaseStatus(status);
      return default_value;
    } else {
      ThrowOnError(api, status);
    }
    api.ReleaseStatus(status);
  }
  str_out.resize(size);
  ThrowOnError(api, api.KernelInfoGetAttribute_string(info, name, &str_out[0], &size));
  str_out.resize(size - 1); // remove the terminating character '\0'
  return str_out;
}

template <typename T>
inline OrtStatus *KernelInfoGetAttributeApi(const OrtApi &api, const OrtKernelInfo *info,
                                            const char *name, T &out);

template <>
inline OrtStatus *KernelInfoGetAttributeApi<int64_t>(const OrtApi &api,
                                                     const OrtKernelInfo *info,
                                                     const char *name, int64_t &out) {
  return api.KernelInfoGetAttribute_int64(info, name, &out);
}

template <>
inline OrtStatus *KernelInfoGetAttributeApi<float>(const OrtApi &api, const OrtKernelInfo *info,
                                                   const char *name, float &out) {
  return api.KernelInfoGetAttribute_float(info, name, &out);
}

template <>
inline OrtStatus *
KernelInfoGetAttributeApi<std::vector<float>>(const OrtApi &api, const OrtKernelInfo *info,
                                              const char *name, std::vector<float> &out) {
  std::size_t size = 0;

  // Feed nullptr for the data buffer to query the true size of the attribute
  OrtStatus *status = api.KernelInfoGetAttributeArray_float(info, name, nullptr, &size);

  if (status == nullptr) {
    out.resize(size);
    status = api.KernelInfoGetAttributeArray_float(info, name, out.data(), &size);
  }

  return status;
}

template <>
inline OrtStatus *
KernelInfoGetAttributeApi<std::vector<int64_t>>(const OrtApi &api, const OrtKernelInfo *info,
                                                const char *name, std::vector<int64_t> &out) {
  std::size_t size = 0;

  // Feed nullptr for the data buffer to query the true size of the attribute
  OrtStatus *status = api.KernelInfoGetAttributeArray_int64(info, name, nullptr, &size);

  if (status == nullptr) {
    out.resize(size);
    ThrowOnError(api, api.KernelInfoGetAttributeArray_int64(info, name, out.data(), &size));
  }
  return status;
}

inline std::size_t ElementSize(ONNXTensorElementDataType elem_type) {
  switch (elem_type) {
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return 8;
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    return 4;
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    return 2;
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return 1;
  default:
    throw std::runtime_error("One element type is not implemented in function "
                             "`ortops::ElementSize()`.");
  }
}

template <>
inline OrtStatus *KernelInfoGetAttributeApi<AttOrtValue>(const OrtApi &api,
                                                         const OrtKernelInfo *info,
                                                         const char *name, AttOrtValue &out) {
  OrtAllocator *cpu_allocator;
  ThrowOnError(api, api.GetAllocatorWithDefaultOptions(&cpu_allocator));
  OrtValue *value_tensor = nullptr;
  OrtStatus *status =
      api.KernelInfoGetAttribute_tensor(info, name, cpu_allocator, &value_tensor);
  if (status != nullptr) {
    return status;
  }
  OrtTensorTypeAndShapeInfo *shape_info;
  ThrowOnError(api, api.GetTensorTypeAndShape(value_tensor, &shape_info));
  ThrowOnError(api, api.GetTensorElementType(shape_info, &out.elem_type));
  std::size_t n_dims;
  ThrowOnError(api, api.GetDimensionsCount(shape_info, &n_dims));
  out.shape.resize(n_dims);
  ThrowOnError(api, api.GetDimensions(shape_info, out.shape.data(), n_dims));
  std::size_t size_tensor;
  ThrowOnError(api, api.GetTensorShapeElementCount(shape_info, &size_tensor));
  void *data;
  std::size_t size_elem = ElementSize(out.elem_type);
  ThrowOnError(api, api.GetTensorMutableData(value_tensor, &data));

  out.bytes.resize(size_tensor * size_elem);
  memcpy(out.bytes.data(), data, out.bytes.size());

  if (value_tensor != nullptr)
    api.ReleaseValue(value_tensor);
  return nullptr;
}

template <>
inline OrtStatus *KernelInfoGetAttributeApi<std::vector<std::string>>(
    const OrtApi & /* api */, const OrtKernelInfo * /* info */, const char * /* name */,
    std::vector<std::string> & /* output */) {
  EXT_THROW("Unable to retrieve attribute as an array of strings. "
            "You should use a single comma separated string.");
}

template <typename T>
inline T KernelInfoGetOptionalAttribute(const OrtApi &api, const OrtKernelInfo *info,
                                        const char *name, T default_value) {
  T out;
  OrtStatus *status = KernelInfoGetAttributeApi<T>(api, info, name, out);

  if (status == nullptr)
    return out;
  OrtErrorCode code = api.GetErrorCode(status);
  if (code == ORT_FAIL) {
    api.ReleaseStatus(status);
    return default_value;
  }

  ThrowOnError(api, status);
  return default_value;
}

inline bool KernelInfoGetOptionalAttributeInt64AsBool(const OrtApi &api,
                                                      const OrtKernelInfo *info,
                                                      const char *name, bool default_value) {
  int64_t value =
      KernelInfoGetOptionalAttribute<int64_t>(api, info, name, default_value ? 1 : 0);
  return value == 1;
}

} // namespace ortops
