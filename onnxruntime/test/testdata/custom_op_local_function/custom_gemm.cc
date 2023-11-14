// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "custom_gemm.h"

#ifndef ORT_ENFORCE
#define ORT_ENFORCE(cond, ...) \
  if (!(cond)) ORT_CXX_API_THROW("Initialization failed.", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
#endif

namespace Cpu {

////////////////
// kernel inputs
////////////////

void _ThrowOnError_(OrtStatus* ort_status, const char* filename,
                    int /*line*/, const OrtApi& api) {
  if (ort_status) {
    OrtErrorCode code = api.GetErrorCode(ort_status);
    if (code == ORT_OK) {
      api.ReleaseStatus(ort_status);
    } else {
      std::string message(api.GetErrorMessage(ort_status));
      api.ReleaseStatus(ort_status);
      if (code != ORT_OK) {
        throw std::runtime_error(message + std::string(":") + std::string(filename));
      }
    }
  }
}

#define ThrowOnError(api, ort_status) \
  _ThrowOnError_(ort_status, __FILE__, __LINE__, api)

std::string KernelInfoGetInputName(const OrtApi& api,
                                   const OrtKernelInfo* info,
                                   int index) {
  size_t size;
  OrtStatus* status = api.KernelInfo_GetInputName(info, index, nullptr, &size);
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
  std::string out;
  out.resize(size);
  ThrowOnError(api, api.KernelInfo_GetInputName(info, index, &out[0], &size));
  out.resize(size - 1);  // remove the terminating character '\0'
  return out;
}

////////////////////
// kernel attributes
////////////////////

std::string KernelInfoGetOptionalAttributeString(
    const OrtApi& api, const OrtKernelInfo* info, const char* name,
    const std::string& default_value) {
  size_t size = 0;
  std::string out;

  OrtStatus* status =
      api.KernelInfoGetAttribute_string(info, name, nullptr, &size);

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
  out.resize(size);
  ThrowOnError(api,
               api.KernelInfoGetAttribute_string(info, name, &out[0], &size));
  out.resize(size - 1);  // remove the terminating character '\0'
  return out;
}

template <typename T>
OrtStatus* KernelInfoGetAttributeApi(const OrtApi& api,
                                     const OrtKernelInfo* info,
                                     const char* name, T& out);

template <>
OrtStatus*
KernelInfoGetAttributeApi<int64_t>(const OrtApi& api, const OrtKernelInfo* info,
                                   const char* name, int64_t& out) {
  return api.KernelInfoGetAttribute_int64(info, name, &out);
}

template <>
OrtStatus*
KernelInfoGetAttributeApi<float>(const OrtApi& api, const OrtKernelInfo* info,
                                 const char* name, float& out) {
  return api.KernelInfoGetAttribute_float(info, name, &out);
}

template <>
OrtStatus* KernelInfoGetAttributeApi<std::vector<float>>(
    const OrtApi& api, const OrtKernelInfo* info, const char* name,
    std::vector<float>& out) {
  size_t size = 0;

  // Feed nullptr for the data buffer to query the true size of the attribute
  OrtStatus* status =
      api.KernelInfoGetAttributeArray_float(info, name, nullptr, &size);

  if (status == nullptr) {
    out.resize(size);
    status =
        api.KernelInfoGetAttributeArray_float(info, name, out.data(), &size);
  }

  return status;
}

template <>
OrtStatus* KernelInfoGetAttributeApi<std::vector<int64_t>>(
    const OrtApi& api, const OrtKernelInfo* info, const char* name,
    std::vector<int64_t>& out) {
  size_t size = 0;

  // Feed nullptr for the data buffer to query the true size of the attribute
  OrtStatus* status =
      api.KernelInfoGetAttributeArray_int64(info, name, nullptr, &size);

  if (status == nullptr) {
    out.resize(size);
    ThrowOnError(api, api.KernelInfoGetAttributeArray_int64(info, name,
                                                            out.data(), &size));
  }
  return status;
}

template <>
OrtStatus* KernelInfoGetAttributeApi<std::vector<std::string>>(
    const OrtApi& /*api*/, const OrtKernelInfo* /*info*/, const char* /*name*/,
    std::vector<std::string>& /*output*/) {
  ORT_CXX_API_THROW(
      "Unable to retrieve attribute as an array of strings. "
      "You should use a single comma separated string.",
      OrtErrorCode::ORT_RUNTIME_EXCEPTION);
}

template <typename T>
T KernelInfoGetOptionalAttribute(const OrtApi& api,
                                 const OrtKernelInfo* info,
                                 const char* name, T default_value) {
  T out;
  OrtStatus* status = KernelInfoGetAttributeApi<T>(api, info, name, out);

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

bool KernelInfoGetOptionalAttributeInt64AsBool(const OrtApi& api,
                                               const OrtKernelInfo* info,
                                               const char* name,
                                               bool default_value) {
  int64_t value = KernelInfoGetOptionalAttribute<int64_t>(
      api, info, name, default_value ? 1 : 0);
  return value == 1;
}

//////
// FP8
//////

uint8_t float_to_e4m3fn(float v, bool saturate = true) {
  uint32_t b;
  std::memcpy(&b, &v, sizeof(b));

  uint8_t val = static_cast<uint8_t>((b & 0x80000000) >> 24);  // sign
  if ((b & 0x7fffffff) == 0x7f800000) {                        // infinity
    if (saturate) {
      val |= 126;
    } else {
      val |= 0x7f;
    }
  } else if ((b & 0x7f800000) == 0x7f800000) {  // NaN
    val |= 0x7f;
  } else {
    uint8_t e = static_cast<uint8_t>((b & 0x7F800000) >> 23);  // exponent
    uint32_t m = static_cast<uint32_t>(b & 0x007FFFFF);        // mantissa
    if (e != 0) {
      if (e < 117) {
      } else if (e < 121) {
        // denormalized number
        auto d = 120 - e;
        if (d < 3) {
          val |= 1 << (2 - d);
          val |= m >> (21 + d);
        } else if (m > 0) {
          val |= 1;
        }
        auto mask = 1 << (20 + d);
        if ((m & mask) &&
            ((val & 1) || ((m & (mask - 1)) > 0) ||
             ((m & mask) && (m & (mask << 1)) && ((m & (mask - 1)) == 0)))) {
          // rounding
          val += 1;
        }
      } else if (e < 136) {
        // normalized number
        auto ex = e - 120;
        if (ex == 0) {
          val |= 0x4;
          val |= m >> 21;
        } else {
          val |= ex << 3;
          val |= m >> 20;
          if ((val & 0x7F) == 0x7F) {
            val &= 0xFE;
          }
        }
        if ((m & 0x80000) && ((m & 0x100000) || (m & 0x7FFFF))) {
          if ((val & 0x7F) < 0x7E) {
            // rounding
            val += 1;
          } else if (!saturate) {
            val |= 0x7F;
          }
        }
      } else if (saturate) {
        val |= 126;  // 0b01111110
      } else {
        val |= 0x7F;
      }
    }
  }
  return val;
}

void float_to_e4m3fn(int64_t n, const float* src, uint8_t* dst,
                     bool saturate = true) {
  for (int64_t i = 0; i < n; ++i) {
    dst[i] = float_to_e4m3fn(src[i], saturate);
  }
}

void float_to_e4m3fn(int64_t n, const float* src, uint8_t* dst,
                     float scale, bool saturate = true) {
  for (int64_t i = 0; i < n; ++i) {
    dst[i] = float_to_e4m3fn(src[i] / scale, saturate);
  }
}

float e4m3fn_to_float(uint8_t val) {
  uint32_t res;
  if (val == 255) {
    res = 0xffc00000;
  } else if (val == 127) {
    res = 0x7fc00000;
  } else {
    uint32_t expo = (val & 0x78) >> 3;
    uint32_t mant = val & 0x07;
    uint32_t sign = val & 0x80;
    res = sign << 24;
    if (expo == 0) {
      if (mant > 0) {
        expo = 0x7F - 7;
        if ((mant & 0x4) == 0) {
          mant &= 0x3;
          mant <<= 1;
          expo -= 1;
        }
        if ((mant & 0x4) == 0) {
          mant &= 0x3;
          mant <<= 1;
          expo -= 1;
        }
        res |= (mant & 0x3) << 21;
        res |= expo << 23;
      }
    } else {
      res |= mant << 20;
      expo -= 0x7;
      expo += 0x7F;
      res |= expo << 23;
    }
  }
  float float_res;
  std::memcpy(&float_res, &res, sizeof(float));
  return float_res;
}

void e4m3fn_to_float(int64_t n, const uint8_t* src, float* dst) {
  for (int64_t i = 0; i < n; ++i) {
    dst[i] = e4m3fn_to_float(src[i]);
  }
}

void e4m3fn_to_float(int64_t n, const uint8_t* src, float* dst, float scale) {
  for (int64_t i = 0; i < n; ++i) {
    dst[i] = e4m3fn_to_float(src[i]) * scale;
  }
}

//////////////////
// CustomGemmOp...
//////////////////

void* CustomGemmOp::CreateKernel(const OrtApi& api,
                                 const OrtKernelInfo* info) const {
  return std::make_unique<CustomGemmKernel>(api, info).release();
}

const char* CustomGemmOp::GetName() const { return op_name_; }

const char* CustomGemmOp::GetExecutionProviderType() const {
  return "CPUExecutionProvider";
}

size_t CustomGemmOp::GetInputTypeCount() const { return 6; };

ONNXTensorElementDataType CustomGemmOp::GetInputType(size_t index) const {
  switch (index) {
    case 0:  // A
    case 1:  // B
      return ab_type_;
    case 2:  // C
      return c_type_;
    case 3:  // scale A
    case 4:  // scale B
    case 5:  // scale Y
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    default:
      ORT_CXX_API_THROW("Input index is out of boundary.", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
  }
}

OrtCustomOpInputOutputCharacteristic
CustomGemmOp::GetInputCharacteristic(size_t index) const {
  switch (index) {
    case 0:
    case 1:
      return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
    case 2:
      return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
    case 3:
    case 4:
    case 5:
      return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
    default:
      ORT_CXX_API_THROW("Input index is out of boundary.", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
  }
}

size_t CustomGemmOp::GetOutputTypeCount() const { return 1; }

ONNXTensorElementDataType CustomGemmOp::GetOutputType(size_t index) const {
  // D, scale D
  switch (index) {
    case 0:
      return d_type_;
    default:
      ORT_CXX_API_THROW("Output index is out of boundary.", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
  }
}

OrtCustomOpInputOutputCharacteristic
CustomGemmOp::GetOutputCharacteristic(size_t index) const {
  switch (index) {
    case 0:
      return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
    default:
      ORT_CXX_API_THROW("Output index is out of boundary.", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
  }
}

///////////////////
// CustomGemmKernel
///////////////////

CustomGemmKernel::CustomGemmKernel(const OrtApi& api,
                                   const OrtKernelInfo* info) {
  rowMajor_ = KernelInfoGetOptionalAttribute<int64_t>(api, info, "rowMajor", 1);
  transA_ =
      KernelInfoGetOptionalAttributeInt64AsBool(api, info, "transA", false);
  transB_ =
      KernelInfoGetOptionalAttributeInt64AsBool(api, info, "transB", false);
  fastAccumulationMode_ = KernelInfoGetOptionalAttributeInt64AsBool(
      api, info, "fastAccumulationMode", true);
  smCount_ = KernelInfoGetOptionalAttribute<int64_t>(api, info, "smCount", 0);
  alpha_ = KernelInfoGetOptionalAttribute<float>(api, info, "alpha", 1);
  beta_ = KernelInfoGetOptionalAttribute<float>(api, info, "beta", 0);

  // A string attribute.
  std::string compute_type = KernelInfoGetOptionalAttributeString(
      api, info, "computeType", "CUBLAS_COMPUTE_32F");
  if (compute_type == "CUBLAS_COMPUTE_16F") {
    computeType_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  } else if (compute_type == "CUBLAS_COMPUTE_32F") {
    computeType_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  } else if (compute_type == "CUBLAS_COMPUTE_32F_FAST_16F") {
    computeType_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  } else if (compute_type == "CUBLAS_COMPUTE_32F_FAST_TF32") {
    computeType_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  } else {
    ORT_CXX_API_THROW("Wrong computeType", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
  }

  std::string activation =
      KernelInfoGetOptionalAttributeString(api, info, "activation", "DEFUALT");
  if (activation == "DEFUALT") {
    epilogue_ = EpiloqueGemmKernel::Default;
  } else if (activation == "RELU") {
    epilogue_ = EpiloqueGemmKernel::Relu;
  } else if (activation == "GELU") {
    epilogue_ = EpiloqueGemmKernel::Gelu;
  } else {
    ORT_CXX_API_THROW("Wrong activation", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
  }

  ThrowOnError(api, api.KernelInfo_GetInputCount(info, &n_inputs_));
  if (n_inputs_ > 5) {
    std::string name = KernelInfoGetInputName(api, info, 5);
    has_scale_Y_ = !name.empty();
  } else {
    has_scale_Y_ = false;
  }
}

void CustomGemmKernel::set(const std::vector<int64_t>& shape_A,
                           const std::vector<int64_t>& shape_B, int& M, int& N,
                           int& K, int& lda, int& ldb, int& ldd,
                           int row_major) const {
  constexpr int ir = 0;
  constexpr int ic = 1 - ir;
  if (transA_ && !transB_) {  // TN
    M = static_cast<int>(shape_A[ic]);
    N = static_cast<int>(shape_B[ic]);
    K = static_cast<int>(shape_A[ir]);
    lda = static_cast<int>(shape_A[row_major ? ic : ir]);
    ldb = static_cast<int>(shape_B[row_major ? ic : ir]);
    ldd = static_cast<int>(shape_B[row_major ? ic : ir]);
  } else if (!transA_ && !transB_) {  // NN
    M = static_cast<int>(shape_A[ir]);
    N = static_cast<int>(shape_B[ic]);
    K = static_cast<int>(shape_A[ic]);
    lda = static_cast<int>(shape_A[row_major ? ic : ir]);
    ldb = static_cast<int>(shape_B[row_major ? ic : ir]);
    ldd = static_cast<int>(shape_B[row_major ? ic : ir]);
  } else if (!transA_ && transB_) {  // NT
    M = static_cast<int>(shape_A[ir]);
    N = static_cast<int>(shape_B[ir]);
    K = static_cast<int>(shape_A[ic]);
    lda = static_cast<int>(shape_A[row_major ? ic : ir]);
    ldb = static_cast<int>(shape_B[row_major ? ic : ir]);
    ldd = static_cast<int>(shape_B[row_major ? ir : ic]);
  } else {  // TT
    M = static_cast<int>(shape_A[ic]);
    N = static_cast<int>(shape_B[ir]);
    K = static_cast<int>(shape_A[ir]);
    lda = static_cast<int>(shape_A[row_major ? ic : ir]);
    ldb = static_cast<int>(shape_B[row_major ? ic : ir]);
    ldd = static_cast<int>(shape_B[row_major ? ir : ic]);
  }
}

void check_device(const Ort::ConstValue& input, const char* /*name*/) {
  ORT_ENFORCE(input.HasValue(), "Input '", name, "' is empty.");
  auto mem = input.GetTensorMemoryInfo();
  ORT_ENFORCE(mem.GetDeviceType() ==
                  OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU,
              "Input '", name, "' is not on CPU");
}

void check_device(const Ort::UnownedValue& output, const char* /*name*/) {
  auto mem = output.GetTensorMemoryInfo();
  ORT_ENFORCE(mem.GetDeviceType() ==
                  OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU,
              "Output '", name, "' is not on CPU");
}

template <typename TValue>
ONNXTensorElementDataType GetTypeAndShape(const TValue& input,
                                          std::vector<int64_t>& shape,
                                          bool swap = false) {
  auto t = input.GetTensorTypeAndShapeInfo();
  shape = t.GetShape();
  ORT_ENFORCE(shape.size() == 2);
  if (swap) {
    std::swap(shape[0], shape[1]);
  }
  return t.GetElementType();
}

void CustomGemmKernel::Compute(OrtKernelContext* context) {
  Ort::KernelContext ctx(context);

  int n_inputs = static_cast<int>(n_inputs_);
  Ort::ConstValue scale_A, scale_B, scale_Y;
  Ort::ConstValue input_A = ctx.GetInput(0);
  Ort::ConstValue input_B = ctx.GetInput(1);
  Ort::ConstValue input_C;
  bool has_bias;
  if (n_inputs > 2) {
    input_C = ctx.GetInput(2);
    has_bias = beta_ != 0 && input_C.HasValue() && input_C.IsTensor();
  } else {
    has_bias = false;
  }

  check_device(input_A, "A");
  check_device(input_B, "B");
  if (has_bias)
    check_device(input_C, "C");

  bool has_scales = n_inputs > 3;
  bool has_scales_Y = n_inputs > 5 && has_scale_Y_;
  if (has_scales) {
    ORT_ENFORCE(n_inputs == 5 || n_inputs == 6,
                "Number of inputs must be 5 or 6 but is ", n_inputs, ".");
    scale_A = ctx.GetInput(3);
    scale_B = ctx.GetInput(4);
    check_device(scale_A, "scale_A");
    check_device(scale_B, "scale_B");
    if (has_scales_Y) {
      scale_Y = ctx.GetInput(5);
      check_device(scale_Y, "scale_Y");
    }
  } else if (n_inputs != 2 && n_inputs != 3) {
    ORT_CXX_API_THROW("Number of inputs must be 2, 3 or 6", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
  }

  switch (rowMajor_) {
    case 0:
      ComputeColMajor(ctx, n_inputs, has_bias, has_scales, has_scales_Y, input_A,
                      input_B, input_C, scale_A, scale_B, scale_Y);
      break;
    case 1:
      ComputeRowMajor(ctx, n_inputs, has_bias, has_scales, has_scales_Y, input_A,
                      input_B, input_C, scale_A, scale_B, scale_Y);
      break;
    default:
      ORT_CXX_API_THROW("Unexpected value for rowMajor", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
  }
}

void CustomGemmKernel::ComputeRowMajor(
    Ort::KernelContext& ctx, int n_inputs, bool has_bias, bool has_scales,
    bool has_scales_Y, Ort::ConstValue& input_A, Ort::ConstValue& input_B,
    Ort::ConstValue& input_C, Ort::ConstValue& scale_A,
    Ort::ConstValue& scale_B, Ort::ConstValue& scale_Y) {
  std::vector<int64_t> shape_A, shape_B, shape_C, shape_Y;
  ONNXTensorElementDataType dtype_A, dtype_B, dtype_C, dtype_Y;
  dtype_A = GetTypeAndShape(input_A, shape_A);
  dtype_B = GetTypeAndShape(input_B, shape_B);

  int M, N, K, lda, ldb, ldd;
  set(shape_A, shape_B, M, N, K, lda, ldb, ldd, 1);

  std::vector<int64_t> dimensions{M, N};
  Ort::UnownedValue Y = ctx.GetOutput(0, dimensions);
  check_device(Y, "Y");
  dtype_Y = GetTypeAndShape(Y, shape_Y);
  dtype_C = has_bias ? GetTypeAndShape(input_C, shape_C)
                     : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  ComputeGemm(ctx, n_inputs, has_bias, has_scales, has_scales_Y, dtype_A,
              dtype_B, dtype_C, dtype_Y, shape_A, shape_B, shape_C, shape_Y,
              transA_, transB_, input_A.GetTensorRawData(),
              input_B.GetTensorRawData(),
              has_bias ? input_C.GetTensorRawData() : nullptr,
              has_scales ? scale_A.GetTensorRawData() : nullptr,
              has_scales ? scale_B.GetTensorRawData() : nullptr,
              has_scales_Y ? scale_Y.GetTensorRawData() : nullptr,
              Y.GetTensorMutableRawData(), M, N, K, lda, ldb, ldd);
}

void CustomGemmKernel::ComputeColMajor(
    Ort::KernelContext& ctx, int n_inputs, bool has_bias, bool has_scales,
    bool has_scales_Y, Ort::ConstValue& input_A, Ort::ConstValue& input_B,
    Ort::ConstValue& input_C, Ort::ConstValue& scale_A,
    Ort::ConstValue& scale_B, Ort::ConstValue& scale_Y) {
  std::vector<int64_t> shape_A, shape_B, shape_C, shape_Y;
  ONNXTensorElementDataType dtype_A, dtype_B, dtype_C, dtype_Y;
  dtype_A = GetTypeAndShape(input_A, shape_A);
  dtype_B = GetTypeAndShape(input_B, shape_B);

  int M, N, K, lda, ldb, ldd;
  set(shape_A, shape_B, M, N, K, lda, ldb, ldd, 1);

  std::swap(shape_A[0], shape_A[1]);
  std::swap(shape_B[0], shape_B[1]);

  std::vector<int64_t> dimensions{M, N};
  Ort::UnownedValue Y = ctx.GetOutput(0, dimensions);
  check_device(Y, "Y");
  dtype_Y = GetTypeAndShape(Y, shape_Y);
  dtype_C = has_bias ? GetTypeAndShape(input_C, shape_C, true)
                     : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;

  ComputeGemm(ctx, n_inputs, has_bias, has_scales, has_scales_Y, dtype_B,
              dtype_A, dtype_C, dtype_Y, shape_B, shape_A, shape_C, shape_Y,
              transA_, transB_, input_B.GetTensorRawData(),
              input_A.GetTensorRawData(),
              has_bias ? input_C.GetTensorRawData() : nullptr,
              has_scales ? scale_B.GetTensorRawData() : nullptr,
              has_scales ? scale_A.GetTensorRawData() : nullptr,
              has_scales_Y ? scale_Y.GetTensorRawData() : nullptr,
              Y.GetTensorMutableRawData(), N, M, K, ldb, lda, ldd);
}

void CustomGemmKernel::ComputeGemm(
    Ort::KernelContext& ctx, int n_inputs, bool has_bias, bool has_scales,
    bool has_scales_Y, ONNXTensorElementDataType dtype_A,
    ONNXTensorElementDataType dtype_B, ONNXTensorElementDataType dtype_C,
    ONNXTensorElementDataType dtype_Y, const std::vector<int64_t>& shape_A,
    const std::vector<int64_t>& shape_B, const std::vector<int64_t>& shape_C,
    const std::vector<int64_t>& shape_Y, bool trans_A, bool trans_B,
    const void* p_input_a, const void* p_input_b, const void* p_input_c,
    const void* p_scale_a, const void* p_scale_b, const void* p_scale_y,
    void* p_output_y, int M, int N, int K, int lda, int ldb, int ldd) {
  if (dtype_A == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      dtype_B == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      dtype_C == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      dtype_Y == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
      computeType_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    ComputeGemm(ctx, n_inputs, has_bias, has_scales, has_scales_Y, shape_A,
                shape_B, shape_C, shape_Y, trans_A, trans_B,
                static_cast<const float*>(p_input_a),
                static_cast<const float*>(p_input_b),
                static_cast<const float*>(p_input_c),
                static_cast<const float*>(p_scale_a),
                static_cast<const float*>(p_scale_b),
                static_cast<const float*>(p_scale_y),
                static_cast<float*>(p_output_y), M, N, K, lda, ldb, ldd);
  } else if (dtype_A == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN &&
             dtype_B == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN &&
             dtype_C == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
             dtype_Y == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
             computeType_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    std::vector<float> c_input_a(M * K);
    std::vector<float> c_input_b(N * K);
    e4m3fn_to_float(c_input_a.size(), static_cast<const uint8_t*>(p_input_a),
                    c_input_a.data(), *(static_cast<const float*>(p_scale_a)));
    e4m3fn_to_float(c_input_b.size(), static_cast<const uint8_t*>(p_input_b),
                    c_input_b.data(), *(static_cast<const float*>(p_scale_b)));
    ComputeGemm(ctx, n_inputs, has_bias, has_scales, has_scales_Y, shape_A,
                shape_B, shape_C, shape_Y, trans_A, trans_B, c_input_a.data(),
                c_input_b.data(), static_cast<const float*>(p_input_c),
                static_cast<const float*>(p_scale_a),
                static_cast<const float*>(p_scale_b),
                static_cast<const float*>(p_scale_y),
                static_cast<float*>(p_output_y), M, N, K, lda, ldb, ldd);
  } else {
    ORT_CXX_API_THROW("Not implemented", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
  }
}

void CustomGemmKernel::ComputeGemm(
    Ort::KernelContext& /*ctx*/, int /*n_inputs*/, bool /*has_bias*/, bool has_scales,
    bool has_scales_Y, const std::vector<int64_t>& /*shape_A*/,
    const std::vector<int64_t>& /*shape_B*/, const std::vector<int64_t>& /*shape_C*/,
    const std::vector<int64_t>& /*shape_Y*/, bool transa, bool transb,
    const float* p_input_a, const float* p_input_b, const float* p_input_c,
    const float* p_scale_a, const float* p_scale_b, const float* p_scale_y,
    float* p_output_y, int M, int N, int K, int lda, int ldb, int ldd) {
  ORT_ENFORCE(has_scales || p_scale_a == nullptr || *p_scale_a == 1,
              "scale_A must be empty or one for float");
  ORT_ENFORCE(has_scales || p_scale_b == nullptr || *p_scale_b == 1,
              "scale_B must be empty or one for float.");
  ORT_ENFORCE(has_scales_Y || p_scale_y == nullptr || *p_scale_y == 1,
              "scale_Y must be empty or one for float.");

  int i, j, k;
  int MN = M * N;
  if (p_input_c == nullptr) {
    for (i = 0; i < MN; ++i) {
      p_output_y[i] = 0;
    }
  } else {
    for (i = 0; i < MN; ++i) {
      p_output_y[i] = beta_ * p_input_c[i];
    }
  }

  if (rowMajor_ == 1) {
    // rowMajor_ == 0
    if (transa) {
      if (transb) {
        for (i = 0; i < M; ++i) {
          float A_PART;
          for (k = 0; k < K; ++k) {
            A_PART = alpha_ * p_input_a[k * lda + i];
            for (j = 0; j < N; ++j) {
              p_output_y[i * ldd + j] += A_PART * p_input_b[j * ldb + k];
            }
          }
        }
      } else {
        for (i = 0; i < M; ++i) {
          float A_PART;
          for (k = 0; k < K; ++k) {
            A_PART = alpha_ * p_input_a[k * lda + i];
            for (j = 0; j < N; ++j) {
              p_output_y[i * ldd + j] += A_PART * p_input_b[k * ldb + j];
            }
          }
        }
      }
    } else if (transb) {
      for (i = 0; i < M; ++i) {
        float A_PART;
        for (k = 0; k < K; ++k) {
          A_PART = alpha_ * p_input_a[i * lda + k];
          for (j = 0; j < N; ++j) {
            p_output_y[i * ldd + j] += A_PART * p_input_b[j * ldb + k];
          }
        }
      }
    } else {
      for (i = 0; i < M; ++i) {
        float A_PART;
        for (k = 0; k < K; ++k) {
          A_PART = alpha_ * p_input_a[i * lda + k];
          for (j = 0; j < N; ++j) {
            p_output_y[i * ldd + j] += A_PART * p_input_b[k * ldb + j];
          }
        }
      }
    }
  } else {
    // rowMajor_ == 0
    if (transa) {
      if (transb) {
        for (i = 0; i < M; ++i) {
          float A_PART;
          for (k = 0; k < K; ++k) {
            A_PART = alpha_ * p_input_a[i * lda + k];
            for (j = 0; j < N; ++j) {
              p_output_y[j * ldd + i] += A_PART * p_input_b[k * ldb + j];
            }
          }
        }
      } else {
        for (i = 0; i < M; ++i) {
          float A_PART;
          for (k = 0; k < K; ++k) {
            A_PART = alpha_ * p_input_a[k * lda + i];
            for (j = 0; j < N; ++j) {
              p_output_y[j * ldd + i] += A_PART * p_input_b[k * ldb + j];
            }
          }
        }
      }
    } else if (transb) {
      for (i = 0; i < M; ++i) {
        float A_PART;
        for (k = 0; k < K; ++k) {
          A_PART = alpha_ * p_input_a[i * lda + k];
          for (j = 0; j < N; ++j) {
            p_output_y[j * ldd + i] += A_PART * p_input_b[j * ldb + k];
          }
        }
      }
    } else {
      for (i = 0; i < M; ++i) {
        float A_PART;
        for (k = 0; k < K; ++k) {
          A_PART = alpha_ * p_input_a[k * lda + i];
          for (j = 0; j < N; ++j) {
            p_output_y[j * ldd + i] += A_PART * p_input_b[j * ldb + k];
          }
        }
      }
    }
  }
}

}  // namespace Cpu
