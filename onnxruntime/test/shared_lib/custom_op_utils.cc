// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "custom_op_utils.h"
#include "core/common/common.h"
#include "core/framework/ortdevice.h"
#include "core/framework/ortmemoryinfo.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
template <typename T1, typename T2, typename T3>
void cuda_add(int64_t, T3*, const T1*, const T2*, cudaStream_t compute_stream);

template <typename T>
void cuda_slice(const T*, int64_t, int64_t, T*, cudaStream_t compute_stream);
#endif

void MyCustomKernel::Compute(OrtKernelContext* context) {
  // Setup inputs
  Ort::KernelContext ctx(context);
  auto input_X = ctx.GetInput(0);
  auto input_Y = ctx.GetInput(1);
  const float* X = input_X.GetTensorData<float>();
  const float* Y = input_Y.GetTensorData<float>();

  // Setup output
  auto dimensions = input_X.GetTensorTypeAndShapeInfo().GetShape();
  auto output = ctx.GetOutput(0, dimensions);
  float* out = output.GetTensorMutableData<float>();

  auto output_info = output.GetTensorTypeAndShapeInfo();
  int64_t size = output_info.GetElementCount();

#ifdef USE_CUDA
  OrtMemoryInfo mem_info("", OrtAllocatorType::OrtDeviceAllocator, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0));
#else
  OrtMemoryInfo mem_info("", OrtAllocatorType::OrtArenaAllocator, OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, 0));
#endif
  OrtAllocator* allocator;
  Ort::ThrowOnError(ort_.KernelContext_GetAllocator(context, &mem_info, &allocator));
  void* allocated = allocator->Alloc(allocator, 2);
  EXPECT_NE(allocated, nullptr) << "KernelContext_GetAllocator() can successfully allocate some memory";
  allocator->Free(allocator, allocated);

  // Do computation
#ifdef USE_CUDA
  // Launch on stream 0 or user provided stream
  void* stream;
  Ort::ThrowOnError(ort_.KernelContext_GetGPUComputeStream(context, &stream));
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  cuda_add(size, out, X, Y, cuda_stream);
  // cudaStreamSynchronize(nullptr);
  // If everything is setup correctly, custom op implementations need not have such explicit synchronization logic as above.
  // To make sure custom kernels and ORT CUDA kernels are implicitly synchronized:
  // (1) Create your session with a compute stream passed in via SessionOptions and use the same compute
  //     stream to launch the custom op (OR)
  // (2) Use the API KernelContext_GetGPUComputeStream() to query the CUDA compute stream being used by ORT kernels in this session
  //     and use the same compute stream to launch the custom op.
  // Here, an example for (1) is shown (See test_inference.cc to see how this custom op is used.)
#else
  ORT_UNUSED_PARAMETER(ort_);
  for (int64_t i = 0; i < size; i++) {
    out[i] = X[i] + Y[i];
  }
#endif
}

#ifdef USE_CUDA
void MyCustomKernelSecondInputOnCpu::Compute(OrtKernelContext* context) {
  // Setup inputs
  Ort::KernelContext ctx(context);
  auto input_X = ctx.GetInput(0);
  auto input_Y = ctx.GetInput(1);
  const float* X = input_X.GetTensorData<float>();
  const float* Y = input_Y.GetTensorData<float>();

  // check if the second input is on CPU
  cudaPointerAttributes attributes;
  cudaPointerGetAttributes(&attributes, Y);
  auto y_mem_type = attributes.device;
  // TODO: check why the below ORT API does not work as expected:
  // `auto y_mem_type = input_Y.GetTensorMemoryInfo().GetMemoryType();`
  ASSERT_EQ(y_mem_type, OrtMemType::OrtMemTypeCPUInput);

  // copy the second input to GPU
  const int64_t y_size = input_Y.GetTensorTypeAndShapeInfo().GetElementCount();
  float* Y_cuda{};
  cudaMalloc(&Y_cuda, y_size * sizeof(float));
  cudaMemcpy(Y_cuda, Y, y_size * sizeof(float), cudaMemcpyHostToDevice);

  // Setup output
  auto dimensions = input_X.GetTensorTypeAndShapeInfo().GetShape();
  auto output = ctx.GetOutput(0, dimensions);
  float* out = output.GetTensorMutableData<float>();

  auto output_info = output.GetTensorTypeAndShapeInfo();
  int64_t size = output_info.GetElementCount();

  // Do computation

  // Launch on stream 0 or user provided stream
  cuda_add(size, out, X, Y_cuda, compute_stream_ == nullptr ? 0 : reinterpret_cast<cudaStream_t>(compute_stream_));
  // cudaStreamSynchronize(nullptr);
  // If everything is setup correctly, custom op implementations need not have such explicit synchronization logic as above.
  // To make sure custom kernels and ORT CUDA kernels are implicitly synchronized:
  // (1) Create your session with a compute stream passed in via SessionOptions and use the same compute
  //     stream to launch the custom op (OR)
  // (2) Use the API KernelContext_GetGPUComputeStream() to query the CUDA compute stream being used by ORT kernels in this session
  //     and use the same compute stream to launch the custom op.
  // Here, an example for (1) is shown (See test_inference.cc to see how this custom op is used.)
  cudaFree(Y_cuda);
}
#endif

void MyCustomKernelMultipleDynamicInputs::Compute(OrtKernelContext* context) {
  // Setup inputs
  Ort::KernelContext ctx(context);
  auto input_X = ctx.GetInput(0);
  auto input_Y = ctx.GetInput(1);
  // Even though this kernel backs an operator where-in both inputs can be any type and need not be homogeneous
  // as a proof-of-concept, support the case where-in the first input is of float type and the second input
  // is of double type. Users need to extend this logic to handle any arbitrary type should the need arise.
  const float* X = input_X.GetTensorData<float>();
  const double* Y = input_Y.GetTensorData<double>();

  // Setup output
  auto dimensions = input_X.GetTensorTypeAndShapeInfo().GetShape();
  auto output = ctx.GetOutput(0, dimensions);
  float* out = output.GetTensorMutableData<float>();

  auto output_info = output.GetTensorTypeAndShapeInfo();
  const int64_t size = output_info.GetElementCount();

  // Do computation
#ifdef USE_CUDA
  // Launch on stream 0 or user provided stream
  void* stream;
  Ort::ThrowOnError(ort_.KernelContext_GetGPUComputeStream(context, &stream));
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  cuda_add(size, out, X, Y, cuda_stream);
  // cudaStreamSynchronize(nullptr);
  // If everything is setup correctly, custom op implementations need not have such explicit synchronization logic as above.
  // To make sure custom kernels and ORT CUDA kernels are implicitly synchronized:
  // (1) Create your session with a compute stream passed in via SessionOptions and use the same compute
  //     stream to launch the custom op (OR)
  // (2) Use the API KernelContext_GetGPUComputeStream() to query the CUDA compute stream being used by ORT kernels in this session
  //     and use the same compute stream to launch the custom op.
  // Here, an example for (1) is shown (See test_inference.cc to see how this custom op is used.)
#else
  ORT_UNUSED_PARAMETER(ort_);
  for (int64_t i = 0; i < size; i++) {
    out[i] = static_cast<float>(X[i] + Y[i]);
  }
#endif
}

void MyCustomKernelWithOptionalInput::Compute(OrtKernelContext* context) {
  // Setup inputs
  Ort::KernelContext ctx(context);
  auto input_X1 = ctx.GetInput(0);
  auto input_X2 = ctx.GetInput(1);
  auto input_X3 = ctx.GetInput(2);

  const float* X1 = input_X1.GetTensorData<float>();

  // The second input may or may not be present
  const float* X2 = (input_X2 != nullptr) ? input_X2.GetTensorData<float>() : nullptr;
  const float* X3 = input_X3.GetTensorData<float>();

  // Setup output
  int64_t output_dim_value = 1;
  auto output = ctx.GetOutput(0, &output_dim_value, 1);
  float* out = output.GetTensorMutableData<float>();

  // Only CPU EP is supported in this kernel
  for (int64_t i = 0; i < output_dim_value; i++) {
    out[i] = X1[i] + (X2 != nullptr ? X2[i] : 0) + X3[i];
  }
}

void MyCustomStringLengthsKernel::Compute(OrtKernelContext* context) {
  Ort::KernelContext kcontext(context);
  constexpr std::array<const int64_t, 1> output_shape = {1};
  const size_t num_inputs = kcontext.GetInputCount();
  Ort::Logger logger = kcontext.GetLogger();

  ORT_CXX_LOGF(logger, OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE, "Getting string lengths for %d inputs",
               static_cast<int>(num_inputs));

  // Each output is set to the length of the corresponding input string.
  for (size_t i = 0; i < num_inputs; ++i) {
    auto input = kcontext.GetInput(i);
    auto output = kcontext.GetOutput(i, output_shape.data(), output_shape.size());
    int64_t* str_len_ptr = output.GetTensorMutableData<int64_t>();

    *str_len_ptr = input.GetStringTensorElementLength(0);
  }
}

void AddInputForCustomStringLengthsKernel(std::string input_str, OrtAllocator* allocator,
                                          std::vector<Ort::Value>& ort_inputs, std::vector<std::string>& input_names,
                                          std::vector<std::string>& output_names,
                                          std::vector<std::vector<int64_t>>& expected_dims,
                                          std::vector<std::vector<int64_t>>& expected_outputs) {
  const size_t input_index = ort_inputs.size();
  constexpr std::array<int64_t, 1> input_dims = {1};
  Ort::Value& ort_value = ort_inputs.emplace_back(
      Ort::Value::CreateTensor(allocator, input_dims.data(), input_dims.size(),
                               ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING));
  std::ostringstream oss(std::ostringstream::ate);

  oss.str("input_");
  oss << input_index;
  input_names.emplace_back(oss.str());

  oss.str("output_");
  oss << input_index;
  output_names.emplace_back(oss.str());

  expected_dims.push_back({1});
  expected_outputs.push_back({static_cast<int64_t>(input_str.size())});
  ort_value.FillStringTensorElement(input_str.data(), 0);
}

void MyCustomEchoReversedArgsKernel::Compute(OrtKernelContext* context) {
  Ort::KernelContext kcontext(context);
  constexpr std::array<int64_t, 1> output_shape = {1};
  const size_t num_inputs = kcontext.GetInputCount();

  for (size_t i = 0; i < num_inputs; ++i) {
    const size_t out_index = num_inputs - i - 1;
    auto input = kcontext.GetInput(i);
    auto output = kcontext.GetOutput(out_index, output_shape.data(), output_shape.size());

    auto type_shape_info = input.GetTensorTypeAndShapeInfo();
    auto elem_type = type_shape_info.GetElementType();

    // Only support STRING, INT64_T, and FLOAT
    switch (elem_type) {
      case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: {
        const size_t str_len = input.GetStringTensorElementLength(0);
        std::string str;

        str.resize(str_len);
        input.GetStringTensorElement(str.size(), 0, str.data());
        output.FillStringTensorElement(str.c_str(), 0);
        break;
      }
      case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
        int64_t* out_ptr = output.GetTensorMutableData<int64_t>();
        const int64_t* inp_ptr = input.GetTensorData<int64_t>();

        out_ptr[0] = inp_ptr[0];
        break;
      }
      case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
        float* out_ptr = output.GetTensorMutableData<float>();
        const float* inp_ptr = input.GetTensorData<float>();

        out_ptr[0] = inp_ptr[0];
        break;
      }
      default:
        ORT_CXX_API_THROW("MyCustomEchoReversedArgsKernel only supports tensor inputs of type STRING, INT64_T, and FLOAT",
                          OrtErrorCode::ORT_INVALID_GRAPH);
    }
  }
}

void MyCustomKernelWithAttributes::Compute(OrtKernelContext* context) {
  // Setup inputs
  Ort::KernelContext ctx(context);
  auto input_X = ctx.GetInput(0);
  const float* X = input_X.GetTensorData<float>();

  // Setup output
  auto dimensions = input_X.GetTensorTypeAndShapeInfo().GetShape();
  auto output = ctx.GetOutput(0, dimensions);
  float* out = output.GetTensorMutableData<float>();

  const int64_t size = output.GetTensorTypeAndShapeInfo().GetElementCount();

  // This kernel only supports CPU EP
  if (string_arr_ == "add") {  // Test that the string attribute parsing went correctly
    for (int64_t i = 0; i < size; i++) {
      out[i] = X[i] +
               float_attr_ + static_cast<float>(int_attr_) +
               floats_attr_[0] + floats_attr_[1] +
               static_cast<float>(ints_attr_[0]) + static_cast<float>(ints_attr_[1]);
    }
  } else {  // if the string attribute parsing had not gone correctly - it will trigger this path and fail the test due to result mis-match
    for (int64_t i = 0; i < size; i++) {
      out[i] = 0.f;
    }
  }
}

template <typename T>
static void custom_slice(const T* X, int64_t from, int64_t to, T* Y, void* compute_stream) {
#ifdef USE_CUDA
  // We do not expect the compute_stream to be nullptr as we have queried the compute stream
  // being used by the ORT session. If it is a nullptr, there was an issue with the stream querying.
  // We don't launch the operation to trigger a test failure in that case.
  if (compute_stream) {
    cuda_slice(X, from, to, Y, reinterpret_cast<cudaStream_t>(compute_stream));
  }
#else
  ORT_UNUSED_PARAMETER(compute_stream);
  for (auto i = from; i < to; i++) {
    Y[i - from] = X[i];
  }
#endif
}

void SliceCustomOpKernel::Compute(OrtKernelContext* context) {
  // Setup inputs and outputs
  Ort::KernelContext ctx(context);
  auto input_X = ctx.GetInput(0);
  auto input_from = ctx.GetInput(1);
  auto input_to = ctx.GetInput(2);

  ONNXTensorElementDataType input_X_type = input_X.GetTensorTypeAndShapeInfo().GetElementType();

#if USE_CUDA
  int64_t slice_from = 0;
  int64_t slice_to = 0;
  cudaMemcpy(&slice_from, input_from.GetTensorData<int64_t>(), sizeof(int64_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&slice_to, input_to.GetTensorData<int64_t>(), sizeof(int64_t), cudaMemcpyDeviceToHost);
#else
  int64_t slice_from = *input_from.GetTensorData<int64_t>();
  int64_t slice_to = *input_to.GetTensorData<int64_t>();
#endif
  std::vector<int64_t> output_dims = {slice_to - slice_from};
  auto output = ctx.GetOutput(0, output_dims);
  // do slice
  switch (input_X_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:

      custom_slice(input_X.GetTensorData<float>(), slice_from, slice_to,
                   output.GetTensorMutableData<float>(), ctx.GetGPUComputeStream());
      // cudaStreamSynchronize(nullptr);
      // If everything is setup correctly, custom op implementations need not have such explicit synchronization logic as above.
      // To make sure custom kernels and ORT CUDA kernels are implicitly synchronized:
      // (1) Create your session with a compute stream passed in via SessionOptions and use the same compute
      //     stream to launch the custom op (OR)
      // (2) Use the API KernelContext_GetGPUComputeStream() to query the CUDA compute stream being used by ORT kernels in this session
      //     and use the same compute stream to launch the custom op.
      // Here, an example for (2) is shown (See test_inference.cc to see how this custom op is used.)
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      custom_slice(input_X.GetTensorData<double>(), slice_from, slice_to,
                   output.GetTensorMutableData<double>(), ctx.GetGPUComputeStream());
      // cudaStreamSynchronize(nullptr);
      // If everything is setup correctly, custom op implementations need not have such explicit synchronization logic as above.
      // To make sure custom kernels and ORT CUDA kernels are implicitly synchronized:
      // (1) Create your session with a compute stream passed in via SessionOptions and use the same compute
      //     stream to launch the custom op (OR)
      // (2) Use the API KernelContext_GetGPUComputeStream() to query the CUDA compute stream being used by ORT kernels in this session
      //     and use the same compute stream to launch the custom op.
      // Here, an example for (2) is shown (See test_inference.cc to see how this custom op is used.)
      break;
    default:
      ORT_THROW("Unsupported input type");
  }
}

StandaloneCustomKernel::StandaloneCustomKernel(const OrtKernelInfo* k_info) {
  Ort::ConstKernelInfo info{k_info};
  info_copy_ = info.Copy();

  const char* add_type_constraint_names[1] = {"T"};
  ONNXTensorElementDataType add_type_constraint_values[1] = {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};
  op_add_ = Ort::Op::Create(info_copy_, "Add", "", /* must match onnx version number exactly */ 14,
                            add_type_constraint_names,
                            add_type_constraint_values,
                            1, nullptr, 0, 2, 1);

#if !defined(REDUCED_OPS_BUILD)
  InitTopK();
  InitGru();
#endif
}

#if !defined(REDUCED_OPS_BUILD)
void StandaloneCustomKernel::InitTopK() {
  const char* type_constraint_names[2] = {"T", "I"};
  ONNXTensorElementDataType type_constraint_values[2] = {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64};

  constexpr int64_t axis_value = -1;
  auto axis = Ort::OpAttr("axis", &axis_value, 1, OrtOpAttrType::ORT_OP_ATTR_INT);

  constexpr int64_t largest_value = 0;  // return in ascending order
  auto largest = Ort::OpAttr("largest", &largest_value, 1, OrtOpAttrType::ORT_OP_ATTR_INT);

  constexpr int64_t sorted_value = 1;
  auto sorted = Ort::OpAttr("sorted", &sorted_value, 1, OrtOpAttrType::ORT_OP_ATTR_INT);

  Ort::OpAttr top_attrs[3] = {std::move(axis), std::move(largest), std::move(sorted)};
  op_topk_ = Ort::Op::Create(info_copy_, "TopK", "", /* must match onnx version number exactly */ 11,
                             type_constraint_names,
                             type_constraint_values,
                             2, top_attrs, 3, 2, 2);
}

void StandaloneCustomKernel::InvokeTopK(OrtKernelContext* context) {
  auto mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeCPU);

  float raw_x[10] = {6., 3., 4., 8., 7., 1., 9., 0., 5., 2.};
  int64_t raw_x_shape[1] = {10};
  auto topk_x = Ort::Value::CreateTensor(mem_info, raw_x, 10, raw_x_shape, 1);

  int64_t raw_k[1] = {2};
  int64_t raw_k_shape[1] = {1};
  auto topk_k = Ort::Value::CreateTensor(mem_info, raw_k, 1, raw_k_shape, 1);

  float raw_values[2] = {};
  int64_t raw_values_shape[1] = {2};
  auto topk_values = Ort::Value::CreateTensor(mem_info, raw_values, 2, raw_values_shape, 1);

  int64_t raw_indices[2] = {};
  int64_t raw_indices_shape[1] = {2};
  auto topk_indices = Ort::Value::CreateTensor(mem_info, raw_indices, 2, raw_indices_shape, 1);

  const Ort::Value topk_inputs[2] = {std::move(topk_x), std::move(topk_k)};
  Ort::Value topk_outputs[2] = {std::move(topk_values), std::move(topk_indices)};
  op_topk_.Invoke(context, topk_inputs, 2, topk_outputs, 2);

  if (std::abs(raw_values[0] - 0.) > 1e-6 || std::abs(raw_values[1] - 1.) > 1e-6) {
    ORT_THROW("topk instant operator returns wrong values");
  }
  if (raw_indices[0] != 7 || raw_indices[1] != 5) {
    ORT_THROW("topk instant operator returns wrong indices");
  }
}

void StandaloneCustomKernel::InitGru() {
  const char* type_constraint_names[2] = {"T", "T1"};
  ONNXTensorElementDataType type_constraint_values[2] = {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32};

  const char* activition_names[4] = {"LeakyRelu", "Tanh", "Sigmoid", "ScaledTanh"};
  Ort::OpAttr activations = Ort::OpAttr("activations", activition_names, 4, OrtOpAttrType::ORT_OP_ATTR_STRINGS);

  float alphas[2] = {0.5f, 2.f};
  Ort::OpAttr activation_alpha = Ort::OpAttr("activation_alpha ", alphas, 2, OrtOpAttrType::ORT_OP_ATTR_FLOATS);

  float betas[1] = {2.f};
  Ort::OpAttr activation_beta = Ort::OpAttr("activation_beta ", betas, 1, OrtOpAttrType::ORT_OP_ATTR_FLOATS);

  const char* direction_string = "bidirectional";
  Ort::OpAttr direction = Ort::OpAttr("direction", direction_string, 1, OrtOpAttrType::ORT_OP_ATTR_STRING);

  int64_t linear_before_reset_value = 0;
  Ort::OpAttr linear_before_reset = Ort::OpAttr("linear_before_reset", &linear_before_reset_value, 1,
                                                OrtOpAttrType::ORT_OP_ATTR_INT);

  int64_t hidden_size_value = 2;
  Ort::OpAttr hidden_size = Ort::OpAttr("hidden_size", &hidden_size_value, 1, OrtOpAttrType::ORT_OP_ATTR_INT);

  // can push_back to vector as well
  const Ort::OpAttr gru_attrs[6] = {std::move(activations), std::move(activation_alpha),
                                    std::move(activation_beta), std::move(direction),
                                    std::move(linear_before_reset), std::move(hidden_size)};

  op_gru_ = Ort::Op::Create(info_copy_, "GRU", "", /* must match onnx version number exactly */ 14,
                            type_constraint_names,
                            type_constraint_values,
                            2, gru_attrs, 6, 6, 2);
}

void StandaloneCustomKernel::InvokeGru(OrtKernelContext* context) {
  auto mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeCPU);

  float raw_x[2] = {1.0f, 2.0f};
  int64_t raw_x_shape[3] = {1, 1, 2};
  auto X = Ort::Value::CreateTensor(mem_info, raw_x, 2, raw_x_shape, 3);

  float raw_w[24] = {
      -0.494659f, 0.0453352f, -0.487793f, 0.417264f,    // Wz
      -0.0091708f, -0.255364f, -0.106952f, -0.266717f,  // Wr
      -0.0888852f, -0.428709f, -0.283349f, 0.208792f,   // Wh
      -0.494659f, 0.0453352f, -0.487793f, 0.417264f,    // WBz
      -0.0091708f, -0.255364f, -0.106952f, -0.266717f,  // WBr
      -0.0888852f, -0.428709f, -0.283349f, 0.208792f    // WBh
  };
  int64_t raw_w_shape[3] = {2, 6, 2};
  auto W = Ort::Value::CreateTensor(mem_info, raw_w, 24, raw_w_shape, 3);

  float raw_r[24] = {
      0.146626f, -0.0620289f, -0.0815302f, 0.100482f,  // Rz
      -0.228172f, 0.405972f, 0.31576f, 0.281487f,      // Rr
      -0.394864f, 0.42111f, -0.386624f, -0.390225f,    // Rh
      0.146626f, -0.0620289f, -0.0815302f, 0.100482f,  // RBz
      -0.228172f, 0.405972f, 0.31576f, 0.281487f,      // RBr
      -0.394864f, 0.42111f, -0.386624f, -0.390225f};   // RBh
  int64_t raw_r_shape[3] = {2, 6, 2};
  auto R = Ort::Value::CreateTensor(mem_info, raw_r, 24, raw_r_shape, 3);

  float raw_b[24] = {
      0.381619f, 0.0323954f,   // Wbz
      -0.258721f, 0.45056f,    // Wbr
      -0.250755f, 0.0967895f,  // Wbh
      0.0f, 0.0f,              // Rbz
      -0.0f, 0.0f,             // Rbr
      -0.0f, 0.0f,             // Rbh
      0.381619f, 0.0323954f,   // WBbz
      -0.258721f, 0.45056f,    // WBbr
      -0.250755f, 0.0967895f,  // WBbh
      0.0f, 0.0f,              // RBbz
      -0.0f, 0.0f,             // RBbr
      -0.0f, 0.0f};            // RBbh
  int64_t raw_b_shape[2] = {2, 12};
  auto B = Ort::Value::CreateTensor(mem_info, raw_b, 24, raw_b_shape, 2);

  int32_t raw_seq_lens = 1;
  int64_t seq_lens_shape[1] = {1};
  auto sequence_lens = Ort::Value::CreateTensor(mem_info, &raw_seq_lens, 1, seq_lens_shape, 1);

  std::vector<float> raw_initial_h(4, 0.25f);
  int64_t initial_h_shape[3] = {2, 1, 2};
  auto initial_h = Ort::Value::CreateTensor(mem_info, raw_initial_h.data(), 4, initial_h_shape, 3);

  float raw_y[4] = {};
  int64_t raw_y_shape[64] = {1, 2, 1, 2};
  auto Y = Ort::Value::CreateTensor(mem_info, raw_y, 4, raw_y_shape, 4);

  float raw_yh[4] = {};
  int64_t raw_yh_shape[64] = {2, 1, 2};
  auto YH = Ort::Value::CreateTensor(mem_info, raw_yh, 4, raw_yh_shape, 3);

  const Ort::Value inputs[6] = {std::move(X), std::move(W), std::move(R), std::move(B),
                                std::move(sequence_lens), std::move(initial_h)};
  Ort::Value outputs[2] = {std::move(Y), std::move(YH)};

  const float expected_y[4] = {-0.832559f,
                               0.236267f,
                               0.124924f,
                               0.148701f};

  op_gru_.Invoke(context, inputs, 6, outputs, 2);

  for (int i = 0; i < 4; ++i) {
    if (std::abs(raw_y[i] - expected_y[i]) > 1e-6) {
      ORT_THROW("GRU op give unexpected output.");
    }
  }
}

void StandaloneCustomKernel::InitInvokeConv(OrtKernelContext* context) {
  auto mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeCPU);
  const char* type_constraint_names[] = {"T"};
  ONNXTensorElementDataType type_constraint_values[] = {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};  // float

  int64_t dilation_values[] = {2};
  Ort::OpAttr dilations = Ort::OpAttr("dilations", &dilation_values, 1, OrtOpAttrType::ORT_OP_ATTR_INTS);

  int64_t group_value = 1;
  Ort::OpAttr group = Ort::OpAttr("group", &group_value, 1, OrtOpAttrType::ORT_OP_ATTR_INT);

  int64_t kernel_shape_values[] = {2};
  Ort::OpAttr kernel_shape = Ort::OpAttr("kernel_shape", &kernel_shape_values, 1, OrtOpAttrType::ORT_OP_ATTR_INTS);

  int64_t pad_values[] = {2, 2};
  Ort::OpAttr pads = Ort::OpAttr("pads", &pad_values, 2, OrtOpAttrType::ORT_OP_ATTR_INTS);

  int64_t stride_values[] = {2};
  Ort::OpAttr strides = Ort::OpAttr("strides", &stride_values, 1, OrtOpAttrType::ORT_OP_ATTR_INTS);

  const Ort::OpAttr conv_attrs[] = {std::move(dilations), std::move(group), std::move(kernel_shape),
                                    std::move(pads), std::move(strides)};
  auto op_conv = Ort::Op::Create(info_copy_, "Conv", "", 11,
                                 type_constraint_names,
                                 type_constraint_values,
                                 1, conv_attrs, 5, 2, 1);

  std::vector<int64_t> X_shape = {3, 1, 8};
  std::vector<float> X_value = {0.11094123125076294f, -0.0038032233715057373f, 0.3896123170852661f, 0.33259105682373047f,
                                0.02794349193572998f, -0.08360505104064941f, -0.4100455045700073f, -0.09502679109573364f,
                                -0.11361867189407349f, -0.025495320558547974f, 0.3696536421775818f, 0.3529144525527954f,
                                -0.34991076588630676f, -0.22024285793304443f, 0.23085933923721313f, -0.4575521945953369f,
                                -0.17685726284980774f, -0.06030535697937012f, -0.3996139168739319f, -0.19385704398155212f,
                                -0.10454908013343811f, -0.14503943920135498f, -0.31941986083984375f, -0.15372398495674133f};

  auto X = Ort::Value::CreateTensor(mem_info, reinterpret_cast<float*>(X_value.data()), X_value.size(), reinterpret_cast<int64_t*>(X_shape.data()), X_shape.size());

  std::vector<int64_t> W_shape = {2, 1, 2};
  std::vector<float> W_value = {0.13225573301315308f, 0.09750443696975708f, 0.3469849228858948f, 0.4743430018424988f};
  auto W = Ort::Value::CreateTensor(mem_info, reinterpret_cast<float*>(W_value.data()), W_value.size(), reinterpret_cast<int64_t*>(W_shape.data()), W_shape.size());

  std::vector<int64_t> Y_shape = {3, 2, 5};
  float Y_values[3 * 2 * 5] = {};
  auto Y = Ort::Value::CreateTensor(mem_info, Y_values, 3 * 2 * 5, reinterpret_cast<int64_t*>(Y_shape.data()), Y_shape.size());

  const Ort::Value inputs[] = {std::move(X), std::move(W)};
  Ort::Value outputs[] = {std::move(Y)};

  op_conv.Invoke(context, inputs, 2, outputs, 1);

  float Y_expected[] = {0.010817262344062328f, 0.05266154557466507f, 0.054253075271844864f, -0.03628557175397873f,
                        -0.05423086881637573f, 0.05262419581413269f, 0.22330480813980103f, 0.14844439923763275f,
                        -0.1848062425851822f, -0.14227961003780365f, -0.011078324168920517f, 0.02101614698767662f,
                        0.014770962297916412f, -0.023767895996570587f, 0.03053247183561325f, -0.053894221782684326f,
                        0.13591864705085754f, -0.03771348297595978f, -0.011907249689102173f, 0.08010470867156982f,
                        -0.01724436692893505f, -0.06235451623797417f, -0.06304522603750229f, -0.044972069561481476f,
                        -0.042245108634233475f, -0.08389100432395935f, -0.2509208619594574f, -0.18825212121009827f,
                        -0.18779152631759644f, -0.11083387583494186f};

  for (int i = 0; i < 3 * 2 * 5; ++i) {
    if (std::abs(Y_values[i] - Y_expected[i]) > 1e-6) {
      ORT_THROW("Conv op give unexpected output.");
    }
  }
}

#endif  // !defined(REDUCED_OPS_BUILD)

void StandaloneCustomKernel::Compute(OrtKernelContext* context) {
  Ort::KernelContext ctx(context);
  auto input_X = ctx.GetInput(0);
  auto input_Y = ctx.GetInput(1);

  auto dimensions = input_X.GetTensorTypeAndShapeInfo().GetShape();
  auto output = ctx.GetOutput(0, dimensions);

  const OrtValue* inputs[2] = {input_X, input_Y};
  OrtValue* outputs[1] = {output};

  op_add_.Invoke(context, inputs, 2, outputs, 1);

#if !defined(USE_CUDA) && !defined(REDUCED_OPS_BUILD)
  InvokeTopK(context);
  InvokeGru(context);
  InitInvokeConv(context);
#endif
}

StandaloneCustomKernel::~StandaloneCustomKernel() {
}
