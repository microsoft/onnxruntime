// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "custom_op_utils.h"
#include "core/common/common.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
template <typename T1, typename T2, typename T3>
void cuda_add(int64_t, T3*, const T1*, const T2*, cudaStream_t compute_stream);

template <typename T>
void cuda_slice(const T*, int64_t, int64_t, T*, cudaStream_t compute_stream);
#endif

void MyCustomKernel::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const OrtValue* input_Y = ort_.KernelContext_GetInput(context, 1);
  const float* X = ort_.GetTensorData<float>(input_X);
  const float* Y = ort_.GetTensorData<float>(input_Y);

  // Setup output
  OrtTensorDimensions dimensions(ort_, input_X);
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
  float* out = ort_.GetTensorMutableData<float>(output);

  OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  int64_t size = ort_.GetTensorShapeElementCount(output_info);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);

  // Do computation
#ifdef USE_CUDA
  // Launch on stream 0 or user provided stream
  cuda_add(size, out, X, Y, compute_stream_ == nullptr ? 0 : reinterpret_cast<cudaStream_t>(compute_stream_));
  // cudaStreamSynchronize(nullptr);
  // If everything is setup correctly, custom op implementations need not have such explicit synchronization logic as above.
  // To make sure custom kernels and ORT CUDA kernels are implicitly synchronized:
  // (1) Create your session with a compute stream passed in via SessionOptions and use the same compute
  //     stream to launch the custom op (OR)
  // (2) Use the API KernelContext_GetGPUComputeStream() to query the CUDA compute stream being used by ORT kernels in this session
  //     and use the same compute stream to launch the custom op.
  // Here, an example for (1) is shown (See test_inference.cc to see how this custom op is used.)
#else
  ORT_UNUSED_PARAMETER(compute_stream_);
  for (int64_t i = 0; i < size; i++) {
    out[i] = X[i] + Y[i];
  }
#endif
}

void MyCustomKernelMultipleDynamicInputs::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const OrtValue* input_Y = ort_.KernelContext_GetInput(context, 1);
  // Even though this kernel backs an operator where-in both inputs can be any type and need not be homogeneous
  // as a proof-of-concept, support the case where-in the first input is of float type and the second input
  // is of double type. Users need to extend this logic to handle any arbitrary type should the need arise.
  const float* X = ort_.GetTensorData<float>(input_X);
  const double* Y = ort_.GetTensorData<double>(input_Y);

  // Setup output
  OrtTensorDimensions dimensions(ort_, input_X);
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
  float* out = ort_.GetTensorMutableData<float>(output);

  OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  int64_t size = ort_.GetTensorShapeElementCount(output_info);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);

  // Do computation
#ifdef USE_CUDA
  // Launch on stream 0 or user provided stream
  cuda_add(size, out, X, Y, compute_stream_ == nullptr ? 0 : reinterpret_cast<cudaStream_t>(compute_stream_));
  // cudaStreamSynchronize(nullptr);
  // If everything is setup correctly, custom op implementations need not have such explicit synchronization logic as above.
  // To make sure custom kernels and ORT CUDA kernels are implicitly synchronized:
  // (1) Create your session with a compute stream passed in via SessionOptions and use the same compute
  //     stream to launch the custom op (OR)
  // (2) Use the API KernelContext_GetGPUComputeStream() to query the CUDA compute stream being used by ORT kernels in this session
  //     and use the same compute stream to launch the custom op.
  // Here, an example for (1) is shown (See test_inference.cc to see how this custom op is used.)
#else
  ORT_UNUSED_PARAMETER(compute_stream_);
  for (int64_t i = 0; i < size; i++) {
    out[i] = static_cast<float>(X[i] + Y[i]);
  }
#endif
}

void MyCustomKernelWithOptionalInput::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_X1 = ort_.KernelContext_GetInput(context, 0);
  const OrtValue* input_X2 = ort_.KernelContext_GetInput(context, 1);
  const OrtValue* input_X3 = ort_.KernelContext_GetInput(context, 2);

  const float* X1 = ort_.GetTensorData<float>(input_X1);
  // The second input may or may not be present
  const float* X2 = (input_X2 != nullptr) ? ort_.GetTensorData<float>(input_X2) : nullptr;
  const float* X3 = ort_.GetTensorData<float>(input_X3);

  // Setup output
  int64_t output_dim_value = 1;
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, &output_dim_value, 1);
  float* out = ort_.GetTensorMutableData<float>(output);

  // Only CPU EP is supported in this kernel
  for (int64_t i = 0; i < output_dim_value; i++) {
    out[i] = X1[i] + (X2 != nullptr ? X2[i] : 0) + X3[i];
  }
}

void MyCustomKernelWithAttributes::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const float* X = ort_.GetTensorData<float>(input_X);

  // Setup output
  OrtTensorDimensions dimensions(ort_, input_X);
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
  float* out = ort_.GetTensorMutableData<float>(output);

  OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  int64_t size = ort_.GetTensorShapeElementCount(output_info);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);

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
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const OrtValue* input_from = ort_.KernelContext_GetInput(context, 1);
  const OrtValue* input_to = ort_.KernelContext_GetInput(context, 2);
  OrtTensorTypeAndShapeInfo* input_X_info = ort_.GetTensorTypeAndShape(input_X);
  ONNXTensorElementDataType input_X_type = ort_.GetTensorElementType(input_X_info);
  ort_.ReleaseTensorTypeAndShapeInfo(input_X_info);
#if USE_CUDA
  int64_t slice_from = 0;
  int64_t slice_to = 0;
  cudaMemcpy(&slice_from, ort_.GetTensorData<int64_t>(input_from), sizeof(int64_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&slice_to, ort_.GetTensorData<int64_t>(input_to), sizeof(int64_t), cudaMemcpyDeviceToHost);
#else
  int64_t slice_from = *ort_.GetTensorData<int64_t>(input_from);
  int64_t slice_to = *ort_.GetTensorData<int64_t>(input_to);
#endif
  std::vector<int64_t> output_dims = {slice_to - slice_from};
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, output_dims.data(), output_dims.size());
  // do slice
  switch (input_X_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:

      custom_slice(ort_.GetTensorData<float>(input_X), slice_from, slice_to,
                   ort_.GetTensorMutableData<float>(output), ort_.KernelContext_GetGPUComputeStream(context));
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
      custom_slice(ort_.GetTensorData<double>(input_X), slice_from, slice_to,
                   ort_.GetTensorMutableData<double>(output), ort_.KernelContext_GetGPUComputeStream(context));
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

StandaloneCustomKernel::StandaloneCustomKernel(Ort::CustomOpApi ort, const OrtKernelInfo* info, void*) : ort_(ort) {
  info_copy_ = ort.CopyKernelInfo(info);
  ORT_ENFORCE(info_copy_, "Failed to copy kernel info");
  const char* add_type_constraint_names[1] = {"T"};
  int add_type_constraint_values[1] = {1};
  op_add_ = ort.CreateOp(info_copy_, "Add", "", 14,
                         (const char**)add_type_constraint_names,
                         (const ONNXTensorElementDataType*)add_type_constraint_values,
                         1, nullptr, 0, 2, 1);
  ORT_ENFORCE(op_add_, "op_add not initialzied");
  InitTopK(ort_);
  ORT_ENFORCE(op_topk_, "op_topk not initialzied");
  InitGru(ort_);
  ORT_ENFORCE(op_gru_, "op_gru not initialzied");
}

void StandaloneCustomKernel::InitTopK(Ort::CustomOpApi ort) {
  const char* type_constraint_names[2] = {"T", "I"};
  int type_constraint_values[2] = {1, 7};

  int64_t axis_value = -1;
  OrtOpAttr* axis = ort.CreateOpAttr("axis", &axis_value, 1, OrtOpAttrType::ORT_OP_ATTR_INT);

  int64_t largest_value = 0;  // return in ascending order
  OrtOpAttr* largest = ort.CreateOpAttr("largest", &largest_value, 1, OrtOpAttrType::ORT_OP_ATTR_INT);

  int64_t sorted_value = 1;
  OrtOpAttr* sorted = ort.CreateOpAttr("sorted", &sorted_value, 1, OrtOpAttrType::ORT_OP_ATTR_INT);

  if (!axis || !largest || !sorted) {
    ORT_THROW("Failed to create attributes for topk");
  }

  OrtOpAttr* top_attrs[3] = {axis, largest, sorted};
  op_topk_ = ort.CreateOp(info_copy_, "TopK", "", 14,
                          (const char**)type_constraint_names,
                          (const ONNXTensorElementDataType*)type_constraint_values,
                          2, top_attrs, 3, 2, 2);

  ort.ReleaseOpAttr(axis);
  ort.ReleaseOpAttr(largest);
  ort.ReleaseOpAttr(sorted);
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

  const OrtValue* topk_inputs[2] = {(OrtValue*)topk_x, (OrtValue*)topk_k};
  OrtValue* topk_outputs[2] = {(OrtValue*)topk_values, (OrtValue*)topk_indices};
  ort_.InvokeOp(context, op_topk_, topk_inputs, 2, topk_outputs, 2);

  if (std::abs(raw_values[0] - 0.) > 1e-6 || std::abs(raw_values[1] - 1.) > 1e-6) {
    ORT_THROW("topk instant operator returns wrong values");
  }
  if (raw_indices[0] != 7 || raw_indices[1] != 5) {
    ORT_THROW("topk instant operator returns wrong indices");
  }
}

void StandaloneCustomKernel::InitGru(Ort::CustomOpApi ort) {
  const char* type_constraint_names[2] = {"T", "T1"};
  int type_constraint_values[2] = {1, 6};

  const char* activition_names[4] = {"LeakyRelu", "Tanh", "Sigmoid", "ScaledTanh"};
  OrtOpAttr* activations = ort.CreateOpAttr("activations", activition_names, 4, OrtOpAttrType::ORT_OP_ATTR_STRINGS);

  float alphas[2] = {0.5f, 2.f};
  OrtOpAttr* activation_alpha = ort.CreateOpAttr("activation_alpha ", alphas, 2, OrtOpAttrType::ORT_OP_ATTR_FLOATS);

  float betas[1] = {2.f};
  OrtOpAttr* activation_beta = ort.CreateOpAttr("activation_beta ", betas, 1, OrtOpAttrType::ORT_OP_ATTR_FLOATS);

  const char* direction_string = "bidirectional";
  OrtOpAttr* direction = ort.CreateOpAttr("direction", direction_string, 1, OrtOpAttrType::ORT_OP_ATTR_STRING);

  int64_t linear_before_reset_value = 0;
  OrtOpAttr* linear_before_reset = ort.CreateOpAttr("linear_before_reset", &linear_before_reset_value, 1, OrtOpAttrType::ORT_OP_ATTR_INT);

  int64_t hidden_size_value = 2;
  OrtOpAttr* hidden_size = ort.CreateOpAttr("hidden_size", &hidden_size_value, 1, OrtOpAttrType::ORT_OP_ATTR_INT);

  if (!activations || !activation_alpha || !activation_beta || !direction || !linear_before_reset || !hidden_size) {
    ORT_THROW("failed to create attributes for gru.");
  }

  OrtOpAttr* gru_attrs[6] = {activations, activation_alpha, activation_beta, direction, linear_before_reset, hidden_size};
  op_gru_ = ort.CreateOp(info_copy_, "GRU", "", 14,
                         (const char**)type_constraint_names,
                         (const ONNXTensorElementDataType*)type_constraint_values,
                         2, gru_attrs, 6, 6, 2);

  ort.ReleaseOpAttr(activations);
  ort.ReleaseOpAttr(activation_alpha);
  ort.ReleaseOpAttr(activation_beta);
  ort.ReleaseOpAttr(direction);
  ort.ReleaseOpAttr(linear_before_reset);
  ort.ReleaseOpAttr(hidden_size);
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

  const OrtValue* inputs[6] = {(OrtValue*)X, (OrtValue*)W, (OrtValue*)R, (OrtValue*)B, (OrtValue*)sequence_lens, (OrtValue*)initial_h};
  OrtValue* outputs[2] = {(OrtValue*)Y, (OrtValue*)YH};

  const float expected_y[4] = {-0.832559f,
                               0.236267f,
                               0.124924f,
                               0.148701f};

  ort_.InvokeOp(context, op_gru_, inputs, 6, outputs, 2);

  for (int i = 0; i < 4; ++i) {
    if (std::abs(raw_y[i] - expected_y[i]) > 1e-6) {
      ORT_THROW("GRU op give unexpected output.");
    }
  }
}

void StandaloneCustomKernel::InitInvokeConv(OrtKernelContext* context) {
  auto mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeCPU);
  const char* type_constraint_names[] = {"T"};
  int type_constraint_values[] = {1}; //float

  int64_t dilation_values[] = {2};
  OrtOpAttr* dilations = ort_.CreateOpAttr("dilations", &dilation_values, 1, OrtOpAttrType::ORT_OP_ATTR_INTS);
  
  int64_t group_value = 1;
  OrtOpAttr* group = ort_.CreateOpAttr("group", &group_value, 1, OrtOpAttrType::ORT_OP_ATTR_INT);

  int64_t kernel_shape_values[] = {2};
  OrtOpAttr* kernel_shape = ort_.CreateOpAttr("kernel_shape", &kernel_shape_values, 1, OrtOpAttrType::ORT_OP_ATTR_INTS);

  int64_t pad_values[] = {2, 2};
  OrtOpAttr* pads = ort_.CreateOpAttr("pads", &pad_values, 2, OrtOpAttrType::ORT_OP_ATTR_INTS);

  int64_t stride_values[] = {2};
  OrtOpAttr* strides = ort_.CreateOpAttr("strides", &stride_values, 1, OrtOpAttrType::ORT_OP_ATTR_INTS);

  if (!dilations || !group || !kernel_shape || !pads || !strides) {
    ORT_THROW("failed to create attributes for conv.");
  }

  OrtOp* op_conv{};
  OrtOpAttr* conv_attrs[] = {dilations, group, kernel_shape, pads, strides};
  op_conv = ort_.CreateOp(info_copy_, "Conv", "", 11,
                          (const char**)type_constraint_names,
                          (const ONNXTensorElementDataType*)type_constraint_values,
                          1, conv_attrs, 5, 2, 1);
  ORT_ENFORCE(op_conv, "op_conv not initialzied");

  ort_.ReleaseOpAttr(dilations);
  ort_.ReleaseOpAttr(group);
  ort_.ReleaseOpAttr(kernel_shape);
  ort_.ReleaseOpAttr(pads);
  ort_.ReleaseOpAttr(strides);

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

  const OrtValue* inputs[] = {(OrtValue*)X, (OrtValue*)W};
  OrtValue* outputs[] = {(OrtValue*)Y};

  ort_.InvokeOp(context, op_conv, inputs, 2, outputs, 1);

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

  ort_.ReleaseOp(op_conv);
}

void StandaloneCustomKernel::Compute(OrtKernelContext* context) {
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const OrtValue* input_Y = ort_.KernelContext_GetInput(context, 1);
  OrtTensorDimensions dimensions(ort_, input_X);
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
  const OrtValue* inputs[2] = {input_X, input_Y};
  OrtValue* outputs[1] = {output};
  ort_.InvokeOp(context, op_add_, inputs, 2, outputs, 1);
#ifndef USE_CUDA
  InvokeTopK(context);
  InvokeGru(context);
  InitInvokeConv(context);
#endif
}

StandaloneCustomKernel::~StandaloneCustomKernel() {
  ort_.ReleaseOp(op_add_);
  ort_.ReleaseOp(op_topk_);
  ort_.ReleaseOp(op_gru_);
  ort_.ReleaseKernelInfo(info_copy_);
}
