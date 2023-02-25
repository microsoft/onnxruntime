// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"

#include "src/fastertransformer/models/vit_int8/ViTINT8.h"
#include "src/fastertransformer/utils/nvtx_utils.h"

using namespace fastertransformer;

struct FTViTINT8CustomKernel {
  FTViTINT8CustomKernel(const OrtKernelInfo*, void* compute_stream);

  void Compute(OrtKernelContext* context);

  void Release();

 private:
  cudnnHandle_t    cudnn_handle_;
  cublasHandle_t   cublas_handle_;
  cublasLtHandle_t cublaslt_handle_;
  cudaStream_t     stream_ = 0;
  cublasINT8MMWrapper* cublas_wrapper_;
  std::mutex* cublas_wrapper_mutex_;
  AttentionType attention_type_;
  ViTINT8Weight<float> params_;
  ViTTransformerINT8<float>* vit_;
  cublasAlgoMap* cublas_algo_map_;

  void* compute_stream_;
};

struct FTViTINT8CustomOp : Ort::CustomOpBase<FTViTINT8CustomOp, FTViTINT8CustomKernel> {
  explicit FTViTINT8CustomOp(const char* provider, void* compute_stream) : provider_(provider), compute_stream_(compute_stream) {}

  void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* info) const { return new FTViTINT8CustomKernel(info, compute_stream_); };

  const char* GetName() const { return name_; };

  void SetName(const char* name) { name_ = name; };

  const char* GetExecutionProviderType() const { return provider_; };

  size_t GetInputTypeCount() const { return num_inputs_; };

  void SetInputTypeCount(size_t num) { num_inputs_ = num; };

  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED; };

  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t) const { return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC; };   

  size_t GetOutputTypeCount() const { return num_outputs_; };

  void SetOutputTypeCount(size_t num) { num_outputs_ = num; };

  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED; };

  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t) const { return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC; };

 private:
  const char* provider_;
  void* compute_stream_;
  const char* name_{"FTViTINT8"};
  size_t num_inputs_ = 1;  // set to 1 to match with default min_arity for variadic input  
  size_t num_outputs_ = 1; // set to 1 to match with default min_arity for variadic output 
};
