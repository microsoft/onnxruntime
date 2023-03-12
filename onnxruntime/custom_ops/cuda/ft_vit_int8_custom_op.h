// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"

#include "src/fastertransformer/models/vit_int8/ViTINT8.h"
#include "src/fastertransformer/utils/nvtx_utils.h"

using namespace fastertransformer;

struct FTViTINT8CustomKernel {
  FTViTINT8CustomKernel(const OrtKernelInfo* info,
                        void* compute_stream,
                        int batch_size,
                        int img_size,
                        int patch_size,
                        int embed_dim,
                        int head_num,
                        int layer_num,
                        int has_cls_token,
                        int is_fp16,
                        int int8_mode);

  void Compute(OrtKernelContext* context);

  ~FTViTINT8CustomKernel();

 private:
  cudnnHandle_t    cudnn_handle_;
  cublasHandle_t   cublas_handle_;
  cublasLtHandle_t cublaslt_handle_;
  cudaStream_t     stream_ = 0;
  cublasINT8MMWrapper* cublas_wrapper_;
  std::mutex* cublas_wrapper_mutex_;
  fastertransformer::Allocator<AllocatorType::CUDA>* allocator_;
  AttentionType attention_type_;
  ViTINT8Weight<float> params_fp32_;
  ViTINT8Weight<half> params_fp16_;
  ViTTransformerINT8<float>* vit_fp32_;
  ViTTransformerINT8<half>* vit_fp16_;
  cublasAlgoMap* cublas_algo_map_;
  void* compute_stream_;

  int batch_size_;
  int img_size_;
  int patch_size_;
  int embed_dim_;
  int seq_len_;
  int in_chans_;
  int is_fp16_;
  int weights_num_;
};

struct FTViTINT8CustomOp : Ort::CustomOpBase<FTViTINT8CustomOp, FTViTINT8CustomKernel> {
  explicit FTViTINT8CustomOp(const char* provider, void* compute_stream);

  void* CreateKernel(const OrtApi&, const OrtKernelInfo*) const;

  OrtMemType GetInputMemoryType(size_t index) const;

  const char* GetName() const { return "FTViTINT8"; };

  const char* GetExecutionProviderType() const { return provider_; };

  size_t GetInputTypeCount() const { return 223; };

  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED; };

  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t) const { return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL; };   

  size_t GetOutputTypeCount() const { return 1; };

  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED; };

  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t) const { return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL; };

 private:
  const char* provider_;
  void* compute_stream_;
};
