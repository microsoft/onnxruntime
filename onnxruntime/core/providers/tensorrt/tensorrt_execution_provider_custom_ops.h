// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"

using namespace onnxruntime;

namespace onnxruntime {


struct TensorRTCustomKernel {
  TensorRTCustomKernel(const OrtKernelInfo* /*info*/, void* compute_stream)
      : compute_stream_(compute_stream) {
  }

  void Compute(OrtKernelContext* context){};  // The implementation is in TensorRT repos.

 private:
  void* compute_stream_;
};

struct DisentangledAttentionCustomOp : Ort::CustomOpBase<DisentangledAttentionCustomOp, TensorRTCustomKernel> {
  explicit DisentangledAttentionCustomOp(const char* provider, void* compute_stream) : provider_(provider), compute_stream_(compute_stream) {}

  void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* info) const { return new TensorRTCustomKernel(info, compute_stream_); };
  const char* GetName() const { return "DisentangledAttention_TRT"; };
  const char* GetExecutionProviderType() const { return provider_; };

  size_t GetInputTypeCount() const { return 3; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    // Both the inputs need to be necessarily of float type
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

 private:
  const char* provider_{"TensorrtExecutionProvider"};
  void* compute_stream_;
};

/*
struct OrtCustomOpDomain {
  std::string domain_;
  std::vector<const OrtCustomOp*> custom_ops_;
};
*/
/*
void CreateTensorRTCustomOpDomain_old(OrtCustomOpDomain* domain) {
  std::unique_ptr<OrtCustomOpDomain> custom_op_domain = std::make_unique<OrtCustomOpDomain>();
  custom_op_domain->domain_ = "";

  std::unique_ptr<OrtCustomOp> disentangled_attention_custom_op = std::make_unique<DisentangledAttentionCustomOp>("TensorrtExecutionProvider", nullptr);
  custom_op_domain->custom_ops_.push_back(disentangled_attention_custom_op.release());
  //custom_ops.push_back(disentangled_attention_custom_op.release());

  domain = custom_op_domain.release();
}
*/

void CreateTensorRTCustomOpDomain(OrtProviderCustomOpDomain** domain) {
  std::unique_ptr<OrtProviderCustomOpDomain> custom_op_domain = std::make_unique<OrtProviderCustomOpDomain>();
  custom_op_domain->domain_ = "";

  std::unique_ptr<OrtCustomOp> disentangled_attention_custom_op = std::make_unique<DisentangledAttentionCustomOp>("TensorrtExecutionProvider", nullptr);
  custom_op_domain->custom_ops_.push_back(disentangled_attention_custom_op.release());
  //custom_ops.push_back(disentangled_attention_custom_op.release());

  *domain = custom_op_domain.release();
}

}