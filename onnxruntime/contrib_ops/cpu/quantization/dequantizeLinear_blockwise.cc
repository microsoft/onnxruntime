// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// This module define DequantizeLinearBlockWise operator, it is basically
// dequantize input tensor and unpack it into float/half tensor.
//
#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/op_kernel.h"

#include "core/mlas/inc/mlas_q4.h"
#include "core/providers/common.h"
#include "dequantizeLinear_blockwise_imp.h"

namespace onnxruntime {
namespace contrib {

class DequantizeLinearBlockWise final : public OpKernel {
 public:
  DequantizeLinearBlockWise(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("K", &K_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("N", &N_));
    ORT_ENFORCE(K_ * N_ > 0, "K and N must be greater than 0.");
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("block_size", &block_size_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("bits", &nbits_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("axis", &axis_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<std::string>("packing", &packing_));
    if (packing_ == "default") {
      ORT_ENFORCE(axis_ == 1, "axis_ should be 1 for default packing.");
      ORT_ENFORCE(nbits_ == 4,
                  "Only 4b quantization is supported for DequantizeLinearBlockWise op,"
                  " additional bits support is planned.");
    } else if (packing_ == "gptq") {
      ORT_ENFORCE(axis_ == 0, "axis_ should be 0 for gptq packing.");
      ORT_ENFORCE(nbits_ == 4, "nbits_ should be 4.");
    } else if (packing_ == "hqq") {
      ORT_ENFORCE(axis_ == 0, "axis_ should be 0 for hqq packing.");
      ORT_ENFORCE(nbits_ == 4, "nbits_ should be 4.");
    } else {
      ORT_THROW("Unsupported packing type: ", packing_);
    }
  }

  Status Compute(OpKernelContext* context) const override;
  Status ComputeGPTQ(OpKernelContext* context) const;
  Status ComputeHQQ(OpKernelContext* context) const;

 private:
  int64_t K_;
  int64_t N_;
  int64_t block_size_;
  int64_t axis_;
  int64_t nbits_;
  std::string packing_;
};

Status DequantizeLinearBlockWise::ComputeGPTQ(OpKernelContext* ctx) const {
  const int in_features = K_;
  const int out_features = N_;
  const int groupsize = block_size_;
  const auto* input_qweight = ctx->Input<Tensor>(0);
  const auto* input_scale = ctx->Input<Tensor>(1);
  const auto* input_zeros = ctx->Input<Tensor>(2);
  const auto* input_gidx = ctx->Input<Tensor>(3);
  const auto& weight_shape = input_qweight->Shape();

  auto OutputShape = TensorShape({in_features, out_features});

  Tensor* Y = ctx->Output(0, OutputShape);
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  auto fp16_weight_shape = weight_shape;
  fp16_weight_shape[0] = in_features;
  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

  const auto* zero_point = input_zeros && input_zeros->DataRaw() ? input_zeros->DataRaw() : nullptr;
  if (nbits_ != 4) {
    GPTQPacking::GeneralDequant(thread_pool, input_qweight->Data<int32_t>(),
                                input_scale->Data<float>(),
                                static_cast<const uint32_t*>(zero_point),
                                input_gidx->Data<int32_t>(),
                                Y->MutableData<float>(),
                                in_features, weight_shape[1], nbits_, groupsize);
  } else if (input_gidx && input_gidx->Shape().Size() > 1) {
    GPTQPacking::DequantWeightNbitGidx(thread_pool, input_qweight->Data<int32_t>(),
                                       input_scale->Data<float>(),
                                       static_cast<const uint32_t*>(zero_point),
                                       input_gidx->Data<int32_t>(),
                                       Y->MutableData<float>(),
                                       in_features, weight_shape[1], nbits_);
  } else {
    GPTQPacking::DequantWeightNbit(thread_pool, input_qweight->Data<int32_t>(),
                                   input_scale->Data<float>(),
                                   static_cast<const uint32_t*>(zero_point),
                                   Y->MutableData<float>(),
                                   in_features, weight_shape[1], nbits_, groupsize);
  }
  return Status::OK();
}

Status DequantizeLinearBlockWise::ComputeHQQ(OpKernelContext* ctx) const {
  const int in_features = K_;
  // const int out_features = N_;
  const int groupsize = block_size_;
  const auto* input_qweight = ctx->Input<Tensor>(0);
  const auto* input_scale = ctx->Input<Tensor>(1);
  const auto* input_zeros = ctx->Input<Tensor>(2);
  const auto& weight_shape = input_qweight->Shape();
  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();
  auto OutputShape = TensorShape({in_features, N_});

  Tensor* Y = ctx->Output(0, OutputShape);

  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }
  auto fp16_weight_shape = weight_shape;
  fp16_weight_shape[0] = in_features;
  if (nbits_ != 4) {
    GPTQPacking::GeneralDequant(thread_pool, input_qweight->Data<int32_t>(),
                                input_scale->Data<float>(),
                                input_zeros->Data<float>(),
                                nullptr,
                                Y->MutableData<float>(),
                                in_features, weight_shape[1], nbits_, groupsize);
  } else{
    GPTQPacking::DequantWeightNbit(thread_pool, input_qweight->Data<int32_t>(),
                                   input_scale->Data<float>(),
                                   input_zeros->Data<float>(),
                                   Y->MutableData<float>(),
                                   in_features, weight_shape[1], nbits_, groupsize);
  }
  return Status::OK();
}

Status DequantizeLinearBlockWise::Compute(OpKernelContext* ctx) const {
  if (packing_ == "gptq") {
    return this->ComputeGPTQ(ctx);
  }
  if (packing_ == "hqq") {
    return this->ComputeHQQ(ctx);
  }
  const Tensor* b = ctx->Input<Tensor>(0);
  const Tensor* scales = ctx->Input<Tensor>(1);
  const Tensor* zero_points = ctx->Input<Tensor>(2);

  const uint8_t* blob_data = b->Data<uint8_t>();
  const auto* scales_data = scales->Data<float>();
  const auto* zero_points_data = zero_points == nullptr ? nullptr : zero_points->Data<uint8_t>();

  TensorShape b_shape({N_, K_});

  Tensor* Y = ctx->Output(0, b_shape);
  // Bail out early if the output is going to be empty
  if (Y->Shape().Size() == 0) return Status::OK();
  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

  MlasDequantizeBlockwise<float, 4>(
      Y->MutableData<float>(),            // dequantized output
      blob_data,                          // quantized input
      scales_data,                        // quantization scales
      zero_points_data,                   // quantization zero points
      static_cast<int32_t>(block_size_),  // quantization block size
      axis_,                              // columnwise quantization or row-wise
      static_cast<int32_t>(K_),           // number of rows in quantized input
      static_cast<int32_t>(N_),           // number of columns in quantized input
      thread_pool);

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    DequantizeLinearBlockWise,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(),
                               DataTypeImpl::GetTensorType<int32_t>(),
                               DataTypeImpl::GetTensorType<uint32_t>()}),
    DequantizeLinearBlockWise);

}  // namespace contrib
}  // namespace onnxruntime
