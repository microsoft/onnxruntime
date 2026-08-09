// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019-2020, NXP Semiconductor, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/cpu/math/gemm.h"
#include "core/providers/cpu/math/gemm_helper.h"
#include "core/providers/acl/acl_execution_provider.h"

// ACL
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "arm_compute/runtime/Allocator.h"
#include "arm_compute/runtime/PoolManager.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"

// NEON
#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"

namespace onnxruntime {
namespace acl {

typedef struct {
  std::shared_ptr<arm_compute::IFunction> layer;
  std::shared_ptr<arm_compute::Tensor> a, b, c, d;
} ACLNEGEMM;

typedef std::map<OpKernel*, ACLNEGEMM>::iterator GEMMLayersIterator;

template <typename T>
class Gemm : public onnxruntime::Gemm<T> {
 public:
  Gemm(const OpKernelInfo& info) : onnxruntime::Gemm<T>(info) {
    provider_ = (const_cast<ACLExecutionProvider*>(
        static_cast<const ACLExecutionProvider*>(info.GetExecutionProvider())));

    int64_t temp;

    ORT_ENFORCE(info.GetAttr<int64_t>("transA", &temp).IsOK());
    trans_A_ = temp == 0 ? CblasNoTrans : CblasTrans;
    ORT_ENFORCE(info.GetAttr<int64_t>("transB", &temp).IsOK());
    trans_B_ = temp == 0 ? CblasNoTrans : CblasTrans;

    ORT_ENFORCE(info.GetAttr<float>("alpha", &alpha_).IsOK());
    ORT_ENFORCE(info.GetAttr<float>("beta", &beta_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override {
    if (this->packed_b_) {
      // Prepacked RHS not supported, defaulting to cpu execution provider
      return onnxruntime::Gemm<T>::Compute(context);
    }

    const auto A = context->Input<Tensor>(0);
    const auto B = context->Input<Tensor>(1);
    const auto C = context->Input<Tensor>(2);

    GemmHelper helper(A->Shape(), trans_A_ != CblasNoTrans, B->Shape(), trans_B_ != CblasNoTrans,
                      C != nullptr ? C->Shape() : TensorShape({}));

    if (!helper.State().IsOK())
      return helper.State();

    int64_t M = helper.M();
    int64_t N = helper.N();
    auto D = context->Output(0, TensorShape({M, N}));

    bool FC = alpha_ == 1 && (beta_ == 1 || beta_ == 0);
    bool useC = C != nullptr && beta_ != 0;

    if (trans_A_ == CblasTrans) {  // transpose input
      LOGS_DEFAULT(WARNING) << "Transposed input not supported ; defaulting to cpu implementation";
      return onnxruntime::Gemm<T>::Compute(context);
    }

    arm_compute::TensorShape cShape = ACLTensorShape(C != nullptr ? C->Shape() : TensorShape({}));
    if (useC &&
        (cShape.num_dimensions() > 2 ||
         (cShape.num_dimensions() == 2 && cShape[0] > 1 && cShape[1] > 1))) {  // Multi-dimensional Bias
      LOGS_DEFAULT(WARNING) << "Multi-dimensional Bias not supported in this implementation; defaulting to cpu implementation";
      return onnxruntime::Gemm<T>::Compute(context);
    }

    if (useC && (cShape.num_dimensions() == 1 && cShape[0] != (long unsigned int)N)) {  // Broadcast
      LOGS_DEFAULT(WARNING) << "Multi-dimensional Bias not supported in this implementation; defaulting to cpu implementation";
      return onnxruntime::Gemm<T>::Compute(context);
    }

    LOGS_DEFAULT(VERBOSE) << "Gemm ACL:";
    if (useC && cShape.num_dimensions() == 2) {
      if ((cShape[0] == 1 && cShape[1] != (long unsigned int)N) ||
          (cShape[1] == 1 && cShape[0] != (long unsigned int)N)) {
        return onnxruntime::Gemm<T>::Compute(context);
      }
      cShape = arm_compute::TensorShape(N);
      LOGS_DEFAULT(VERBOSE) << "Bias reshaped to: {" << N << "}";
    }

    int64_t K = helper.K();
    if (A) {
      LOGS_DEFAULT(VERBOSE) << "A " << A->Shape().ToString().c_str();
    }
    if (B) {
      LOGS_DEFAULT(VERBOSE) << "B " << B->Shape().ToString().c_str();
    }
    if (C) {
      LOGS_DEFAULT(VERBOSE) << "C " << C->Shape().ToString().c_str();
    }
    LOGS_DEFAULT(VERBOSE) << "D " << D->Shape().ToString().c_str();
    LOGS_DEFAULT(VERBOSE) << "M " << (int)M << ", N " << (int)N << ", K " << (int)K;
    LOGS_DEFAULT(VERBOSE) << "Alfa " << alpha_ << ", Beta " << beta_;
    LOGS_DEFAULT(VERBOSE) << "trans_A_ " << (trans_A_ == CblasTrans);
    LOGS_DEFAULT(VERBOSE) << "trans_B_ " << (trans_B_ == CblasTrans);
    LOGS_DEFAULT(VERBOSE) << std::endl;

    ACLNEGEMM* pGEMM;
    GEMMLayersIterator it = gemmLayers.find((OpKernel*)this);
    if (it == gemmLayers.end()) {
      ACLNEGEMM tGEMM;
      tGEMM.a = std::make_shared<arm_compute::Tensor>();
      tGEMM.b = std::make_shared<arm_compute::Tensor>();
      tGEMM.c = std::make_shared<arm_compute::Tensor>();
      tGEMM.d = std::make_shared<arm_compute::Tensor>();

      tGEMM.a->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(A->Shape()), arm_compute::Format::F32));
      tGEMM.b->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(B->Shape()), arm_compute::Format::F32));
      tGEMM.c->allocator()->init(arm_compute::TensorInfo(cShape, arm_compute::Format::F32));
      // dimensions are stored in the opposite order to ACL's
      tGEMM.d->allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(N, M), arm_compute::Format::F32));

      if (FC) {
        auto layer = std::make_shared<arm_compute::NEFullyConnectedLayer>(provider_->memory_manager);
        arm_compute::FullyConnectedLayerInfo fc_info;
        fc_info.transpose_weights = trans_B_ == CblasTrans;
        layer->configure(tGEMM.a.get(), tGEMM.b.get(), useC ? tGEMM.c.get() : nullptr, tGEMM.d.get(), fc_info);
        tGEMM.layer = std::move(layer);
      } else {
        return onnxruntime::Gemm<T>::Compute(context);
      }

      std::pair<GEMMLayersIterator, bool> ret;
      ret = gemmLayers.insert(std::pair<OpKernel*, ACLNEGEMM>((OpKernel*)this, tGEMM));
      pGEMM = &ret.first->second;
    } else {
      // TODO: valildate shapes
      pGEMM = &it->second;
    }

    const T* a_data = A->Data<T>();
    const T* b_data = B->Data<T>();
    T* d_data = D->MutableData<T>();

    ACLImportMemory(pGEMM->a->allocator(), (void*)a_data, A->Shape().Size() * 4);
    ACLImportMemory(pGEMM->b->allocator(), (void*)b_data, B->Shape().Size() * 4);
    if (useC) {
      const T* c_data = C->Data<T>();
      ACLImportMemory(pGEMM->c->allocator(), (void*)c_data, C->Shape().Size() * 4);
    }

    if (D->Shape().Size() != 0 && pGEMM->d->info()->has_padding()) {
      pGEMM->d.get()->allocator()->allocate();
    } else {
      ACLImportMemory(pGEMM->d->allocator(), (void*)d_data, D->Shape().Size() * 4);
    }

    ACLPrintTensorShape("a", *pGEMM->a);
    ACLPrintTensorShape("b", *pGEMM->b);
    ACLPrintTensorShape("c", *pGEMM->c);
    ACLPrintTensorShape("d", *pGEMM->d);

    pGEMM->layer->run();

    if (D->Shape().Size() != 0 && pGEMM->d->info()->has_padding()) {
      importDataFromTensor<T>(pGEMM->d.get(), d_data);
    }

    pGEMM->a->allocator()->free();
    pGEMM->b->allocator()->free();
    pGEMM->c->allocator()->free();
    pGEMM->d->allocator()->free();

    return Status::OK();
  }

  ~Gemm() {
    gemmLayers.erase(this);
  }

 private:
  ACLExecutionProvider* provider_;
  static thread_local std::map<OpKernel*, ACLNEGEMM> gemmLayers;

  CBLAS_TRANSPOSE trans_A_;
  CBLAS_TRANSPOSE trans_B_;
  float alpha_;
  float beta_;
};

template <typename T>
thread_local std::map<OpKernel*, ACLNEGEMM> onnxruntime::acl::Gemm<T>::gemmLayers;

}  // namespace acl
}  // namespace onnxruntime
