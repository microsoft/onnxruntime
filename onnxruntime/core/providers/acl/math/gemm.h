// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/cpu/math/gemm.h"
#include "core/providers/acl/acl_execution_provider.h"

// ACL
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "arm_compute/runtime/Allocator.h"
#include "arm_compute/runtime/PoolManager.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"

//NEON
#include "arm_compute/runtime/NEON/functions/NEGEMM.h"
#include "arm_compute/runtime/NEON/functions/NETranspose.h"

#undef GEMM_ACL
#define CACHE_TRANSPOSED_DATA

namespace onnxruntime {
namespace acl {

typedef struct {
  std::shared_ptr<arm_compute::NEGEMM> layer;
  std::shared_ptr<arm_compute::Tensor> a, b, c, d;
  std::shared_ptr<arm_compute::MemoryManagerOnDemand> mm_layer;
} ACLNEGEMM;

typedef std::map<OpKernel*, ACLNEGEMM>::iterator GEMMLayersIterator;

template <typename T>
class Gemm : public onnxruntime::Gemm<T> {
 public:
  Gemm(const OpKernelInfo& info) : onnxruntime::Gemm<T>(info) {
    int64_t temp;

    ORT_ENFORCE(info.GetAttr<int64_t>("transA", &temp).IsOK());
    trans_A_ = temp == 0 ? CblasNoTrans : CblasTrans;
    ORT_ENFORCE(info.GetAttr<int64_t>("transB", &temp).IsOK());
    trans_B_ = temp == 0 ? CblasNoTrans : CblasTrans;

    ORT_ENFORCE(info.GetAttr<float>("alpha", &alpha_).IsOK());
    ORT_ENFORCE(info.GetAttr<float>("beta", &beta_).IsOK());
  }

#ifdef GEMM_ACL
  Status Compute(OpKernelContext* context) const override {
    const auto X = context->Input<Tensor>(0);
    const auto W = context->Input<Tensor>(1);
    const auto B = context->Input<Tensor>(2);

    GemmHelper helper(X->Shape(), trans_A_ != CblasNoTrans, W->Shape(), trans_B_ != CblasNoTrans, B->Shape());

    if (!helper.State().IsOK())
      return helper.State();

    int64_t M = helper.M();
    int64_t N = helper.N();
    auto Y = context->Output(0, TensorShape({M, N}));

    int64_t K = helper.K();
    LOGS_DEFAULT(VERBOSE) << "Gemm ACL:" << std::endl;
    if (X) LOGS_DEFAULT(VERBOSE) << "X " << X->Shape().ToString().c_str() << std::endl;
    if (W) LOGS_DEFAULT(VERBOSE) << "W " << W->Shape().ToString().c_str() << std::endl;
    if (B) LOGS_DEFAULT(VERBOSE) << "B " << B->Shape().ToString().c_str() << std::endl;
    LOGS_DEFAULT(VERBOSE) << "Y " << Y->Shape().ToString().c_str() << std::endl;
    LOGS_DEFAULT(VERBOSE) << "M " << (int)M << ", N " << (int)N << ", K " << (int)K << std::endl;
    LOGS_DEFAULT(VERBOSE) << "Alfa " << alpha_ << ", Beta " << beta_ << std::endl;
    LOGS_DEFAULT(VERBOSE) << "trans_A_ " << (trans_A_ == CblasTrans) << std::endl;
    LOGS_DEFAULT(VERBOSE) << "trans_B_ " << (trans_B_ == CblasTrans) << std::endl;
    LOGS_DEFAULT(VERBOSE) << std::endl;

    ACLNEGEMM* pGEMM;
    GEMMLayersIterator it = gemmLayers.find((OpKernel*)this);
    if (it == gemmLayers.end()) {
      ACLNEGEMM tGEMM;
      tGEMM.a = std::make_shared<arm_compute::Tensor>();
      tGEMM.b = std::make_shared<arm_compute::Tensor>();
      tGEMM.c = std::make_shared<arm_compute::Tensor>();
      tGEMM.d = std::make_shared<arm_compute::Tensor>();

      tGEMM.a->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(X->Shape()), arm_compute::Format::F32));
      tGEMM.c->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(B->Shape()), arm_compute::Format::F32));
      // dimensions are stored in the opposite order to ACL's
      tGEMM.d->allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(N, M), tGEMM.a->info()->format()));

      // transpose
      if (trans_B_ == CblasTrans) {
        auto trans_layer = std::make_shared<arm_compute::NETranspose>();
        tGEMM.b->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(W->Shape()), arm_compute::Format::F32));

        arm_compute::Tensor tmp;
        tmp.allocator()->init(arm_compute::TensorInfo(ACLTensorShape(W->Shape()), arm_compute::Format::F32));

        trans_layer->configure(&tmp, tGEMM.b.get());

        const T* b_data = W->template Data<T>();
        ACLImportMemory(tmp.allocator(), (void*)b_data, W->Shape().Size() * 4);

        tGEMM.b->allocator()->allocate();

        trans_layer->run();
      } else {
        tGEMM.b->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(W->Shape()), arm_compute::Format::F32));
      }

      tGEMM.mm_layer = ACLCreateMemoryManager();
      tGEMM.layer = std::make_shared<arm_compute::NEGEMM>(tGEMM.mm_layer);

      // configure GEMM
      tGEMM.layer->configure(tGEMM.a.get(), tGEMM.b.get(), tGEMM.c.get(), tGEMM.d.get(), alpha_, beta_, arm_compute::GEMMInfo());

      // non-transpose
      if (trans_B_ != CblasTrans) {
        const T* b_data = W->template Data<T>();
        ACLImportMemory(tGEMM.b->allocator(), (void*)b_data, W->Shape().Size() * 4);
      }

      std::pair<GEMMLayersIterator, bool> ret;
      ret = gemmLayers.insert(std::pair<OpKernel*, ACLNEGEMM>((OpKernel*)this, tGEMM));
      pGEMM = &ret.first->second;
    } else {
      //TODO: valildate shapes
      pGEMM = &it->second;

      // transpose
      if (trans_B_ == CblasTrans) {
#ifndef CACHE_TRANSPOSED_DATA
        auto trans_layer = std::make_shared<arm_compute::NETranspose>();
        pGEMM->b->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(W->Shape()), arm_compute::Format::F32));

        arm_compute::Tensor tmp;
        tmp.allocator()->init(arm_compute::TensorInfo(ACLTensorShape(W->Shape()), arm_compute::Format::F32));

        trans_layer->configure(&tmp, pGEMM->b.get());

        const T* b_data = W->template Data<T>();
        ACLImportMemory(tmp.allocator(), (void*)b_data, W->Shape().Size() * 4);

        // allocate memory for b
        pGEMM->b->allocator()->allocate();

        trans_layer->run();
#else
        LOGS_DEFAULT(VERBOSE) << "Reuse transposed data" << std::endl;
#endif
      } else {
        const T* b_data = W->template Data<T>();
        ACLImportMemory(pGEMM->b->allocator(), (void*)b_data, W->Shape().Size() * 4);
      }
    }

    const T* a_data = X->template Data<T>();
    const T* c_data = B->template Data<T>();
    const T* d_data = Y->template Data<T>();

    ACLImportMemory(pGEMM->a->allocator(), (void*)a_data, X->Shape().Size() * 4);
    ACLImportMemory(pGEMM->c->allocator(), (void*)c_data, B->Shape().Size() * 4);
    ACLImportMemory(pGEMM->d->allocator(), (void*)d_data, Y->Shape().Size() * 4);

    ACLPrintTensorShape("a", *pGEMM->a);
    ACLPrintTensorShape("b", *pGEMM->b);
    ACLPrintTensorShape("c", *pGEMM->c);
    ACLPrintTensorShape("d", *pGEMM->d);

    arm_compute::Allocator alloc_mm{};
    pGEMM->mm_layer->populate(alloc_mm, 1);
    pGEMM->layer->run();
    pGEMM->mm_layer->clear();

    pGEMM->a->allocator()->free();
#ifdef CACHE_TRANSPOSED_DATA
    if (trans_B_ != CblasTrans)
      pGEMM->b->allocator()->free();
#else
    pGEMM->b->allocator()->free();
#endif
    pGEMM->c->allocator()->free();
    pGEMM->d->allocator()->free();

    return Status::OK();
  }

  ~Gemm() {
    gemmLayers.erase(this);
  }

#else

  Status Compute(OpKernelContext* context) const override {
    const auto X = context->Input<Tensor>(0);
    const auto W = context->Input<Tensor>(1);
    const auto B = context->Input<Tensor>(2);

    GemmHelper helper(X->Shape(), trans_A_ != CblasNoTrans, W->Shape(), trans_B_ != CblasNoTrans, B->Shape());

    if (!helper.State().IsOK())
      return helper.State();

    int64_t M = helper.M();
    int64_t N = helper.N();
    auto Y = context->Output(0, TensorShape({M, N}));

    int64_t K = helper.K();
    LOGS_DEFAULT(VERBOSE) << "Gemm CPU:" << std::endl;
    if (X) LOGS_DEFAULT(VERBOSE) << "X " << X->Shape().ToString().c_str() << std::endl;
    if (W) LOGS_DEFAULT(VERBOSE) << "W " << W->Shape().ToString().c_str() << std::endl;
    if (B) LOGS_DEFAULT(VERBOSE) << "B " << B->Shape().ToString().c_str() << std::endl;
    LOGS_DEFAULT(VERBOSE) << "Y " << Y->Shape().ToString().c_str() << std::endl;
    LOGS_DEFAULT(VERBOSE) << "M " << (int)M << ", N " << (int)N << ", K " << (int)K << std::endl;
    LOGS_DEFAULT(VERBOSE) << "Alfa " << alpha_ << ", Beta " << beta_ << std::endl;
    LOGS_DEFAULT(VERBOSE) << std::endl;

    return onnxruntime::Gemm<T>::Compute(context);
  }
#endif

 private:
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
