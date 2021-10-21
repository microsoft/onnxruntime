// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class CDist final : public OpKernel {
 private:
  typedef void (*DistFunc)(const T* a, const T* b, T* dest, size_t ma, size_t mb, size_t n,
                           concurrency::ThreadPool* tp);
  enum class Mode { EUCLIDEAN,
                    SQEUCLIDEAN } mode_;

 public:
  CDist(const OpKernelInfo& info) : OpKernel(info) {
    std::string metric;
    ORT_ENFORCE(info.GetAttr<std::string>("metric", &metric).IsOK());
    if (metric.compare("sqeuclidean") == 0)
      mode_ = Mode::SQEUCLIDEAN;
    else if (metric.compare("euclidean") == 0) {
      mode_ = Mode::EUCLIDEAN;
    } else
      ORT_NOT_IMPLEMENTED();
  }

  common::Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
