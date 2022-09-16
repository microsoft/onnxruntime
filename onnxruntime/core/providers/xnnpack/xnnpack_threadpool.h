// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/platform/threadpool.h"

struct pthreadpool;

namespace onnxruntime {
namespace concurrency {

class XnnpackThreadPool final : public ThreadPool {
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(XnnpackThreadPool);

 public:
  explicit XnnpackThreadPool(size_t thread_num);
  ~XnnpackThreadPool() override;
  int NumThreads() const override;
  void ParallelFor(std::ptrdiff_t total, double cost_per_unit,
                   const std::function<void(std::ptrdiff_t first, std::ptrdiff_t last)>& fn) override;

  void ParallelFor(std::ptrdiff_t total, const TensorOpCost& cost_per_unit,
                   const std::function<void(std::ptrdiff_t first, std::ptrdiff_t)>& fn) override;

  void SimpleParallelFor(std::ptrdiff_t total, const std::function<void(std::ptrdiff_t)>& fn) override;

  void Schedule(std::function<void()> fn) override;

  void StartProfiling() override;

  std::string StopProfiling() override;
  pthreadpool* Get() { return xnnpack_thread_pool_; }

 private:
  pthreadpool* xnnpack_thread_pool_{nullptr};
};

}  // namespace concurrency
}  // namespace onnxruntime
