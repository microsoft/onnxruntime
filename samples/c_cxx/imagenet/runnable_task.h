// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <functional>
#include "sync_api.h"

class RunnableTask : public std::unary_function<void, void> {
 public:
  virtual void operator()(_Inout_opt_ ONNXRUNTIME_CALLBACK_INSTANCE pci) noexcept = 0;
  virtual ~RunnableTask() = default;
};
