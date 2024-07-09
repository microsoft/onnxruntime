// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <emscripten/val.h>

#include "core/framework/data_transfer.h"
#include "core/framework/execution_provider.h"

namespace onnxruntime {
namespace webnn {

class DataTransfer : public IDataTransfer {
 public:
  explicit DataTransfer(const emscripten::val& wnn_context, const IExecutionProvider* provider) : wnn_context_(wnn_context), provider_(provider){};
  ~DataTransfer() override = default;

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

  common::Status CopyTensor(const Tensor& src, Tensor& dst) const override;

 private:
  emscripten::val wnn_context_;
  const IExecutionProvider* provider_;
};

}  // namespace webnn
}  // namespace onnxruntime
