// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <emscripten/val.h>

#include "core/framework/data_transfer.h"

namespace onnxruntime {
namespace webnn {

class DataTransfer : public IDataTransfer {
 public:
  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

  common::Status CopyTensor(const Tensor& src, Tensor& dst) const override;
};

}  // namespace webnn
}  // namespace onnxruntime
