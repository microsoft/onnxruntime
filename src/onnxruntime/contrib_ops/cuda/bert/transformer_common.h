// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

class TransformerOptions {
 public:
  static const TransformerOptions* GetInstance();

  bool IsPrecisionMode() const { return is_precision_mode_; }

  bool DisablePersistentSoftmax() const { return disable_persistent_softmax_; }

  bool DisableHalf2() const { return disable_half2_; }

  void Initialize(int value) {
    is_precision_mode_ = (value & 0x01) > 0;
    disable_persistent_softmax_ = (value & 0x02) > 0;
    disable_half2_ = (value & 0x04) > 0;
    initialized_ = true;
  }

 private:
  // Default is false. If the mode is on, prefer precision than speed.
  bool is_precision_mode_{false};

  // Disable persistent softmax.
  bool disable_persistent_softmax_{false};

  // Disable half2 kernel.
  bool disable_half2_{false};

  bool initialized_{false};

  static TransformerOptions instance;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
