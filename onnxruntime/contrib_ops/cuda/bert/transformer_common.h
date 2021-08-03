// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {
namespace cuda {

class TransformerOptions {
 public:
  static const TransformerOptions* GetInstance();

  bool IsPrecisionMode() const { return is_precision_mode_; }

  bool DisablePersistentSoftmax() const { return disable_persistent_softmax_; }

  bool IsExperimentalMode() const { return is_experimental_; }

  bool IsBetaMode() const { return is_beta_; }

  void Initialize(int value) {
    is_precision_mode_ = (value & 0x01) > 0;
    disable_persistent_softmax_ = (value & 0x02) > 0;
    is_experimental_ = (value & 0x04) > 0;
    is_beta_ = (value & 0x08) > 0;
    initialized_ = true;
  }

 private:
  // Default is false. If the mode is on, prefer precision than speed.
  bool is_precision_mode_{false};

  // Disable persistent softmax.
  bool disable_persistent_softmax_{false};

  // Use experimental algorithm
  bool is_experimental_{false};

  // Use beta algorithm
  bool is_beta_{false};

  bool initialized_{false};

  static TransformerOptions instance;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
