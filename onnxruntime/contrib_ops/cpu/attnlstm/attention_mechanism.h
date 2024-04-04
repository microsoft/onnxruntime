// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/gsl.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class IAttentionMechanism {
 public:
  virtual ~IAttentionMechanism() = default;

  virtual void PrepareMemory(
      const gsl::span<const T>& memory,
      const gsl::span<const int>& memory_sequence_lengths) = 0;

  virtual void Compute(
      const gsl::span<const T>& query,
      const gsl::span<const T>& prev_alignment,
      const gsl::span<T>& output,
      const gsl::span<T>& alignment) const = 0;

  virtual const gsl::span<const T> Values() const = 0;

  virtual const gsl::span<const T> Keys() const = 0;

  virtual int GetMaxMemorySteps() const = 0;

  virtual bool NeedPrevAlignment() const = 0;
};

}  // namespace contrib
}  // namespace onnxruntime
