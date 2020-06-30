// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
Status ComputeMaskIndex(const T* mask,        // input mask, NULL when no mask input.
                        int* mask_index,      // output mask index
                        int batch_size,       // batch size
                        int sequence_length)  // sequence length
{
  if (nullptr == mask) {
    for (int i = 0; i < batch_size; i++) {
      *mask_index++ = sequence_length;
    }
    for (int i = 0; i < batch_size; i++) {
      *mask_index++ = 0;
    }
  } else {
    const T* m = mask;
    for (int i = 0; i < batch_size; i++) {
      int min_index = sequence_length;
      for (int j = 0; j < sequence_length; j++) {
        if (m[j] > T(0)) {
          min_index = j;
          break;
        }
      }

      int max_index = -1;
      for (int j = sequence_length - 1; j >= 0; j--) {
        if (m[j] > T(0)) {
          max_index = j;
          break;
        }
      }

      if (max_index >= 0) {
        for (int k = min_index + 1; k < max_index; k++) {
          if (m[k] == T(0)) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                   "MaskIndex only support attention mask with one contiguous block of 1 in a batch.");
          }
        }
      }

      mask_index[i] = max_index + 1;
      mask_index[batch_size + i] = min_index;
      m += sequence_length;
    }
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
