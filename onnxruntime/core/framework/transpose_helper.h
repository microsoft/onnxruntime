// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

/*
This file contains optimizations for moving a single axis either inwards or outwards.

If moving outwards we can use a single reader and multiple writers. The number of writers is equal to the value of
the axis being moved.

  e.g. if the input is NHWC with shape {N, 300, 300, 3}, we can transpose to NCHW by reading once and having
       one writer for each of the 3 channels at a different offset in the output, updating the offset for each item
       in the batch of N.

Similarly if one axis is moving inwards we can use a single writer and multiple readers. The number of readers is equal
to the value of the axis being moved.

  e.g. if the input is NCHW with shape {N, 3, 300, 300}, we can transpose to NHWC by writing once using one reader for
       each of the 3 channels at a different offset in the input, updating the read offset for each item in the batch
       of N.

This can be generalized for any input where only one axis is being moved, with the block size for each read/write
being dependent on which axis is moving, what direction it's moving in, and where it's moving to.

We use simple pointer arithmetic if the size of each read/write is a power of 2 and between 8 and 64 bits.
We use memcpy if the block size is larger.

We fall back to the default implementation in all other cases, and if the input is std::string.
*/

#include <sstream>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/tensor.h"
#include "core/platform/threadpool.h"

#include "core/common/gsl.h"

namespace onnxruntime {
bool IsTransposeMovingSingleAxis(gsl::span<const size_t> permutations, size_t& from, size_t& to);
void SingleAxisTranspose(gsl::span<const size_t> permutations, const Tensor& input, Tensor& output, size_t from,
                         size_t to, const TensorShape* input_shape_override = nullptr,
                         concurrency::ThreadPool* tp = nullptr);
}  // namespace onnxruntime
