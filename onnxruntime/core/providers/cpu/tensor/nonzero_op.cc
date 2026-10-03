// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/nonzero_op.h"

#include <algorithm>
#include <cassert>
#include <core/common/safeint.h>
#include "core/common/inlined_containers.h"
#include "core/common/narrow.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
// kernel builder functions
#define NONZERO_9_TYPED_KERNEL(type)                                               \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                        \
      NonZero,                                                                     \
      9, 12,                                                                       \
      type,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<type>()), \
      NonZero<type>)

#define NONZERO_TYPED_KERNEL(type)                                                 \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                  \
      NonZero,                                                                     \
      13,                                                                          \
      type,                                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<type>()), \
      NonZero<type>)

NONZERO_9_TYPED_KERNEL(bool)
NONZERO_9_TYPED_KERNEL(uint8_t)
NONZERO_9_TYPED_KERNEL(int32_t)
NONZERO_9_TYPED_KERNEL(int64_t)
NONZERO_9_TYPED_KERNEL(float)

// start with a subset of types, enable more as needed...
NONZERO_TYPED_KERNEL(bool)
NONZERO_TYPED_KERNEL(uint8_t)
// NONZERO_TYPED_KERNEL(uint16_t)
// NONZERO_TYPED_KERNEL(uint32_t)
// NONZERO_TYPED_KERNEL(uint64_t)
// NONZERO_TYPED_KERNEL(int8_t)
// NONZERO_TYPED_KERNEL(int16_t)
NONZERO_TYPED_KERNEL(int32_t)
NONZERO_TYPED_KERNEL(int64_t)
// NONZERO_TYPED_KERNEL(MLFloat16)
// NONZERO_TYPED_KERNEL(BFloat16)
NONZERO_TYPED_KERNEL(float)
// NONZERO_TYPED_KERNEL(double)
// NONZERO_TYPED_KERNEL_WITH_TYPE_NAME(std::string, string)

#undef NONZERO_9_TYPED_KERNEL
#undef NONZERO_TYPED_KERNEL

namespace {

// Below this many elements the shard bookkeeping costs more than it saves, so the
// scan runs on the calling thread.
constexpr std::ptrdiff_t kMinElementsToParallelize = 16 * 1024;

// Cap the shard count so each shard still has a worthwhile amount of work.
constexpr std::ptrdiff_t kMinElementsPerShard = 8 * 1024;

std::ptrdiff_t ComputeShardCount(concurrency::ThreadPool* tp, std::ptrdiff_t total) {
  if (tp == nullptr || total < kMinElementsToParallelize) {
    return 1;
  }

  const std::ptrdiff_t max_useful_shards = total / kMinElementsPerShard;
  const std::ptrdiff_t degree = concurrency::ThreadPool::DegreeOfParallelism(tp);
  return std::max<std::ptrdiff_t>(1, std::min(degree, max_useful_shards));
}

// Counts the elements of data[first, last) that compare unequal to T{}.
// The comparison is written exactly as the previous implementation had it so that
// the treatment of NaN (selected) and -0.0 (not selected) is unchanged.
template <typename T>
int64_t CountNonZero(const T* data, std::ptrdiff_t first, std::ptrdiff_t last) {
  int64_t count = 0;
  for (std::ptrdiff_t i = first; i < last; ++i) {
    if (data[i] != T{}) {
      ++count;
    }
  }

  return count;
}

// Writes the coordinates of the non-zero elements of data[first, last) into the
// [rank, num_non_zero] row-major output, starting at column `col`.
//
// Coordinates are derived per selected element rather than tracked incrementally for
// every element. That keeps the scan itself a plain comparison loop the compiler can
// vectorize, and confines the divisions to the elements actually written out. The
// low-rank cases are spelled out because they cover almost all real inputs and let the
// compiler emit a single division for the quotient/remainder pair.
template <typename T>
void FillNonZeroCoordinates(const T* data, std::ptrdiff_t first, std::ptrdiff_t last,
                            const TensorShape& shape, int64_t num_non_zero,
                            int64_t col, int64_t* output) {
  const size_t rank = shape.NumDimensions();

  switch (rank) {
    case 1: {
      for (std::ptrdiff_t i = first; i < last; ++i) {
        if (data[i] != T{}) {
          output[col] = static_cast<int64_t>(i);
          ++col;
        }
      }
      break;
    }
    case 2: {
      const int64_t dim1 = shape[1];
      int64_t* row0 = output;
      int64_t* row1 = output + num_non_zero;
      for (std::ptrdiff_t i = first; i < last; ++i) {
        if (data[i] != T{}) {
          const int64_t flat = static_cast<int64_t>(i);
          row0[col] = flat / dim1;
          row1[col] = flat % dim1;
          ++col;
        }
      }
      break;
    }
    case 3: {
      const int64_t dim1 = shape[1];
      const int64_t dim2 = shape[2];
      const int64_t dim12 = dim1 * dim2;
      int64_t* row0 = output;
      int64_t* row1 = output + num_non_zero;
      int64_t* row2 = output + 2 * num_non_zero;
      for (std::ptrdiff_t i = first; i < last; ++i) {
        if (data[i] != T{}) {
          const int64_t flat = static_cast<int64_t>(i);
          const int64_t axis0 = flat / dim12;
          const int64_t rem = flat - axis0 * dim12;
          row0[col] = axis0;
          row1[col] = rem / dim2;
          row2[col] = rem % dim2;
          ++col;
        }
      }
      break;
    }
    default: {
      for (std::ptrdiff_t i = first; i < last; ++i) {
        if (data[i] != T{}) {
          int64_t rem = static_cast<int64_t>(i);
          for (size_t axis = rank; axis-- > 0;) {
            const int64_t dim = shape[axis];
            output[axis * num_non_zero + col] = rem % dim;
            rem /= dim;
          }
          ++col;
        }
      }
      break;
    }
  }
}

}  // namespace

template <typename T>
Status NonZero<T>::Compute(OpKernelContext* context) const {
  const auto X = context->Input<Tensor>(0);
  ORT_ENFORCE(X, "X input is required!");

  const auto& X_shape = X->Shape();
  assert(X_shape.Size() >= 0);

  const T* data = X->Data<T>();

  // A scalar has no coordinates, but the output is reported with a single coordinate
  // row to stay compatible with the shape this kernel has always produced. See
  // https://github.com/onnx/onnx/issues/2428 for the ambiguity in the spec.
  if (X_shape.IsScalar()) {
    const int64_t num_non_zero = (*data != T{}) ? 1 : 0;
    Tensor* const Y = context->Output(0, {1, num_non_zero});
    ORT_ENFORCE(Y, "failed to get first output!");
    if (num_non_zero != 0) {
      *Y->MutableData<int64_t>() = 0;
    }

    return Status::OK();
  }

  // Size() reports -1 if any dimension is negative. Narrowing through size_t keeps that
  // case throwing, as it did when the previous implementation sized its scratch buffer.
  const std::ptrdiff_t total = narrow<std::ptrdiff_t>(narrow<size_t>(X_shape.Size()));
  const int64_t coordinate_size = narrow<int64_t>(X_shape.NumDimensions());

  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  const std::ptrdiff_t num_shards = ComputeShardCount(tp, total);

  // First pass: count the non-zero elements so the output can be allocated at its exact
  // size, and so each shard knows which columns of the output it owns. Shard boundaries
  // come from PartitionWork, which is deterministic, so the second pass reproduces the
  // same ranges and the output stays in ascending flat-index order.
  InlinedVector<int64_t> shard_counts(narrow<size_t>(num_shards), 0);
  if (num_shards == 1) {
    shard_counts[0] = CountNonZero(data, 0, total);
  } else {
    concurrency::ThreadPool::TrySimpleParallelFor(
        tp, num_shards,
        [&](std::ptrdiff_t shard) {
          const auto work = concurrency::ThreadPool::PartitionWork(shard, num_shards, total);
          shard_counts[narrow<size_t>(shard)] = CountNonZero(data, work.start, work.end);
        });
  }

  // Turn the per-shard counts into the starting output column of each shard.
  int64_t num_non_zero = 0;
  for (auto& count : shard_counts) {
    const int64_t shard_start = num_non_zero;
    num_non_zero += count;
    count = shard_start;
  }

  Tensor* const Y = context->Output(0, {coordinate_size, num_non_zero});
  ORT_ENFORCE(Y, "failed to get first output!");

  if (num_non_zero == 0) {
    return Status::OK();
  }

  // Second pass: write the coordinates straight into the transposed output layout.
  // Shards write disjoint column ranges, so no synchronization is needed.
  int64_t* output = Y->MutableData<int64_t>();
  if (num_shards == 1) {
    FillNonZeroCoordinates(data, 0, total, X_shape, num_non_zero, 0, output);
  } else {
    concurrency::ThreadPool::TrySimpleParallelFor(
        tp, num_shards,
        [&](std::ptrdiff_t shard) {
          const auto work = concurrency::ThreadPool::PartitionWork(shard, num_shards, total);
          FillNonZeroCoordinates(data, work.start, work.end, X_shape, num_non_zero,
                                 shard_counts[narrow<size_t>(shard)], output);
        });
  }

  return Status::OK();
}

}  // namespace onnxruntime
