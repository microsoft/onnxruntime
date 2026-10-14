// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Scatter
#include <algorithm>
#include <atomic>
#include <type_traits>
#include <core/common/safeint.h>

#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/framework/element_type_lists.h"
#include "core/framework/op_kernel.h"
#include "core/framework/op_kernel_type_control_utils.h"
#include "core/platform/threadpool.h"
#include "core/providers/common.h"
#include "core/providers/op_kernel_type_control.h"
#if defined(ENABLE_TRAINING_OPS)
#include "orttraining/training_ops/cpu/tensor/gather_elements_grad_impl.h"
#endif

namespace onnxruntime {

namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Scatter, Input, 0, element_type_lists::All);

ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, ScatterElements, Input, 0, element_type_lists::All);
}  // namespace op_kernel_type_control

using EnabledScatterDataTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Scatter, Input, 0);

using EnabledScatterElementsDataTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, ScatterElements, Input, 0);

template <typename EnabledDataTypes>
class Scatter final : public OpKernel {
 public:
  explicit Scatter(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK(),
                "Missing/Invalid 'axis' attribute value");

    // 'reduction' attribute was added in opset 16.
    // its default value is 'none' in which case the op behaves the same as before opset 16.
    if (!info.GetAttr<std::string>("reduction", &reduction_).IsOK()) {
      reduction_ = "none";
    }
  }

  ~Scatter() = default;
  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t axis_;
  std::string reduction_;
};

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Scatter,
    9, 10,
    KernelDefBuilder()
        .MayInplace(0, 0)
        .TypeConstraint("T",
                        BuildKernelDefConstraintsFromTypeList<EnabledScatterDataTypes>())
        .TypeConstraint("Tind", BuildKernelDefConstraints<int32_t, int64_t>()),
    Scatter<EnabledScatterDataTypes>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    ScatterElements,
    11,
    12,
    KernelDefBuilder()
        .MayInplace(0, 0)
        .TypeConstraint("T",
                        BuildKernelDefConstraintsFromTypeList<EnabledScatterElementsDataTypes>())
        .TypeConstraint("Tind", BuildKernelDefConstraints<int32_t, int64_t>()),
    Scatter<EnabledScatterElementsDataTypes>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    ScatterElements,
    13,
    15,
    KernelDefBuilder()
        .MayInplace(0, 0)
        .TypeConstraint("T",
                        BuildKernelDefConstraintsFromTypeList<EnabledScatterElementsDataTypes>())
        .TypeConstraint("Tind", BuildKernelDefConstraints<int32_t, int64_t>()),
    Scatter<EnabledScatterElementsDataTypes>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    ScatterElements,
    16,
    17,
    KernelDefBuilder()
        .MayInplace(0, 0)
        .TypeConstraint("T",
                        BuildKernelDefConstraintsFromTypeList<EnabledScatterElementsDataTypes>())
        .TypeConstraint("Tind", BuildKernelDefConstraints<int32_t, int64_t>()),
    Scatter<EnabledScatterElementsDataTypes>);

ONNX_CPU_OPERATOR_KERNEL(
    ScatterElements,
    18,
    KernelDefBuilder()
        .MayInplace(0, 0)
        .TypeConstraint("T",
                        BuildKernelDefConstraintsFromTypeList<EnabledScatterElementsDataTypes>())
        .TypeConstraint("Tind", BuildKernelDefConstraints<int32_t, int64_t>()),
    Scatter<EnabledScatterElementsDataTypes>);

template <class T>
struct Func_Assignment {
  void operator()(T* a, const T* b) const {
    (*a) = (*b);
  }
};

template <class T>
struct Func_Add {
  void operator()(T* a, const T* b) const {
    (*a) += (*b);
  }
};

template <>
struct Func_Add<bool> {
  void operator()(bool* a, const bool* b) const {
    (*a) |= (*b);
  }
};

template <>
struct Func_Add<MLFloat16> {
  void operator()(MLFloat16*, const MLFloat16*) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: MLFloat16 data type is not supported with ScatterElements opset 16 when reduction is 'add'.");
  }
};

template <>
struct Func_Add<BFloat16> {
  void operator()(BFloat16*, const BFloat16*) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: BFloat16 data type is not supported with ScatterElements opset 16 when reduction is 'add'.");
  }
};

template <class T>
struct Func_Mul {
  void operator()(T* a, const T* b) const {
    (*a) *= (*b);
  }
};

template <>
struct Func_Mul<bool> {
  void operator()(bool* a, const bool* b) const {
    (*a) &= (*b);
  }
};

template <>
struct Func_Mul<std::string> {
  void operator()(std::string*, const std::string*) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: string data type is not supported with ScatterElements opset 16 when reduction is 'mul'.");
  }
};

template <>
struct Func_Mul<MLFloat16> {
  void operator()(MLFloat16*, const MLFloat16*) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: MLFloat16 data type is not supported with ScatterElements opset 16 when reduction is 'mul'.");
  }
};

template <>
struct Func_Mul<BFloat16> {
  void operator()(BFloat16*, const BFloat16*) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: BFloat16 data type is not supported with ScatterElements opset 16 when reduction is 'mul'.");
  }
};

template <class T>
struct Func_Min {
  void operator()(T* a, const T* b) const {
    (*a) = (*a) < (*b) ? (*a) : (*b);
  }
};

template <>
struct Func_Min<bool> {
  void operator()(bool*, const bool*) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: bool data type is not supported with ScatterElements opset 18 when reduction is 'min'.");
  }
};

template <>
struct Func_Min<std::string> {
  void operator()(std::string*, const std::string*) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: string data type is not supported with ScatterElements opset 18 when reduction is 'min'.");
  }
};

template <>
struct Func_Min<BFloat16> {
  void operator()(BFloat16*, const BFloat16*) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: BFloat16 data type is not supported with ScatterElements opset 18 when reduction is 'min'.");
  }
};

template <class T>
struct Func_Max {
  void operator()(T* a, const T* b) const {
    (*a) = (*a) > (*b) ? (*a) : (*b);
  }
};

template <>
struct Func_Max<bool> {
  void operator()(bool*, const bool*) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: bool data type is not supported with ScatterElements opset 18 when reduction is 'max'.");
  }
};

template <>
struct Func_Max<std::string> {
  void operator()(std::string*, const std::string*) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: string data type is not supported with ScatterElements opset 18 when reduction is 'max'.");
  }
};

template <>
struct Func_Max<BFloat16> {
  void operator()(BFloat16*, const BFloat16*) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: BFloat16 data type is not supported with ScatterElements opset 18 when reduction is 'max'.");
  }
};

// Reads indices straight out of the input tensor, widening and normalizing on the way.
// Keeping the element type here rather than in ScatterData's template arguments avoids
// instantiating the whole scatter twice; the branch is loop-invariant in practice and costs
// far less than the int64_t buffer it replaces.
struct ScatterIndices {
  const int32_t* as_int32{nullptr};
  const int64_t* as_int64{nullptr};
  int64_t axis_dim_limit{0};

  static ScatterIndices Create(const Tensor& indices_input, int64_t axis_dim_limit) {
    ScatterIndices indices;
    indices.axis_dim_limit = axis_dim_limit;
    if (indices_input.GetElementType() == utils::ToTensorProtoElementType<int32_t>()) {
      indices.as_int32 = indices_input.Data<int32_t>();
    } else {
      indices.as_int64 = indices_input.Data<int64_t>();
    }
    return indices;
  }

  // Raw value, for comparing two indices without caring what they normalize to.
  int64_t Raw(int64_t i) const {
    return as_int32 != nullptr ? static_cast<int64_t>(as_int32[i]) : as_int64[i];
  }

  // Only valid once ValidateIndices has accepted the tensor.
  int64_t operator[](int64_t i) const {
    const int64_t idx = Raw(i);
    return idx < 0 ? idx + axis_dim_limit : idx;
  }
};

// Checks every index against the axis bound. The normalized values are not stored: the scatter
// reads the indices tensor directly and normalizes as it goes, which avoids allocating and
// filling an int64_t per element before any real work starts.
template <class TIndex>
Status ValidateIndices(
    const Tensor& data_input, const Tensor& indices_input, int64_t axis,
    concurrency::ThreadPool* tp) {
  const auto& input_data_shape = data_input.Shape();
  const auto* indices_data_raw = indices_input.Data<TIndex>();
  const auto num_indices = indices_input.Shape().Size();
  const auto axis_dim_limit = input_data_shape[narrow<size_t>(axis)];

  // When multiple indices are out-of-bounds, the reported index is nondeterministic
  // (whichever thread wins the CAS). This is acceptable—we only need to report that
  // validation failed and provide one example of a bad index.
  std::atomic<bool> found_error{false};
  std::atomic<int64_t> first_bad_idx{0};

  concurrency::ThreadPool::TryParallelFor(
      tp, narrow<std::ptrdiff_t>(num_indices), 1.0,
      [&](std::ptrdiff_t first, std::ptrdiff_t last) {
        for (std::ptrdiff_t i = first; i < last; ++i) {
          const int64_t idx = static_cast<int64_t>(indices_data_raw[i]);
          if (idx < -axis_dim_limit || idx >= axis_dim_limit) {
            bool expected = false;
            if (found_error.compare_exchange_strong(expected, true)) {
              first_bad_idx.store(idx, std::memory_order_relaxed);
            }
            return;
          }
        }
      });

  if (found_error.load()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "indices element out of data bounds, idx=", first_bad_idx.load(),
                           " must be within the inclusive range [", -axis_dim_limit,
                           ",", axis_dim_limit - 1, "]");
  }

  return Status::OK();
}

// True when, inside every slice of `inner_size` consecutive indices, all of them are equal.
// That is the layout an Expand of an [N, 1] index tensor to [N, C] produces, and it means the
// scatter can move whole contiguous runs instead of individual strided elements.
// Each worker stops as soon as it sees a mismatch, and other workers stop at their next slice
// boundary, so indices that are not row-broadcast are rejected after a small amount of work
// rather than after a full pass.
inline bool IndicesAreConstantAlongInner(const ScatterIndices& indices, int64_t num_indices,
                                         int64_t inner_size, concurrency::ThreadPool* tp) {
  const int64_t num_slices = num_indices / inner_size;

  std::atomic<bool> constant{true};
  concurrency::ThreadPool::TryParallelFor(
      tp, narrow<std::ptrdiff_t>(num_slices), static_cast<double>(inner_size),
      [&](std::ptrdiff_t first, std::ptrdiff_t last) {
        for (std::ptrdiff_t slice = first; slice < last; ++slice) {
          if (!constant.load(std::memory_order_relaxed)) {
            return;
          }
          const int64_t base = static_cast<int64_t>(slice) * inner_size;
          const int64_t first_index = indices.Raw(base);
          for (int64_t i = 1; i < inner_size; ++i) {
            if (indices.Raw(base + i) != first_index) {
              constant.store(false, std::memory_order_relaxed);
              return;
            }
          }
        }
      });

  return constant.load(std::memory_order_relaxed);
}

// Copies the data input over to the output before the updates are applied. This runs before any
// scatter work and covers the whole tensor, so on a large tensor a single-threaded memcpy leaves
// the operator thread pool idle through what is often the most expensive part of the node.
// Small tensors stay on a plain memcpy, where splitting the work would cost more than it saves.
inline void CopyDataToOutput(void* dst, const void* src, size_t bytes, concurrency::ThreadPool* tp) {
  constexpr size_t kMinBytesToParallelize = 1 << 20;
  constexpr size_t kMinBytesPerShard = 256 * 1024;

  const int degree = concurrency::ThreadPool::DegreeOfParallelism(tp);
  if (tp == nullptr || degree <= 1 || bytes < kMinBytesToParallelize) {
    memcpy(dst, src, bytes);
    return;
  }

  const std::ptrdiff_t num_shards =
      std::min<std::ptrdiff_t>(degree, narrow<std::ptrdiff_t>(bytes / kMinBytesPerShard));
  if (num_shards <= 1) {
    memcpy(dst, src, bytes);
    return;
  }

  auto* dst_bytes = static_cast<char*>(dst);
  const auto* src_bytes = static_cast<const char*>(src);
  concurrency::ThreadPool::TrySimpleParallelFor(
      tp, num_shards, [&](std::ptrdiff_t shard) {
        const auto work =
            concurrency::ThreadPool::PartitionWork(shard, num_shards, narrow<std::ptrdiff_t>(bytes));
        memcpy(dst_bytes + work.start, src_bytes + work.start,
               static_cast<size_t>(work.end - work.start));
      });
}

template <class Tdata, typename FuncT>
Status ScatterData(
    const FuncT& func,
    const Tensor* data_input, const Tensor* indices_input, const Tensor* updates_input, int64_t axis,
    concurrency::ThreadPool* tp,
    Tensor* data_output) {
  const TensorShape& input_data_shape = data_input->Shape();

  const auto input_elements = input_data_shape.Size();
  const auto total_input_bytes = data_input->SizeInBytes();

  const auto num_indices = indices_input->Shape().Size();

  const auto* src_base = static_cast<const Tdata*>(data_input->DataRaw());
  auto* dst_base = static_cast<Tdata*>(data_output->MutableDataRaw());

  // We allow runtime to re-use input for output. If input/output Tensor* are the same
  // we do not copy
  if (src_base != dst_base) {
    if (std::is_same<Tdata, std::string>::value) {
      const auto* str_begin = data_input->Data<std::string>();
      const std::string* str_end = str_begin + input_elements;
      auto* dst = data_output->MutableData<std::string>();
      std::copy(str_begin, str_end, dst);
    } else {
      CopyDataToOutput(static_cast<void*>(dst_base), static_cast<const void*>(src_base),
                       total_input_bytes, tp);
    }
  }

  // Now poke updates

  const auto& upd_shape = updates_input->Shape();
  const auto num_dims = input_data_shape.NumDimensions();
  ORT_RETURN_IF_NOT(num_dims > 0, "ScatterElements op: input tensor must have at least one dimension");

  if (num_indices == 0) {
    return Status::OK();
  }

  const auto* update_data = static_cast<const Tdata*>(updates_input->DataRaw());

  // Compute outer_size (product of dims before axis) and inner_size (product of dims after axis).
  // For ScatterElements with axis=a:
  //   output[i0]...[indices[i0..iN]][...][iN] = updates[i0][...][iN]
  // Work units identified by (outer_idx, inner_idx) are completely independent:
  // they never write to the same output element, even with reductions.
  // This allows safe parallelization over outer_size * inner_size work units.
  int64_t outer_size = 1;
  for (int64_t i = 0; i < axis; ++i) {
    outer_size *= upd_shape[narrow<size_t>(i)];
  }
  const int64_t axis_size = upd_shape[narrow<size_t>(axis)];
  int64_t inner_size = 1;
  for (size_t i = narrow<size_t>(axis) + 1; i < num_dims; ++i) {
    inner_size *= upd_shape[i];
  }

  // Compute strides for the input/output tensor
  std::vector<int64_t> input_strides(num_dims);
  input_strides.back() = 1;
  if (num_dims > 1) {
    for (auto i = int64_t(num_dims - 2); i >= 0; --i) {
      input_strides[narrow<size_t>(i)] = input_data_shape[SafeInt<size_t>(i) + 1] * input_strides[SafeInt<size_t>(i) + 1];
    }
  }

  // Compute strides for the updates/indices tensor
  std::vector<int64_t> upd_strides(num_dims);
  upd_strides.back() = 1;
  if (num_dims > 1) {
    for (auto i = int64_t(num_dims - 2); i >= 0; --i) {
      upd_strides[narrow<size_t>(i)] = upd_shape[SafeInt<size_t>(i) + 1] * upd_strides[SafeInt<size_t>(i) + 1];
    }
  }

  const int64_t total_work_units = outer_size * inner_size;
  const int64_t axis_dim_limit = input_data_shape[narrow<size_t>(axis)];
  const ScatterIndices indices_data = ScatterIndices::Create(*indices_input, axis_dim_limit);
  const int64_t input_axis_stride = input_strides[narrow<size_t>(axis)];
  const int64_t upd_axis_stride = upd_strides[narrow<size_t>(axis)];

  // Fast path for indices that are constant across the dimensions after the axis.
  //
  // The generic loop below gives each work unit a single inner index, so it walks one column of
  // the updates with a stride of inner_size. When the indices within a slice are all equal, the
  // whole slice lands on one contiguous run of the output instead, which turns that strided walk
  // into a sequential one the compiler can vectorize.
  //
  // The run is only contiguous if the updates and the data agree on every dimension after the
  // axis; ScatterElements permits the updates to be smaller there, in which case an inner
  // coordinate does not map to the same offset in both tensors and this path does not apply.
  bool inner_dims_match = true;
  for (size_t d = narrow<size_t>(axis) + 1; d < num_dims; ++d) {
    if (upd_shape[d] != input_data_shape[d]) {
      inner_dims_match = false;
      break;
    }
  }

  // Below roughly a cache line the strided access the generic loop performs is already local,
  // so restricting the fast path keeps it from trading away parallelism for nothing.
  constexpr int64_t kMinContiguousRun = 8;

  if (inner_dims_match && inner_size >= kMinContiguousRun &&
      IndicesAreConstantAlongInner(indices_data, num_indices, inner_size, tp)) {
    // Split the contiguous run into blocks so the work still spreads across threads. Blocks are
    // kept reasonably wide so each one is worth vectorizing.
    constexpr int64_t kMinBlockElements = 16;
    const int64_t max_blocks = std::max<int64_t>(1, inner_size / kMinBlockElements);
    const int64_t degree = std::max<int64_t>(1, concurrency::ThreadPool::DegreeOfParallelism(tp));
    const int64_t num_blocks = std::min(degree, max_blocks);
    const int64_t blocked_work_units = outer_size * num_blocks;

    // Every work unit owns a distinct (outer, inner-range) region of the output, so no two of
    // them touch the same element. Within a unit the axis is walked in ascending order, which is
    // the order the generic loop applies updates in, so a destination that receives several
    // updates still accumulates them in the same sequence.
    concurrency::ThreadPool::TryParallelFor(
        tp, narrow<std::ptrdiff_t>(blocked_work_units), static_cast<double>(axis_size * inner_size / num_blocks),
        [&](std::ptrdiff_t first, std::ptrdiff_t last) {
          for (std::ptrdiff_t work_idx = first; work_idx < last; ++work_idx) {
            const int64_t outer_idx = static_cast<int64_t>(work_idx) / num_blocks;
            const int64_t block_idx = static_cast<int64_t>(work_idx) % num_blocks;
            const auto block =
                concurrency::ThreadPool::PartitionWork(narrow<std::ptrdiff_t>(block_idx),
                                                       narrow<std::ptrdiff_t>(num_blocks),
                                                       narrow<std::ptrdiff_t>(inner_size));

            // Offsets contributed by the dimensions before the axis. The dimensions after it
            // contribute the position within the run, which is identical in both tensors because
            // inner_dims_match was checked above.
            int64_t dst_base_offset = 0;
            int64_t upd_base_offset = 0;
            int64_t outer_remain = outer_idx;
            for (int64_t d = axis - 1; d >= 0; --d) {
              const auto dim_size = upd_shape[narrow<size_t>(d)];
              const auto coord = outer_remain % dim_size;
              outer_remain /= dim_size;
              dst_base_offset += coord * input_strides[narrow<size_t>(d)];
              upd_base_offset += coord * upd_strides[narrow<size_t>(d)];
            }

            for (int64_t a = 0; a < axis_size; ++a) {
              const int64_t upd_row = upd_base_offset + a * upd_axis_stride;
              // Constant across the run, so the first element speaks for all of them.
              const int64_t axis_idx = indices_data[upd_row + block.start];
              const int64_t dst_row = dst_base_offset + axis_idx * input_axis_stride;
              for (std::ptrdiff_t k = block.start; k < block.end; ++k) {
                func(dst_base + dst_row + k, update_data + upd_row + k);
              }
            }
          }
        });

    return Status::OK();
  }
  // Parallelize over independent work units.
  // Each work unit processes axis_size elements along the scatter axis.
  // Cost per unit is proportional to axis_size (number of scatter ops per work unit).
  concurrency::ThreadPool::TryParallelFor(
      tp, narrow<std::ptrdiff_t>(total_work_units), static_cast<double>(axis_size),
      [&](std::ptrdiff_t first, std::ptrdiff_t last) {
        for (std::ptrdiff_t work_idx = first; work_idx < last; ++work_idx) {
          // Decompose work_idx into outer_idx and inner_idx
          const int64_t outer_idx = static_cast<int64_t>(work_idx) / inner_size;
          const int64_t inner_idx = static_cast<int64_t>(work_idx) % inner_size;

          // Compute the base offset in the output for dimensions outside the axis.
          // For dims before axis: determined by outer_idx
          // For dims after axis: determined by inner_idx
          int64_t dst_base_offset = 0;
          int64_t outer_remain = outer_idx;
          for (int64_t d = axis - 1; d >= 0; --d) {
            const auto dim_size = upd_shape[narrow<size_t>(d)];
            const auto coord = outer_remain % dim_size;
            outer_remain /= dim_size;
            dst_base_offset += coord * input_strides[narrow<size_t>(d)];
          }
          int64_t inner_remain = inner_idx;
          for (int64_t d = int64_t(num_dims) - 1; d > axis; --d) {
            const auto dim_size = upd_shape[narrow<size_t>(d)];
            const auto coord = inner_remain % dim_size;
            inner_remain /= dim_size;
            dst_base_offset += coord * input_strides[narrow<size_t>(d)];
          }

          // Compute the base index into the updates/indices flat array
          int64_t upd_base_offset = 0;
          outer_remain = outer_idx;
          for (int64_t d = axis - 1; d >= 0; --d) {
            const auto dim_size = upd_shape[narrow<size_t>(d)];
            const auto coord = outer_remain % dim_size;
            outer_remain /= dim_size;
            upd_base_offset += coord * upd_strides[narrow<size_t>(d)];
          }
          inner_remain = inner_idx;
          for (int64_t d = int64_t(num_dims) - 1; d > axis; --d) {
            const auto dim_size = upd_shape[narrow<size_t>(d)];
            const auto coord = inner_remain % dim_size;
            inner_remain /= dim_size;
            upd_base_offset += coord * upd_strides[narrow<size_t>(d)];
          }

          // Process axis_size elements along the axis
          for (int64_t a = 0; a < axis_size; ++a) {
            const int64_t upd_flat_idx = upd_base_offset + a * upd_axis_stride;
            const int64_t axis_idx = indices_data[upd_flat_idx];
            const int64_t dst_offset = dst_base_offset + axis_idx * input_axis_stride;
            func(dst_base + dst_offset, update_data + upd_flat_idx);
          }
        }
      });

  return Status::OK();
}

template <typename TData>
struct ScatterDataDispatchTarget {
  Status operator()(const Tensor* data_input, const Tensor* indices_input, const Tensor* updates_input, int64_t axis,
                    const std::string& reduction, concurrency::ThreadPool* tp, Tensor* data_output) const {
    if (reduction == "add")
      return ScatterData<TData>(
          Func_Add<TData>(), data_input, indices_input, updates_input, axis, tp, data_output);
    else if (reduction == "mul")
      return ScatterData<TData>(
          Func_Mul<TData>(), data_input, indices_input, updates_input, axis, tp, data_output);
    else if (reduction == "min")
      return ScatterData<TData>(
          Func_Min<TData>(), data_input, indices_input, updates_input, axis, tp, data_output);
    else if (reduction == "max")
      return ScatterData<TData>(
          Func_Max<TData>(), data_input, indices_input, updates_input, axis, tp, data_output);
    else  // if (reduction == "none")
      return ScatterData<TData>(
          Func_Assignment<TData>(), data_input, indices_input, updates_input, axis, tp, data_output);
  }
};

template <typename EnabledDataTypes>
Status Scatter<EnabledDataTypes>::Compute(OpKernelContext* context) const {
  const auto* data_input = context->Input<Tensor>(0);
  const auto& input_data_shape = data_input->Shape();
  const auto axis = HandleNegativeAxis(axis_, input_data_shape.NumDimensions());

  const auto* indices_input = context->Input<Tensor>(1);
  const auto* updates_input = context->Input<Tensor>(2);

  if (data_input->DataType() != updates_input->DataType()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "data type is different from updates type");
  }

  auto indices_dims = indices_input->Shape().GetDims();
  auto updates_dims = updates_input->Shape().GetDims();
  if (indices_dims.size() != updates_dims.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Indices and updates must have the same rank");
  }

  for (size_t i = 0; i < indices_dims.size(); ++i) {
    if (indices_dims[i] != updates_dims[i]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Indices vs updates dimensions differs at position=", i,
                             " ", indices_dims[i], " vs ", updates_dims[i]);
    }
  }

  // According to the spec the rank of ind/upd shall be the same as input(data)
  // and we also want to make sure that the dimensions of the of the ind/upd do not
  // exceed that of the input
  auto input_dims = input_data_shape.GetDims();
  if (input_dims.size() != indices_dims.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Indices must have the same rank as Input. Indices rank=",
                           indices_dims.size(), ". Input rank=", input_dims.size());
  }

  for (size_t i = 0; i < input_dims.size(); ++i) {
    // For all axes except the axis of interest, make sure that the corresponding 'indices' shape
    // value is within bounds of the corresponding 'data' shape.
    if (static_cast<int64_t>(i) != axis && input_dims[i] < indices_dims[i]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Indices dim=", indices_dims[i], " at pos=", i,
                             " is greater than input dim=", input_dims[i]);
    }
  }

  Status status{};
  const auto index_type = indices_input->GetElementType();
  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();

  // Validate up front so an out-of-range index is reported before the output is touched.
  if (index_type == utils::ToTensorProtoElementType<int32_t>()) {
    status = ValidateIndices<int32_t>(*data_input, *indices_input, axis, tp);
  } else if (index_type == utils::ToTensorProtoElementType<int64_t>()) {
    status = ValidateIndices<int64_t>(*data_input, *indices_input, axis, tp);
  } else {
    status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Indices type is not supported.");
  }

  if (!status.IsOK()) {
    return status;
  }

  auto* data_output = context->Output(0, input_data_shape);
  const auto data_type = data_input->GetElementType();

  utils::MLTypeCallDispatcherFromTypeList<EnabledDataTypes> dispatcher{data_type};
  status = dispatcher.template InvokeRet<Status, ScatterDataDispatchTarget>(
      data_input, indices_input, updates_input, axis, this->reduction_, tp, data_output);

  return status;
}

#if defined(ENABLE_TRAINING_OPS)

namespace contrib {

template <class T>
struct Func_Add {
  void operator()(T* a, const T* b) const {
    *a = *a + *b;
  }
};

template <class Tin, class Tdata>
Status GatherElementsGradImpl(const Tensor* indices_input, const Tensor* updates_input,
                              const int64_t axis, Tensor* data_output) {
  ORT_RETURN_IF_ERROR(ValidateIndices<Tin>(*data_output, *indices_input, axis, nullptr));
  return ScatterData<Tdata>(Func_Add<Tdata>(), data_output, indices_input, updates_input, axis, nullptr,
                            data_output);
}

#define GATHER_ELEMENTS_GRAD_IMPL_SPECIALIZED(Tin, Tdata) \
  template Status GatherElementsGradImpl<Tin, Tdata>(     \
      const Tensor* indices_input,                        \
      const Tensor* updates_input,                        \
      const int64_t axis,                                 \
      Tensor* data_output)

#define GATHER_ELEMENTS_GRAD_IMPL_TDATA_SPECIALIZED(Tdata) \
  GATHER_ELEMENTS_GRAD_IMPL_SPECIALIZED(int32_t, Tdata);   \
  GATHER_ELEMENTS_GRAD_IMPL_SPECIALIZED(int64_t, Tdata);

GATHER_ELEMENTS_GRAD_IMPL_TDATA_SPECIALIZED(float)
GATHER_ELEMENTS_GRAD_IMPL_TDATA_SPECIALIZED(double)

}  // namespace contrib

#endif

}  // namespace onnxruntime
