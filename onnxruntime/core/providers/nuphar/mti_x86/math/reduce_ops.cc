// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/mti_x86/math/reduce_ops.h"

#include "core/codegen/common/common.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/mti/tensor/pad_ops.h"
#include "core/codegen/mti/tensor/reshape_ops.h"
#include "core/providers/nuphar/compiler/x86/scheduler/nuphar_scheduler.h"
#include <topi/reduction.h>

namespace onnxruntime {
namespace nuphar {

using FReduce = std::function<tvm::Expr(tvm::Expr source, const tvm::Array<tvm::IterVar>& axis)>;

// A special vectorization friendly value reduction along with non-last dims
// axes are sorted, and cannot contain last
// It prefers but not require last dim is multiple of vector_width
tvm::Tensor ReduceValueWithoutSplit(const tvm::Tensor& X,
                                    FReduce func,
                                    const std::vector<int64_t>& axes,
                                    bool keep_dims,
                                    int32_t fuse_dim = 0,
                                    const std::string& name = "ReduceValueWithoutSplit") {
  auto input_shape = X->shape;  // [n1, n2, n3, ..., n(d-1), nd]
  tvm::Array<tvm::IterVar> out_reduce_axis;
  for (const auto& t : axes) {
    out_reduce_axis.push_back(tvm::reduce_axis(tvm::Range(0, input_shape[t]), name + "_k_" + std::to_string(t)));  // k
  }

  tvm::Array<tvm::Expr> output_shape;
  if (keep_dims) {
    size_t j = 0;
    for (size_t i = 0; i < input_shape.size(); ++i) {
      if (gsl::narrow_cast<int64_t>(i) == axes[j]) {
        output_shape.push_back(1);
        if (gsl::narrow_cast<int64_t>(j) < (gsl::narrow_cast<int64_t>(axes.size()) - 1))
          j++;
      } else {
        output_shape.push_back(input_shape[i]);
      }
    }
  } else {
    size_t j = 0;
    for (size_t i = 0; i < input_shape.size(); ++i) {
      if (gsl::narrow_cast<int64_t>(i) == axes[j]) {
        if (gsl::narrow_cast<int64_t>(j) < (gsl::narrow_cast<int64_t>(axes.size()) - 1))
          j++;
      } else {
        output_shape.push_back(input_shape[i]);
      }
    }
  }

  auto l_out = [&](const tvm::Array<tvm::Var>& indices) {
    tvm::Array<tvm::Expr> eval_range;
    size_t j = 0;
    size_t l = 0;
    if (keep_dims) {
      for (size_t i = 0; i < input_shape.size(); ++i) {
        if (gsl::narrow_cast<int64_t>(i) == axes[j]) {
          eval_range.push_back(indices[l] + out_reduce_axis[j]->var);
          if (gsl::narrow_cast<int64_t>(j) < (gsl::narrow_cast<int64_t>(axes.size()) - 1))
            j++;
        } else {
          eval_range.push_back(indices[l]);
        }
        l++;
      }
    } else {
      for (size_t i = 0; i < input_shape.size(); ++i) {
        if (gsl::narrow_cast<int64_t>(i) == axes[j]) {
          eval_range.push_back(out_reduce_axis[j]->var);
          if (gsl::narrow_cast<int64_t>(j) < (gsl::narrow_cast<int64_t>(axes.size()) - 1))
            j++;
        } else {
          eval_range.push_back(indices[l]);
          l++;
        }
      }
    }
    return func(X(eval_range), out_reduce_axis);
  };

  tvm::Map<std::string, tvm::NodeRef> attrs;
  attrs.Set(kNupharVReduceFuseDim, tvm::Expr(fuse_dim));
  attrs.Set(kNupharScheduleNoParallel, tvm::Expr(true));
  return tvm::compute(output_shape, l_out, name + "_regular_reduce", kNupharVReduce, attrs);
}

// A special vectorization friendly value reduction along with last dims
// It prefers last dim is multiple of vector_width
// If not, it will perform Pad to force it. Using last_dim_aligned to bypass it.
tvm::Tensor ReduceValueWithSplitLast(const tvm::Tensor& X,
                                     FReduce func,
                                     const std::vector<int64_t>& axes,
                                     bool keep_dims,
                                     const tvm::Expr& pad_value,
                                     const int32_t vector_size,
                                     bool last_dim_aligned,
                                     int32_t fuse_dim = 0,
                                     const std::string& name = "ReduceValueWithSplitLast") {
  tvm::Tensor Z;
  if (last_dim_aligned) {
    Z = X;
  } else {
    Z = tvm_codegen::PadLastDim(X, vector_size, pad_value);
  }

  auto input_shape = Z->shape;  // [n1, n2, n3, ..., n(d-1), nd]
  ORT_ENFORCE(input_shape.size() > 0);
  ORT_ENFORCE(axes.size() > 0);

  tvm::Array<tvm::IterVar> out_reduce_axis;
  for (size_t i = 0; i < axes.size() - 1; ++i) {
    out_reduce_axis.push_back(tvm::reduce_axis(tvm::Range(0, input_shape[axes[i]]), name + "_k_" + std::to_string(axes[i])));  // k
  }
  tvm::Expr last_end = (input_shape[axes.back()] + vector_size - 1) / vector_size;         // d/w for noPad
  out_reduce_axis.push_back(tvm::reduce_axis(tvm::Range(0, last_end), name + "_k_last"));  // k

  tvm::Array<tvm::Expr> output_shape;  // output_shape as [n1, n2, n3, ..., n(d-1) w], w as vector_size

  auto input_shape_rank_minus_1 = input_shape.size() - 1;
  size_t j = 0;
  if (keep_dims) {
    for (size_t i = 0; i < input_shape_rank_minus_1; ++i) {
      if (gsl::narrow_cast<int64_t>(i) == axes[j]) {
        output_shape.push_back(1);
        j++;
      } else {
        output_shape.push_back(input_shape[i]);
      }
    }
  } else {
    for (size_t i = 0; i < input_shape_rank_minus_1; ++i) {
      if (gsl::narrow_cast<int64_t>(i) == axes[j]) {
        j++;
      } else {
        output_shape.push_back(input_shape[i]);
      }
    }
  }
  output_shape.push_back(vector_size);

  auto l_head = [&](const tvm::Array<tvm::Var>& indices) {
    tvm::Array<tvm::Expr> eval_range;
    size_t j = 0;
    size_t l = 0;
    if (keep_dims) {
      for (size_t i = 0; i < input_shape_rank_minus_1; ++i) {
        if (gsl::narrow_cast<int64_t>(i) == axes[j]) {
          eval_range.push_back(indices[l] + out_reduce_axis[j]->var);
          j++;
        } else {
          eval_range.push_back(indices[l]);
        }
        l++;
      }
    } else {
      for (size_t i = 0; i < input_shape_rank_minus_1; ++i) {
        if (gsl::narrow_cast<int64_t>(i) == axes[j]) {
          eval_range.push_back(out_reduce_axis[j]->var);
          j++;
        } else {
          eval_range.push_back(indices[l]);
          l++;
        }
      }
    }
    eval_range.push_back((out_reduce_axis[j]->var) * vector_size + indices[l]);
    return func(Z(eval_range), out_reduce_axis);
  };

  tvm::Map<std::string, tvm::NodeRef> attrs_head;
  attrs_head.Set(kNupharVReduceFuseDim, tvm::Expr(gsl::narrow_cast<int32_t>(output_shape.size()) - 1));

  auto head_tensor = tvm::compute(output_shape, l_head, name + "_split_lastdim_reduce", kNupharVReduce, attrs_head);  //[n1, n2, n3, ..., n(d-1), w]
  return ReduceValueWithoutSplit(head_tensor, func, {gsl::narrow_cast<int64_t>(head_tensor->shape.size()) - 1}, keep_dims, fuse_dim, name + "_final");
}

// A special vectorization friendly value reduction
// It will detect reduce axes to decide calling which reduce implementation
// A reduce all could become a combination of reshape, reduce 1D, and then reshape.
// The max function calls are 3.
// The last reshape won't be fused if the reduce is that last node of the graph. [TODO] FIXME
tvm::Tensor ReduceValue(const tvm::Tensor& X,
                        FReduce func,
                        const std::vector<int64_t>& axes,
                        bool keep_dims,
                        const tvm::Expr& pad_value,
                        int32_t vector_size,
                        bool last_dim_aligned,
                        int32_t fuse_dim = 0,
                        const std::string& name = "ReduceValue") {
  //reduce all, reshape and call ReduceValueWithSplitLast
  if ((axes.size() == 0 && X->shape.size() > 1) ||
      (axes.size() == X->shape.size() && axes.size() > 1)) {
    auto input_shape = X->shape;
    std::vector<int64_t> axes_new;

    if (fuse_dim == gsl::narrow_cast<int64_t>(X->shape.size()) - 1) {
      // a special case no need reshape
      for (int64_t i = 0; i < gsl::narrow_cast<int64_t>(X->shape.size()); ++i) {
        axes_new.push_back(i);
      }
      return ReduceValueWithSplitLast(X,
                                      func,
                                      axes_new,
                                      keep_dims,
                                      pad_value,
                                      vector_size,
                                      last_dim_aligned,
                                      fuse_dim,
                                      name);
    } else {
      tvm::Array<tvm::Expr> reshape_dim;
      tvm::Expr tail_size = 1;

      for (size_t i = 0; i < gsl::narrow_cast<size_t>(fuse_dim); ++i) {
        reshape_dim.push_back(input_shape[i]);
        axes_new.push_back(i);
      }

      for (size_t i = gsl::narrow_cast<size_t>(fuse_dim); i < input_shape.size(); ++i) {
        tail_size *= input_shape[i];
      }
      reshape_dim.push_back(tail_size);
      axes_new.push_back(fuse_dim);

      auto X_reshape = tvm_codegen::Reshape(X, reshape_dim, "reshape_" + name);

      auto reduce_output = ReduceValueWithSplitLast(X_reshape,
                                                    func,
                                                    axes_new,
                                                    keep_dims,
                                                    pad_value,
                                                    vector_size,
                                                    last_dim_aligned,
                                                    fuse_dim,
                                                    name);

      if (keep_dims) {
        tvm::Array<tvm::Expr> out_shape;
        for (size_t i = 0; i < input_shape.size(); ++i) {
          out_shape.push_back(1);
        }
        return tvm_codegen::Reshape(reduce_output, out_shape, name);
      }

      return reduce_output;
    }
  }

  // axis contain last
  if (axes.back() == gsl::narrow_cast<int64_t>(X->shape.size()) - 1) {
    return ReduceValueWithSplitLast(X,
                                    func,
                                    axes,
                                    keep_dims,
                                    pad_value,
                                    vector_size,
                                    last_dim_aligned,
                                    fuse_dim,
                                    name);
  }

  //axis not contain last, call ReduceValueWithoutSplit
  return ReduceValueWithoutSplit(X, func, axes, keep_dims, fuse_dim, "reduce_first_" + name);
}

// A special vectorization friendly ReduceSum
tvm::Tensor ReduceSum(const tvm::Tensor& X,
                      const std::vector<int64_t>& axes,
                      bool keep_dims,
                      const int32_t vector_size,
                      bool last_dim_aligned,
                      int32_t fuse_dim,
                      const std::string& name) {
  return ReduceValue(X, tvm::sum, axes, keep_dims,
                     tvm::make_const(X->dtype, 0), vector_size, last_dim_aligned, fuse_dim, name);
}

// A special vectorization friendly ReduceMax
tvm::Tensor ReduceMax(const tvm::Tensor& X,
                      const std::vector<int64_t>& axes,
                      bool keep_dims,
                      const int32_t vector_size,
                      bool last_dim_aligned,
                      int32_t fuse_dim,
                      const std::string& name) {
  return ReduceValue(X, topi::MaxOp, axes, keep_dims,
                     X->dtype.min(), vector_size, last_dim_aligned, fuse_dim, name);
}

// A special vectorization friendly ReduceMin
tvm::Tensor ReduceMin(const tvm::Tensor& X,
                      const std::vector<int64_t>& axes,
                      bool keep_dims,
                      const int32_t vector_size,
                      bool last_dim_aligned,
                      int32_t fuse_dim,
                      const std::string& name) {
  return ReduceValue(X, topi::MinOp, axes, keep_dims,
                     X->dtype.max(), vector_size, last_dim_aligned, fuse_dim, name);
}

tvm::Tensor ReduceMean(const tvm::Tensor& X,
                       const std::vector<int64_t>& axes, bool keep_dims,
                       const int32_t vector_size,
                       bool last_dim_aligned,
                       int32_t fuse_dim,
                       const std::string& name) {
  tvm::Tensor sum = ReduceSum(X, axes, keep_dims, vector_size, last_dim_aligned, fuse_dim, name + "_sum");
  tvm::Expr count;
  if (axes.size() > 0) {
    count = tvm::make_const(HalideIR::Int(32), 1);
    for (auto ax : axes) {
      ax = tvm_codegen::HandleNegativeAxis(ax, X->shape.size());
      count = count * X->shape[ax];
    }
  } else {
    // by default, reduce over all axes
    count = tvm_codegen::SizeFromDimension(X->shape, 0);
  }
  return topi::divide(sum, tvm::cast(X->dtype, count), name + "_div");
}

// [WIP] a special vectorization friendly value reduction
// Keep_dim always true
tvm::Tensor ReduceValueLowest_noPad(const tvm::Tensor& X,
                                    topi::FReduce func,
                                    const int32_t vector_size,
                                    const std::string& name) {
  auto input_shape = X->shape;
  tvm::Array<tvm::Expr> head_shape;
  auto input_shape_rank = input_shape.size();
  tvm::Expr head_end = input_shape[input_shape_rank - 1] / vector_size;  // d/w
  for (size_t i = 0; i < input_shape_rank - 1; ++i) {
    head_shape.push_back(input_shape[i]);
  }
  head_shape.push_back(vector_size);
  // head_shape as [n1, w] or [w]
  tvm::Array<tvm::IterVar> head_reduce_axis;
  head_reduce_axis.push_back(tvm::reduce_axis(tvm::Range(0, head_end), "k_head"));

  auto l_head = [&](const tvm::Array<tvm::Var>& indices) {
    // indices as [n1, w] for 2D or [w] for 1D
    tvm::Array<tvm::Expr> eval_range;  // a linearized as head_reduce_axis*w + indices
    auto indices_rank = indices.size() - 1;
    for (size_t i = 0; i < indices_rank; ++i) {
      eval_range.push_back(indices[i]);
    }
    eval_range.push_back((head_reduce_axis[0]->var) * vector_size + indices[indices_rank]);

    return func(X(eval_range), head_reduce_axis);
  };
  //[n1, w] for 2D or [w] for 1D
  auto head_tensor = tvm::compute(head_shape, l_head, name + "_head_reduce");
  //[n1, 1] for 2D or [1] for 1D
  const auto rank_minus_one = gsl::narrow<int64_t>(input_shape_rank) - 1;
  return topi::CommReduce(head_tensor, tvm_codegen::ToTvmArrayInt({rank_minus_one}), func, true, true);
}

tvm::Tensor ReduceSumV(const tvm::Tensor& X, const int32_t vector_size, const std::string& name) {
  return ReduceValueLowest_noPad(X, tvm::sum, vector_size, name);
}

tvm::Tensor ReduceMaxV(const tvm::Tensor& X, const int32_t vector_size, const std::string& name) {
  return ReduceValueLowest_noPad(X, topi::MaxOp, vector_size, name);
}

tvm::Tensor ReduceMinV(const tvm::Tensor& X, const int32_t vector_size, const std::string& name) {
  return ReduceValueLowest_noPad(X, topi::MinOp, vector_size, name);
}

}  // namespace nuphar
}  // namespace onnxruntime
