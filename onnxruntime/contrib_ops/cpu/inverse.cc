// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/platform/threadpool.h"
#include "core/util/math_cpuonly.h"
#include "Eigen/src/Core/Map.h"
#include "Eigen/LU"
#include <functional>

namespace onnxruntime {
namespace contrib {
class Inverse final : public OpKernel {
 public:
  explicit Inverse(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* ctx) const override;

 private:
  template <typename T>
  struct ComputeImpl;
};

ONNX_OPERATOR_KERNEL_EX(
    Inverse,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefConstraints<float, double, MLFloat16>()),
    Inverse);

template <typename T>
using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
struct Inverse::ComputeImpl {
  void operator()(const Tensor* input, Tensor* output,
                  int64_t batch_num, int64_t rows, int64_t cols) const {
    auto batch_offset = batch_num * rows * cols;
    const auto* input_data = input->Data<T>() + batch_offset;
    auto* output_data = output->MutableData<T>() + batch_offset;

    Eigen::Map<const MatrixT<T>> input_matrix(input_data, rows, cols);
    Eigen::Map<MatrixT<T>> output_matrix(output_data, rows, cols);
    output_matrix = input_matrix.inverse();
  }
};

template <>
struct Inverse::ComputeImpl<MLFloat16> {
  void operator()(const Tensor* input, Tensor* output,
                  int64_t batch_num, int64_t rows, int64_t cols) const {
    auto batch_offset = batch_num * rows * cols;
    // Direct cast to half as it just as MLFloat16 containes only uint16_t
    const auto* input_data = reinterpret_cast<const Eigen::half*>(input->Data<MLFloat16>() + batch_offset);
    auto* output_data = reinterpret_cast<Eigen::half*>(output->MutableData<MLFloat16>() + batch_offset);

    Eigen::Map<const MatrixT<Eigen::half>> input_matrix(input_data, rows, cols);
    Eigen::Map<MatrixT<Eigen::half>> output_matrix(output_data, rows, cols);
    output_matrix = input_matrix.inverse();
  }
};

Status Inverse::Compute(OpKernelContext* ctx) const {
  const auto& input = ctx->Input<Tensor>(0);
  const auto elem_type = input->GetElementType();
  const auto& input_shape = input->Shape();
  const auto num_dim = input_shape.NumDimensions();
  auto* output = ctx->Output(0, input_shape);

  int64_t num_batches = 1;
  const int64_t rows = input_shape.GetDims()[num_dim - 2];
  const int64_t cols = input_shape.GetDims()[num_dim - 1];
  if (num_dim > 2) {
    num_batches = input_shape.SizeToDimension(num_dim - 2);
  }

  std::function<void(ptrdiff_t)> fn = [elem_type, input, output, rows, cols](ptrdiff_t batch_num) {
    utils::MLTypeCallDispatcher<float, double, MLFloat16> t_disp(elem_type);
    t_disp.Invoke<ComputeImpl>(input, output, batch_num, rows, cols);
  };

  concurrency::ThreadPool::TryBatchParallelFor(ctx->GetOperatorThreadPool(), num_batches, std::move(fn), 0);

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
