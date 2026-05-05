// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/affine_grid.h"

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/providers/op_kernel_type_control.h"
#include "core/util/math_cpuonly.h"
#include <Eigen/Dense>
#include "core/common/eigen_common_wrapper.h"

namespace onnxruntime {

#define REGISTER_KERNEL_TYPED(T)                                         \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                        \
      AffineGrid,                                                        \
      20,                                                                \
      T,                                                                 \
      KernelDefBuilder()                                                 \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<int64_t>()), \
      AffineGrid<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)

template <typename T>
void generate_base_grid_2d(int64_t H, int64_t W, bool align_corners, Eigen::Matrix<T, Eigen::Dynamic, 2>& base_grid) {
  using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  VectorT row_vec = VectorT::LinSpaced(static_cast<Eigen::Index>(W), static_cast<T>(-1), static_cast<T>(1));
  if (!align_corners) {
    row_vec = row_vec * static_cast<T>(W - 1) / static_cast<T>(W);
  }
  VectorT col_vec = VectorT::LinSpaced(static_cast<Eigen::Index>(H), static_cast<T>(-1), static_cast<T>(1));
  if (!align_corners) {
    col_vec = col_vec * static_cast<T>(H - 1) / static_cast<T>(H);
  }

  base_grid.resize(static_cast<Eigen::Index>(H * W), 2);
  for (Eigen::Index j = 0; j < H; j++) {
    for (Eigen::Index i = 0; i < W; i++) {
      base_grid.row(j * static_cast<Eigen::Index>(W) + i) << row_vec(i), col_vec(j);
    }
  }
}

template <typename T>
void generate_base_grid_3d(int64_t D, int64_t H, int64_t W, bool align_corners, Eigen::Matrix<T, Eigen::Dynamic, 3>& base_grid) {
  using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  VectorT row_vec = VectorT::LinSpaced(static_cast<Eigen::Index>(W), static_cast<T>(-1), static_cast<T>(1));
  if (!align_corners) {
    row_vec = row_vec * static_cast<T>(W - 1) / static_cast<T>(W);
  }
  VectorT col_vec = VectorT::LinSpaced(static_cast<Eigen::Index>(H), static_cast<T>(-1), static_cast<T>(1));
  if (!align_corners) {
    col_vec = col_vec * static_cast<T>(H - 1) / static_cast<T>(H);
  }
  VectorT slice_vec = VectorT::LinSpaced(static_cast<Eigen::Index>(D), static_cast<T>(-1), static_cast<T>(1));
  if (!align_corners) {
    slice_vec = slice_vec * static_cast<T>(D - 1) / static_cast<T>(D);
  }

  base_grid.resize(static_cast<Eigen::Index>(D * H * W), 3);
  for (Eigen::Index k = 0; k < D; k++) {
    for (Eigen::Index j = 0; j < H; j++) {
      for (Eigen::Index i = 0; i < W; i++) {
        base_grid.row(k * static_cast<Eigen::Index>(H * W) + j * static_cast<Eigen::Index>(W) + i) << row_vec(i), col_vec(j), slice_vec(k);
      }
    }
  }
}

template <typename T>
void affine_grid_generator_2d(const Tensor* theta, const Eigen::Matrix<T, 2, Eigen::Dynamic>& base_grid_transposed, int64_t batch_num, int64_t H, int64_t W, Tensor* grid) {
  const Eigen::StorageOptions option = Eigen::RowMajor;
  auto theta_batch_offset = batch_num * 2 * 3;
  const T* theta_data = theta->Data<T>() + theta_batch_offset;
  const Eigen::Matrix<T, 2, 2, option> theta_R{{theta_data[0], theta_data[1]}, {theta_data[3], theta_data[4]}};  // 2x2
  const Eigen::Array<T, 2, 1> theta_T(theta_data[2], theta_data[5]);                                             // 2x1

  const auto grid_batch_offset = static_cast<size_t>(SafeInt<size_t>(batch_num) * H * W * 2);
  T* grid_data = grid->MutableData<T>() + grid_batch_offset;
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 2, option>> grid_matrix(
      grid_data, static_cast<size_t>(SafeInt<size_t>(H) * W), 2);
  grid_matrix = ((theta_R * base_grid_transposed).array().colwise() + theta_T).matrix().transpose();  // ((2x2 * 2xN).array().colwise() + 2x1).matrix().transpose() => Nx2
}

template <typename T>
void affine_grid_generator_3d(const Tensor* theta, const Eigen::Matrix<T, 3, Eigen::Dynamic>& base_grid_transposed, int64_t batch_num, int64_t D, int64_t H, int64_t W, Tensor* grid) {
  const Eigen::StorageOptions option = Eigen::RowMajor;
  auto theta_batch_offset = batch_num * 3 * 4;
  const T* theta_data = theta->Data<T>() + theta_batch_offset;

  const Eigen::Matrix<T, 3, 3, option> theta_R{
      {theta_data[0], theta_data[1], theta_data[2]},
      {theta_data[4], theta_data[5], theta_data[6]},
      {theta_data[8], theta_data[9], theta_data[10]}};  // 3x3

  const Eigen::Array<T, 3, 1> theta_T(theta_data[3], theta_data[7], theta_data[11]);  // 3x1

  const auto grid_batch_offset = static_cast<size_t>(SafeInt<size_t>(batch_num) * D * H * W * 3);
  T* grid_data = grid->MutableData<T>() + grid_batch_offset;
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 3, option>> grid_matrix(
      grid_data, static_cast<size_t>(SafeInt<size_t>(D) * H * W), 3);
  grid_matrix = ((theta_R * base_grid_transposed).array().colwise() + theta_T).matrix().transpose();
}

template <typename T>
Status AffineGrid<T>::Compute(OpKernelContext* context) const {
  const Tensor* theta = context->Input<Tensor>(0);
  const TensorShape& theta_shape = theta->Shape();
  if (theta_shape.NumDimensions() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "AffineGrid : Input theta tensor dimension is not 3");
  }

  const Tensor* size = context->Input<Tensor>(1);
  const TensorShape& size_shape = size->Shape();
  const int64_t* size_data = size->Data<int64_t>();

  if (size_shape.GetDims()[0] == 4) {
    int64_t N = size_data[0], H = size_data[2], W = size_data[3];

    ORT_RETURN_IF(N != theta_shape[0],
                  "AffineGrid: size[0] (", N, ") must equal theta batch dimension (", theta_shape[0], ")");
    ORT_RETURN_IF(theta_shape[1] != 2 || theta_shape[2] != 3,
                  "AffineGrid: theta shape must be [N, 2, 3] for 2D, got [",
                  theta_shape[0], ", ", theta_shape[1], ", ", theta_shape[2], "]");
    ORT_RETURN_IF(H <= 0, "AffineGrid: size[2] (H=", H, ") must be positive");
    ORT_RETURN_IF(W <= 0, "AffineGrid: size[3] (W=", W, ") must be positive");

    TensorShape grid_shape{N, H, W, 2};
    auto grid = context->Output(0, grid_shape);

    Eigen::Matrix<T, Eigen::Dynamic, 2> base_grid;
    generate_base_grid_2d(H, W, align_corners_, base_grid);
    Eigen::Matrix<T, 2, Eigen::Dynamic> base_grid_transposed = base_grid.transpose();

    std::function<void(ptrdiff_t)> fn = [theta, base_grid_transposed, H, W, grid](ptrdiff_t batch_num) {
      affine_grid_generator_2d(theta, base_grid_transposed, batch_num, H, W, grid);
    };

    concurrency::ThreadPool::TryBatchParallelFor(context->GetOperatorThreadPool(), narrow<size_t>(N), std::move(fn), 0);
  } else if (size_shape.GetDims()[0] == 5) {
    int64_t N = size_data[0], D = size_data[2], H = size_data[3], W = size_data[4];

    ORT_RETURN_IF(N != theta_shape[0],
                  "AffineGrid: size[0] (", N, ") must equal theta batch dimension (", theta_shape[0], ")");
    ORT_RETURN_IF(theta_shape[1] != 3 || theta_shape[2] != 4,
                  "AffineGrid: theta shape must be [N, 3, 4] for 3D, got [",
                  theta_shape[0], ", ", theta_shape[1], ", ", theta_shape[2], "]");
    ORT_RETURN_IF(D <= 0, "AffineGrid: size[2] (D=", D, ") must be positive");
    ORT_RETURN_IF(H <= 0, "AffineGrid: size[3] (H=", H, ") must be positive");
    ORT_RETURN_IF(W <= 0, "AffineGrid: size[4] (W=", W, ") must be positive");

    TensorShape grid_shape{N, D, H, W, 3};
    auto grid = context->Output(0, grid_shape);

    Eigen::Matrix<T, Eigen::Dynamic, 3> base_grid;
    generate_base_grid_3d(D, H, W, align_corners_, base_grid);
    Eigen::Matrix<T, 3, Eigen::Dynamic> base_grid_transposed = base_grid.transpose();

    std::function<void(ptrdiff_t)> fn = [theta, base_grid_transposed, D, H, W, grid](ptrdiff_t batch_num) {
      affine_grid_generator_3d(theta, base_grid_transposed, batch_num, D, H, W, grid);
    };

    concurrency::ThreadPool::TryBatchParallelFor(context->GetOperatorThreadPool(), narrow<size_t>(N), std::move(fn), 0);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "AffineGrid : Invalidate size - length of size should be 4 or 5.");
  }
  return Status::OK();
}
}  // namespace onnxruntime
