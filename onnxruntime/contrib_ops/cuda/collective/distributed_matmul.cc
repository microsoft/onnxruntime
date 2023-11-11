// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Distributed computation.
#include "sharding.h"
#include "distributed_matmul.h"
#include "mpi_include.h"

// ORT system.
#include "core/providers/cpu/tensor/slice.h"
#include "core/providers/cuda/tensor/slice.h"
#include "core/providers/cuda/math/matmul.h"
#include "core/providers/cuda/tensor/transpose.h"
#include "core/providers/cuda/cuda_check_memory.h"

// std C++.
#include <iostream>

using namespace onnxruntime::distributed;
namespace onnxruntime {
namespace contrib {
namespace cuda {

#if defined(ORT_USE_NCCL)

static TensorShape InferMatmulOutputShape(
    const TensorShape& shape_A,
    const TensorShape& shape_B) {
  // left_shape: [M, K]
  // right_shape: [K, N]
  // output_shape: [M, N]
  ORT_ENFORCE(
      shape_A.NumDimensions() >= 2 && shape_B.NumDimensions() >= 2,
      "1-D tensor is not supported by this MatMul.");
  ORT_ENFORCE(
      shape_A.NumDimensions() == shape_B.NumDimensions(),
      "A and B must have the same rank after shape broadcasting.");
  size_t rank = shape_A.NumDimensions();
  std::vector<int64_t> shape_Y(rank, 0);
  for (size_t i = 0; i < rank; ++i) {
    const int64_t dim_A = shape_A[i];
    const int64_t dim_B = shape_B[i];
    if (i == rank - 1) {
      shape_Y[i] = dim_B;
    } else if (i == rank - 2) {
      shape_Y[i] = dim_A;
    } else if (dim_A == 1 && dim_B >= 1) {
      // dim_A is 1.
      // dim_B can be either 1 or other positive integer.
      // due ot shape broadcast.
      shape_Y[i] = dim_B;
    } else if (dim_B == 1 && dim_A >= 1) {
      // dim_B is 1.
      // dim_A can be either 1 or other positive integer.
      // due ot shape broadcast.
      shape_Y[i] = dim_A;
    } else {
      ORT_ENFORCE(dim_A == dim_B, "Broadcasting can only happen when one of dim_A and dim_B is 1.");
      shape_Y[i] = dim_A;
    }
  }
  return TensorShape(shape_Y);
};

template <typename T>
DistributedMatMul<T>::DistributedMatMul(const OpKernelInfo& info) : DistributedKernel(info) {
}

template <typename T>
Status DistributedMatMul<T>::ComputeInternal(OpKernelContext* context) const {
  const auto tensor_shard_A = context->Input<Tensor>(0);
  const auto tensor_shard_B = context->Input<Tensor>(1);
  const auto& tensor_shard_shape_A = tensor_shard_A->Shape();
  const auto& tensor_shard_shape_B = tensor_shard_B->Shape();

  auto rank_A = tensor_shard_shape_A.NumDimensions();
  auto rank_B = tensor_shard_shape_B.NumDimensions();
  // TODO(wechi): Fix MatMul(1-D, *) and MatMul(*, 1-D) cases.
  ORT_ENFORCE(rank_A >= 2 && rank_B >= 2, "Broadcast rule for 1-D tensor is different than other cases.");

  const TensorPartitionSpec& spec_A = input_shard_specs_[0];
  const TensorPartitionSpec& spec_B = input_shard_specs_[1];
  const TensorPartitionSpec& spec_Y = output_shard_specs_[0];

  const auto tensor_shape_A = ComputeOriginShape(tensor_shard_shape_A, spec_A);
  const auto tensor_shape_B = ComputeOriginShape(tensor_shard_shape_B, spec_B);

  TensorShape normalized_shape_A;
  TensorShape normalized_shape_B;
  std::tie(normalized_shape_A, normalized_shape_B) = NormalizeShapes(tensor_shape_A, tensor_shape_B);

  TensorPartitionSpec normalized_spec_A;
  TensorPartitionSpec normalized_spec_B;
  std::tie(normalized_spec_A, normalized_spec_B) = NormalizeTensorPartitionSpecs(spec_A, spec_B);

  const auto tensor_shape_Y = InferMatmulOutputShape(normalized_shape_A, normalized_shape_B);
  const auto tensor_shard_shape_Y = ComputeShardShape(tensor_shape_Y, spec_Y);

  // Case 1: A is not sharded, B is sharded.
  //  1. shard on -1: MatMul(RR, RS) -> RS
  //  2. shard on -2: MatMul(RR, SR) -> MatMul(RS, SR) + AllReduce -> RR
  //  3. shard on other axis
  if (normalized_spec_A.HasNoShard() && normalized_spec_B.HasShard()) {
    if (normalized_spec_B.OnlyShardAxis(-1)) {
      // Case 1-1
      // MatMul(RR, RS) -> RS
      ORT_ENFORCE(spec_Y.OnlyShardAxis(-1), "Not supported yet.");
      auto tensor_shard_Y = context->Output(0, tensor_shard_shape_Y);
      ORT_ENFORCE(onnxruntime::cuda::FuncMatMul<T>(
                      this, context, tensor_shard_A, tensor_shard_B, 1.0, false, false, false, false, tensor_shard_Y) == Status::OK());
    } else if (normalized_spec_B.OnlyShardAxis(-2)) {
      // Case 1-2
      // MatMul(RR, SR) -> MatMul(RS, SR) + AllReduce -> RR
      auto tmp_spec_A = CreateTensorShardSpec(spec_A.device_mesh, 0, -1, rank_A);
      auto tmp_tensor_shard_A = ReshardTensor(this, context, spec_A, tmp_spec_A, nccl_->Rank(), tensor_shard_A);

      auto tensor_shard_Y = context->Output(0, tensor_shard_shape_Y);
      ORT_ENFORCE(onnxruntime::cuda::FuncMatMul<T>(
                      this, context, tmp_tensor_shard_A.get(), tensor_shard_B, 1.0, false, false, false, false, tensor_shard_Y) == Status::OK());
      ORT_ENFORCE(FuncAllReduce(
                      nccl_->Comm(), Stream(context), tensor_shard_Y, tensor_shard_Y) == Status::OK());
    } else {
      // Case 1-3
      ORT_THROW("Not supported yet.");
    }
  }

  // Case 2: A is sharded, B is not sharded.
  //  1. shard on -1: MatMul(RS, RR) -> MatMul(RS, SR) -> MatMul(RS, SR) + AllReduce -> RR
  //  2. shard on -2: MatMul(SR, RR) -> SR
  //  3. shard on other axis: : MatMul(SRR, RRR) -> MatMul(SRR, SRR) -> SRR
  if (spec_A.HasShard() && spec_B.HasNoShard()) {
    if (spec_A.OnlyShardAxis(-1) && spec_Y.HasNoShard()) {
      // Case 2-1
      // Y is not really sharded in this case.
      auto tensor_shard_Y = context->Output(0, tensor_shard_shape_Y);

      // TODO: Support cases with multi-dimension device mesh.
      TensorPartitionSpec new_spec_B = CreateTensorShardSpec(spec_B.device_mesh, 0, -2, rank_B);
      auto tensor_reshard_B = ShardTensor(this, context, new_spec_B, nccl_->Rank(), tensor_shard_B);

      ORT_ENFORCE(onnxruntime::cuda::FuncMatMul<T>(
                      this, context, tensor_shard_A, tensor_reshard_B.get(), 1.0, false, false, false, false, tensor_shard_Y) == Status::OK());

      ORT_ENFORCE(FuncAllReduce(
                      nccl_->Comm(), Stream(context), tensor_shard_Y, tensor_shard_Y) == Status::OK());
      return Status::OK();
    } else if (spec_A.OnlyShardAxis(-2) && spec_Y.OnlyShardAxis(-2)) {
      // Case 2-2
      auto tensor_shard_Y = context->Output(0, tensor_shard_shape_Y);
      ORT_ENFORCE(onnxruntime::cuda::FuncMatMul<T>(
                      this, context, tensor_shard_A, tensor_shard_B, 1.0, false, false, false, false, tensor_shard_Y) == Status::OK());
      return Status::OK();
    } else if (spec_A.GetPartitionAxis() < gsl::narrow<int64_t>(tensor_shape_A.NumDimensions()) - 2 && normalized_spec_A.GetPartitionAxis() == spec_Y.GetPartitionAxis()) {
      // Case 2-3
      if (normalized_shape_B[normalized_spec_A.GetPartitionAxis()] == 1) {
        // Case 2-3-1.
        // B is broadcasted to along sharding axis in A.
        // E.g., MatMul(A(SRR), B(RR)) where normalized_shape_A = [2, 3, 4] and normalized_shape_B = [1, 4, 3].
        // No resharding is required.
        auto tensor_shard_Y = context->Output(0, tensor_shard_shape_Y);
        ORT_ENFORCE(onnxruntime::cuda::FuncMatMul<T>(
                        this, context, tensor_shard_A, tensor_shard_B, 1.0, false, false, false, false, tensor_shard_Y) == Status::OK());
        return Status::OK();
      } else {
        // Case 2-3-2.
        // No broadcasting
        // Allocate tensor based on shape sharded non-matrix axis.
        // MatMul(SRR, RRR) -> MatMul(SRR, SRR) -> SRR
        auto tensor_shard_Y = context->Output(0, tensor_shard_shape_Y);
        TensorPartitionSpec new_spec_B = CreateTensorShardSpec(
            spec_B.device_mesh,
            0,
            spec_A.GetNegativePartitionAxis(),
            rank_B);
        auto tensor_reshard_B = ShardTensor(this, context, new_spec_B, nccl_->Rank(), tensor_shard_B);
        ORT_ENFORCE(onnxruntime::cuda::FuncMatMul<T>(
                        this, context, tensor_shard_A, tensor_reshard_B.get(), 1.0, false, false, false, false, tensor_shard_Y) == Status::OK());
        return Status::OK();
      }
    } else {
      ORT_THROW("Not supported yet.");
    }
  }

  // Case 3: A is sharded, B is sharded.
  //  1. shard on (-1, -1): MatMul(RS, RS) -> MatMul(RS, SR) + AllReduce -> RR
  //                                       -> MatMul(RR, RS) -> RS
  //  2. shard on (-1, -2): MatMul(RS, SR) -> MatMul(RS, SR) + AllReduce -> RR
  //  3. shard on (-2, -1): MatMul(SR, RS) -> MatMul(RS, SR) + AllReduce -> RR
  //  4. shard on (-2, -2): MatMul(SR, SR) -> MatMul(RS, SR) + AllReduce -> RR
  //  5. shard on other axes
  if (spec_A.HasShard() && spec_B.HasShard()) {
    if (spec_A.OnlyShardAxis(-1) && spec_B.OnlyShardAxis(-1)) {
      // Case 3-1
      if (spec_Y.HasNoShard()) {
        // Case 3-1-1
        auto tmp_spec_B = CreateTensorShardSpec(spec_B.device_mesh, 0, -2, rank_B);
        auto tmp_tensor_shard_B = ReshardTensor(this, context, spec_B, tmp_spec_B, nccl_->Rank(), tensor_shard_B);
        auto tensor_shard_Y = context->Output(0, tensor_shard_shape_Y);
        ORT_ENFORCE(onnxruntime::cuda::FuncMatMul<T>(
                        this, context, tensor_shard_A, tmp_tensor_shard_B.get(), 1.0, false, false, false, false, tensor_shard_Y) == Status::OK());
        ORT_ENFORCE(FuncAllReduce(
                        nccl_->Comm(), Stream(context), tensor_shard_Y, tensor_shard_Y) == Status::OK());
      } else if (spec_Y.OnlyShardAxis(-1)) {
        // Cas 3-1-2
        auto tmp_spec_A = TensorPartitionSpec::CreateAllReplica(spec_A);
        auto tmp_tensor_shard_A = ReshardTensor(this, context, spec_A, tmp_spec_A, nccl_->Rank(), tensor_shard_A);
        auto tensor_shard_Y = context->Output(0, tensor_shard_shape_Y);
        ORT_ENFORCE(onnxruntime::cuda::FuncMatMul<T>(
                        this, context, tmp_tensor_shard_A.get(), tensor_shard_B, 1.0, false, false, false, false, tensor_shard_Y) == Status::OK());
      } else {
        ORT_THROW("Not supported yet.");
      }
    } else if (spec_A.OnlyShardAxis(-1) && spec_B.OnlyShardAxis(-2) && spec_Y.HasNoShard()) {
      // Case 3-2
      auto tensor_shard_Y = context->Output(0, tensor_shard_shape_Y);

      auto status = onnxruntime::cuda::FuncMatMul<T>(
          this, context, tensor_shard_A, tensor_shard_B, 1.0, false, false, false, false, tensor_shard_Y);
      ORT_ENFORCE(status == Status::OK(), status.ErrorMessage());

      status = FuncAllReduce(
          nccl_->Comm(), Stream(context), tensor_shard_Y, tensor_shard_Y);
      ORT_ENFORCE(status == Status::OK(), status.ErrorMessage());
    } else if (spec_A.OnlyShardAxis(-2) && spec_B.OnlyShardAxis(-1)) {
      // Case 3-3:
      // MatMul(SR, RS) -> MatMul(RS, SR) + AllReduce -> RR
      ORT_ENFORCE(spec_Y.HasNoShard(), "Not supported yet.");

      // A[RS]
      auto tmp_spec_A = CreateTensorShardSpec(spec_A.device_mesh, 0, -1, rank_A);
      auto tmp_tensor_shard_A = ReshardTensor(this, context, spec_A, tmp_spec_A, nccl_->Rank(), tensor_shard_A);

      // B[SR]
      auto tmp_spec_B = CreateTensorShardSpec(spec_B.device_mesh, 0, -2, rank_B);
      auto tmp_tensor_shard_B = ReshardTensor(this, context, spec_B, tmp_spec_B, nccl_->Rank(), tensor_shard_B);

      // Allocate Y[RR]
      auto tensor_shard_Y = context->Output(0, tensor_shard_shape_Y);

      // Run local MatMul(A[RS], B[SR])
      ORT_ENFORCE(onnxruntime::cuda::FuncMatMul<T>(
                      this, context, tmp_tensor_shard_A.get(), tmp_tensor_shard_B.get(), 1.0, false, false, false, false, tensor_shard_Y) == Status::OK());
      ORT_ENFORCE(FuncAllReduce(
                      nccl_->Comm(), Stream(context), tensor_shard_Y, tensor_shard_Y) == Status::OK());
    } else if (spec_A.OnlyShardAxis(-2) && spec_B.OnlyShardAxis(-2)) {
      // Case 3-4
      // MatMul(SR, SR) -> MatMul(RS, SR) + AllReduce -> RR
      ORT_ENFORCE(spec_Y.HasNoShard(), "Not supported yet.");
      auto tmp_spec_A = CreateTensorShardSpec(spec_A.device_mesh, 0, -1, rank_A);
      auto tmp_tensor_shard_A = ReshardTensor(this, context, spec_A, tmp_spec_A, nccl_->Rank(), tensor_shard_A);
      auto tensor_sard_Y = context->Output(0, tensor_shard_shape_Y);
      ORT_ENFORCE(onnxruntime::cuda::FuncMatMul<T>(
                      this, context, tmp_tensor_shard_A.get(), tensor_shard_B, 1.0, false, false, false, false, tensor_sard_Y) == Status::OK());
    } else {
      // Case 3-5
      ORT_THROW("Not supported yet.");
    }
  }

  // Case 4: A is not sharded, B is not sharded.
  //  - Easy!
  return Status::OK();
}

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedMatMul,
    kMSDomain,
    1,
    float,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .AllocateInputsContiguously()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    DistributedMatMul<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedMatMul,
    kMSDomain,
    1,
    MLFloat16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .AllocateInputsContiguously()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    DistributedMatMul<MLFloat16>);

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
