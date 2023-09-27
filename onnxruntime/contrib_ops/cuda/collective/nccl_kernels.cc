// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include "nccl_kernels.h"
#include "core/providers/cpu/tensor/slice.h"
#include "core/providers/cuda/tensor/slice.h"
#include "core/providers/cuda/math/matmul.h"
#include "mpi_include.h"
#include "core/providers/cuda/tensor/transpose.h"
#include "core/providers/cuda/cuda_check_memory.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define NCCL_RETURN_IF_ERROR(expr) ORT_RETURN_IF_ERROR(NCCL_CALL(expr))

static ncclDataType_t GetNcclDataType(onnxruntime::MLDataType type) {
  if (type == DataTypeImpl::GetType<uint8_t>()) {
    return ncclUint8;
  } else if (type == DataTypeImpl::GetType<bool>()) {
    // CUDA bool is 8-bit large.
    return ncclUint8;
  } else if (type == DataTypeImpl::GetType<int8_t>()) {
    return ncclInt8;
  } else if (type == DataTypeImpl::GetType<int32_t>()) {
    return ncclInt32;
  } else if (type == DataTypeImpl::GetType<int64_t>()) {
    return ncclInt64;
  } else if (type == DataTypeImpl::GetType<MLFloat16>()) {
    return ncclFloat16;
  } else if (type == DataTypeImpl::GetType<float>()) {
    return ncclFloat32;
  } else if (type == DataTypeImpl::GetType<double>()) {
    return ncclFloat64;
  } else {
    ORT_THROW("Tensor type not supported in NCCL.");
  }
}

#ifdef USE_MPI
static Status CreateNcclCommByMPI(int world_size, int rank, ncclComm_t* comm) {
  // Create new NCCL communicator
  ncclUniqueId nccl_id;
  if (rank == 0) {
    NCCL_RETURN_IF_ERROR(ncclGetUniqueId(&nccl_id));
  }
  MPI_CHECK(MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));
  NCCL_RETURN_IF_ERROR(ncclCommInitRank(comm, world_size, nccl_id, rank));

  return Status::OK();
}
#endif

NcclContext::NcclContext() {
#ifdef USE_MPI
  int is_mpi_initialized = 0;
  MPI_Initialized(&is_mpi_initialized);
  if (!is_mpi_initialized) {
    int mpi_threads_provided = 0;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &mpi_threads_provided);
  }

  // get world_size and rank from MPI
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

  // Initialize global Parallel Group NCCL Communicator
  auto ret = CreateNcclCommByMPI(world_size_, rank_, &comm_);
  ORT_ENFORCE(ret.IsOK());

#else
  ORT_THROW("ORT must be built with MPI to use NCCL.");
#endif
}

NcclContext::~NcclContext() {
  if (comm_ != nullptr) {
    ncclCommDestroy(comm_);
  }

#ifdef USE_MPI
  int is_mpi_finalized = 0;
  MPI_Finalized(&is_mpi_finalized);
  if (!is_mpi_finalized) {
    MPI_Finalize();
  }
#endif
}

NcclKernel::NcclKernel(const OpKernelInfo& info) : CudaKernel(info) {
  static NcclContext context;
  nccl_ = &context;
}

AllReduce::AllReduce(const OpKernelInfo& info) : NcclKernel(info) {
}

//Status FuncReshard(
//  ncclComm_t comm,
//  cudaStream_t stream,
//  const Tensor* input,
//  Tensor* output
//  TensorShard
//) {
//}

Status FuncAllReduce(
  ncclComm_t comm,
  cudaStream_t stream,
  const Tensor* input,
  Tensor* output
) {
  const void* input_data = input->DataRaw();
  const auto input_shape = input->Shape();
  int64_t input_count = input_shape.Size();

  void* output_data = output->MutableDataRaw();

  ncclDataType_t dtype = GetNcclDataType(input->DataType());
  NCCL_RETURN_IF_ERROR(ncclAllReduce(input_data, output_data, input_count, dtype, ncclSum, comm, stream));
  return Status::OK();
}

Status AllReduce::ComputeInternal(OpKernelContext* context) const {
  ncclComm_t comm = nccl_->Comm();

  auto input_tensor = context->Input<Tensor>(0);
  const void* input_data = input_tensor->DataRaw();
  const auto in_shape = input_tensor->Shape();
  int64_t input_count = in_shape.Size();

  void* output_data = context->Output(0, in_shape)->MutableDataRaw();

  ncclDataType_t dtype = GetNcclDataType(input_tensor->DataType());
  NCCL_RETURN_IF_ERROR(ncclAllReduce(input_data, output_data, input_count, dtype, ncclSum, comm, Stream(context)));
  return Status::OK();
}

AllGather::AllGather(const OpKernelInfo& info) : NcclKernel(info) {
  info.GetAttrOrDefault("group_size", &group_size_, static_cast<int64_t>(1));
  info.GetAttrOrDefault("axis", &axis_, static_cast<int64_t>(0));
  cuda_ep_ = static_cast<const CUDAExecutionProvider*>(info.GetExecutionProvider());
}

std::vector<size_t> calculate_perms(int64_t axis, int64_t another_axis, size_t rank) {
  std::vector<size_t> permutation(rank);
  std::iota(std::begin(permutation), std::end(permutation), 0);
  permutation[axis] = another_axis;
  permutation[another_axis] = axis;
  return permutation;
}

void FuncAllGather(
  const NcclKernel* nccl_kernel,
  OpKernelContext* ctx,
  const Tensor* input,
  const int64_t group_size,
  const int64_t axis,
  Tensor* output
) {
  ORT_ENFORCE(output->Shape().Size() == input->Shape().Size() * group_size, "Input and output shapes mismatch.");
  ORT_ENFORCE(group_size >= 0, "group_size should be non-negative.");
  ORT_ENFORCE(axis >= 0, "axis should be non-negative.");
  AllocatorPtr alloc;
  ORT_ENFORCE(ctx->GetTempSpaceAllocator(&alloc) == Status::OK(), "Fail to find allocator.");
  if (axis == 0) {
    const void* input_data = input->DataRaw();
    const auto input_shape = input->Shape();
    void* output_data = output->MutableDataRaw();
    ncclAllGather(
      input_data,
      output_data,
      input_shape.Size(),
      GetNcclDataType(input->DataType()),
      nccl_kernel->Comm(),
      nccl_kernel->Stream(ctx)
    );
  } else {
    const auto source_shape = input->Shape();
    TensorShape transposed_shape(source_shape);
    transposed_shape[0] = source_shape[axis];
    transposed_shape[axis] = source_shape[0];

    auto transposed_buffer = Tensor::Create(input->DataType(), transposed_shape, alloc);

    // swap axis 0 and axis axis
    std::vector<size_t> perm = calculate_perms(0, axis, source_shape.NumDimensions());

    ORT_ENFORCE(onnxruntime::cuda::Transpose::DoTranspose(nccl_kernel->GetDeviceProp(),
                                                          nccl_kernel->Stream(ctx),
                                                          nccl_kernel->GetCublasHandle(ctx),
                                                          perm, *input, *transposed_buffer) == Status::OK());

    TensorShape gathered_shape(transposed_shape);
    gathered_shape[0] = group_size * transposed_shape[0];
    auto gathered_buffer = Tensor::Create(input->DataType(), gathered_shape, alloc);

    ncclAllGather(
      transposed_buffer->DataRaw(),
      gathered_buffer->MutableDataRaw(),
      transposed_shape.Size(),
      GetNcclDataType(input->DataType()),
      nccl_kernel->Comm(),
      nccl_kernel->Stream(ctx)
    );

    ORT_ENFORCE(gathered_buffer->Shape().Size() == output->Shape().Size());
    ORT_ENFORCE(onnxruntime::cuda::Transpose::DoTranspose(nccl_kernel->GetDeviceProp(),
                                                          nccl_kernel->Stream(ctx),
                                                          nccl_kernel->GetCublasHandle(ctx),
                                                          perm, *gathered_buffer, *output) == Status::OK());
  }
}

std::unique_ptr<Tensor> FuncAllGather(
  const NcclKernel* nccl_kernel,
  OpKernelContext* ctx,
  const Tensor* input,
  const int64_t group_size,
  const int64_t axis
) {
  ORT_ENFORCE(group_size >= 0);
  ORT_ENFORCE(axis >= 0);
  AllocatorPtr alloc;
  ORT_ENFORCE(ctx->GetTempSpaceAllocator(&alloc) == Status::OK());
  TensorShape output_shape(input->Shape());
  output_shape[axis] = group_size * output_shape[axis];
  auto output = Tensor::Create(input->DataType(), output_shape, alloc);
  FuncAllGather(nccl_kernel, ctx, input, group_size, axis, output.get());
  return output;
}

Status AllGather::ComputeInternal(OpKernelContext* context) const {
  ncclComm_t comm = nccl_->Comm();

  auto input_tensor = context->Input<Tensor>(0);
  const void* input_data = input_tensor->DataRaw();
  const auto& in_shape = input_tensor->Shape();
  int64_t input_count = in_shape.Size();

  if (axis_ > 0) {
    // Need transpose
    // TODO: fuse transpose with allgather
    std::vector<size_t> permutation(in_shape.NumDimensions());
    AllocatorPtr alloc;
    auto status = context->GetTempSpaceAllocator(&alloc);
    if (!status.IsOK())
      return status;

    std::iota(std::begin(permutation), std::end(permutation), 0);

    // swap rank 0 and rank axis
    permutation[axis_] = 0;
    permutation[0] = axis_;
    std::vector<int64_t> transposed_input_dims;
    transposed_input_dims.reserve(in_shape.NumDimensions());
    for (auto e : permutation) {
      transposed_input_dims.push_back(in_shape[e]);
    }

    // Allocate a temporary tensor to hold transposed input
    auto temp_input = Tensor::Create(input_tensor->DataType(), TensorShape(transposed_input_dims), alloc);

    // Perform the transpose
    ORT_RETURN_IF_ERROR(onnxruntime::cuda::Transpose::DoTranspose(cuda_ep_->GetDeviceProp(),
                                                                  Stream(context),
                                                                  GetCublasHandle(context),
                                                                  permutation, *input_tensor, *temp_input));
    // Allocate a tempoarary buffer for all gather
    TensorShape all_gather_out_shape(transposed_input_dims);
    all_gather_out_shape[0] = group_size_ * all_gather_out_shape[0];
    auto all_gather_output = Tensor::Create(temp_input->DataType(), all_gather_out_shape, alloc);
    ncclDataType_t dtype = GetNcclDataType(temp_input->DataType());
    NCCL_RETURN_IF_ERROR(ncclAllGather(temp_input->DataRaw(),
                                       all_gather_output->MutableDataRaw(),
                                       input_count, dtype, comm, Stream(context)));
    // release temp_input
    temp_input.release();
    // transpose to output
    TensorShape out_shape(in_shape);
    out_shape[axis_] = group_size_ * out_shape[axis_];
    auto* output_tensor = context->Output(0, out_shape);

    return onnxruntime::cuda::Transpose::DoTranspose(cuda_ep_->GetDeviceProp(),
                                                     Stream(context),
                                                     GetCublasHandle(context),
                                                     permutation, *all_gather_output, *output_tensor);
  } else {
    // construct output shape
    TensorShape out_shape(in_shape);
    out_shape[axis_] = group_size_ * out_shape[axis_];

    void* output_data = context->Output(0, out_shape)->MutableDataRaw();

    ncclDataType_t dtype = GetNcclDataType(input_tensor->DataType());
    NCCL_RETURN_IF_ERROR(ncclAllGather(input_data, output_data, input_count, dtype, comm, Stream(context)));
    return Status::OK();
  }
}

AllToAll::AllToAll(const OpKernelInfo& info) : NcclKernel(info) {
  info.GetAttrOrDefault("group_size", &group_size_, static_cast<int64_t>(1));
}

Status AllToAll::ComputeInternal(OpKernelContext* context) const {
  const ncclComm_t comm = nccl_->Comm();
  auto input_tensor = context->Input<Tensor>(0);
  const char* input_data = static_cast<const char*>(input_tensor->DataRaw());
  const auto in_shape = input_tensor->Shape();
  const int64_t input_count = in_shape.Size();
  auto out_shape = in_shape;
  const int64_t element_size = input_tensor->DataType()->Size();
  const int64_t rank_stride = input_count / group_size_;
  const ncclDataType_t dtype = GetNcclDataType(input_tensor->DataType());

  char* output_data = static_cast<char*>(context->Output(0, out_shape)->MutableDataRaw());

  CheckIfMemoryOnCurrentGpuDevice(input_data);
  CheckIfMemoryOnCurrentGpuDevice(output_data);

  NCCL_RETURN_IF_ERROR(ncclGroupStart());
  for (int32_t r = 0; r < group_size_; r++) {
    NCCL_RETURN_IF_ERROR(ncclSend(input_data, rank_stride, dtype, r, comm, Stream(context)));
    NCCL_RETURN_IF_ERROR(ncclRecv(output_data, rank_stride, dtype, r, comm, Stream(context)));
    input_data += (rank_stride * element_size);
    output_data += (rank_stride * element_size);
  }
  NCCL_RETURN_IF_ERROR(ncclGroupEnd());

  return Status::OK();
}


template <typename T>
DistributedMatMul<T>::DistributedMatMul(const OpKernelInfo& info) : NcclKernel(info) {
  std::vector<int64_t> device_mesh_elements = info.GetAttrsOrDefault<int64_t>("device_mesh_elements");
  std::vector<int64_t> device_mesh_shape = info.GetAttrsOrDefault<int64_t>("device_mesh_shape");
  std::vector<std::string> input_shard_specs = info.GetAttrsOrDefault<std::string>("input_shard_specs");
  std::vector<std::string> output_shard_specs = info.GetAttrsOrDefault<std::string>("output_shard_specs");

  for (size_t i = 0; i < input_shard_specs.size(); ++i) {
    auto spec = CreateTensorPartitionSpec(input_shard_specs[i], device_mesh_shape, device_mesh_elements);
    input_shard_specs_.push_back(spec);
  }
  for (size_t i = 0; i < output_shard_specs.size(); ++i) {
    auto spec = CreateTensorPartitionSpec(output_shard_specs[i], device_mesh_shape, device_mesh_elements);
    output_shard_specs_.push_back(spec);
  }
}

//std::unique_ptr<Tensor> GatherTensor(
//  const NcclKernel* nccl_kernel,
//  OpKernelContext* ctx,
//  const TensorPartitionSpec& spec,
//  const int64_t device_id,
//  const Tensor* tensor
//) {
//  auto device_prop = nccl_kernel->GetDeviceProp();
//  const int64_t shard_axis = spec.GetPartitionAxis();
//}

TensorShape ComputeShardShape(const TensorShape source_shape, int64_t shard_axis, int64_t shard_count) {
  if (shard_axis < 0) {
    shard_axis += source_shape.NumDimensions();
  }
  TensorShape shard_shape(source_shape);
  ORT_ENFORCE(shard_axis < source_shape.NumDimensions(), "Shard axis must be less than the number of dimensions of the source tensor.");
  ORT_ENFORCE(source_shape[shard_axis] % shard_count == 0, "Number of shards must be divisible by sharded axis' dimension.");
  shard_shape[shard_axis] = source_shape[shard_axis] / shard_count;
  return shard_shape;
}

void GatherTensor(
  // Use OpKernel and do a pointer cast to unify functional calls with other eps.
  // TODO: remove CudaKernel and OpKernelContext.
  const NcclKernel* nccl_kernel,
  // Do NOT use ctx to access inputs and outputs.
  // Inputs and outputs are passed in as function arguments.
  OpKernelContext* ctx,
  const TensorPartitionSpec& spec,
  const Tensor* tensor,
  Tensor* gathered
) {
  const int64_t shard_axis = spec.GetPartitionAxis();
  const int64_t shard_count = spec.GetPartitionCount(shard_axis);

  FuncAllGather(
    nccl_kernel,
    ctx,
    tensor,
    shard_count,
    shard_axis,
    gathered
  );
}

std::unique_ptr<Tensor> GatherTensor(
  // Use OpKernel and do a pointer cast to unify functional calls with other eps.
  // TODO: remove CudaKernel and OpKernelContext.
  const NcclKernel* nccl_kernel,
  // Do NOT use ctx to access inputs and outputs.
  // Inputs and outputs are passed in as function arguments.
  OpKernelContext* ctx,
  const TensorPartitionSpec& spec,
  const Tensor* tensor
) {
  const int64_t shard_axis = spec.GetPartitionAxis();
  const int64_t shard_count = spec.GetPartitionCount(shard_axis);
  TensorShape gathered_shape(tensor->Shape());
  gathered_shape[shard_axis] *= shard_count;

  AllocatorPtr alloc;
  ORT_ENFORCE(ctx->GetTempSpaceAllocator(&alloc) == Status::OK());
  auto gathered = Tensor::Create(tensor->DataType(), gathered_shape, alloc);

  FuncAllGather(
    nccl_kernel,
    ctx,
    tensor,
    shard_count,
    shard_axis,
    gathered.get()
  );

  return gathered;
}

void ShardTensor(
  // Use OpKernel and do a pointer cast to unify functional calls with other eps.
  // TODO: remove CudaKernel and OpKernelContext.
  const NcclKernel* nccl_kernel,
  // Do NOT use ctx to access inputs and outputs.
  // Inputs and outputs are passed in as function arguments.
  OpKernelContext* ctx,
  const TensorPartitionSpec& spec,
  const int64_t device_id,
  const Tensor* tensor,
  Tensor* shard_tensor
) {
  const int64_t shard_axis = spec.GetPartitionAxis();
  const int64_t shard_count = spec.GetPartitionCount(shard_axis);
  TensorShape shard_shape = ComputeShardShape(
    tensor->Shape(),
    shard_axis,
    shard_count
  );
  const int64_t shard_dim = shard_shape[shard_axis];
  const std::vector<int64_t> starts = {shard_dim * device_id};
  const std::vector<int64_t> ends = {shard_dim * (device_id + 1)};
  const std::vector<int64_t> axes = {shard_axis};
  const std::vector<int64_t> steps = {1};

  ORT_ENFORCE(FuncSlice(
    nccl_kernel,
    ctx,
    tensor,
    starts,
    ends,
    axes,
    steps,
    shard_tensor
  ) == Status::OK());
}

std::unique_ptr<Tensor> ShardTensor(
  const NcclKernel* nccl_kernel,
  OpKernelContext* ctx,
  const TensorPartitionSpec& spec,
  const int64_t device_id,
  const Tensor* tensor
) {
  // Shard all-replica tensor per spec.

  AllocatorPtr alloc;
  ORT_ENFORCE(ctx->GetTempSpaceAllocator(&alloc) == Status::OK());

  TensorShape shard_shape = ComputeShardShape(
    tensor->Shape(),
    spec.GetPartitionAxis(),
    spec.GetPartitionCount(spec.GetPartitionAxis())
  );
  auto shard_buffer = Tensor::Create(tensor->DataType(), shard_shape, alloc);

  // Shard with pre-allocated buffer.
  ShardTensor(
    nccl_kernel,
    ctx,
    spec,
    device_id,
    tensor,
    shard_buffer.get()
  );

  return shard_buffer;
}


void ReshardTensor(
  // Use OpKernel and do a pointer cast to unify functional calls with other eps.
  // TODO: remove CudaKernel and OpKernelContext.
  const NcclKernel* nccl_kernel,
  // Do NOT use ctx to access inputs and outputs.
  // Inputs and outputs are passed in as function arguments.
  OpKernelContext* ctx,
  const TensorPartitionSpec& src_spec,
  const TensorPartitionSpec& dst_spec,
  const int64_t device_id,
  const Tensor* src,
  Tensor* dst
) {
  if (src_spec.HasShard() && dst_spec.HasNoShard()) {
    GatherTensor(
      nccl_kernel,
      ctx,
      src_spec,
      src,
      dst
    );
    return;
  } else if (src_spec.HasNoShard() && dst_spec.HasShard()) {
    ShardTensor(
      nccl_kernel,
      ctx,
      dst_spec,
      device_id,
      src,
      dst
    );
  } else if (src_spec.HasShard() && dst_spec.HasShard()) {
    int64_t src_axis = src_spec.GetPartitionAxis();
    int64_t dst_axis = dst_spec.GetPartitionAxis();
    ORT_ENFORCE(src_axis != dst_axis, "No reshard is needed. Don't call this function.");

    auto all_replica_buffer = GatherTensor(
      nccl_kernel,
      ctx,
      src_spec,
      src
    );

    ShardTensor(
      nccl_kernel,
      ctx,
      dst_spec,
      device_id,
      all_replica_buffer.get(),
      dst
    );
  } else {
    ORT_THROW("Not supported yet. Probably resharding is not needed.");
  }
}

std::unique_ptr<Tensor> ReshardTensor(
  // Use OpKernel and do a pointer cast to unify functional calls with other eps.
  // TODO: remove CudaKernel and OpKernelContext.
  const NcclKernel* nccl_kernel,
  // Do NOT use ctx to access inputs and outputs.
  // Inputs and outputs are passed in as function arguments.
  OpKernelContext* ctx,
  const TensorPartitionSpec& src_spec,
  const TensorPartitionSpec& dst_spec,
  const int64_t device_id,
  const Tensor* src
) {
  // Implement ReshardTensor but returning a unique_ptr to Tensor instead.
  const auto origin_shape = ComputeOriginShape(src->Shape(), src_spec);
  const auto dst_shape = ComputeShardShape(origin_shape, dst_spec);

  AllocatorPtr alloc;
  ORT_ENFORCE(ctx->GetTempSpaceAllocator(&alloc) == Status::OK());
  auto dst = Tensor::Create(src->DataType(), dst_shape, alloc);
  ReshardTensor(
    nccl_kernel,
    ctx,
    src_spec,
    dst_spec,
    device_id,
    src,
    dst.get()
  );
  return dst;
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
  //  1. shard on -1
  //  2. shard on -2
  //  3. shard on other axis
  if (normalized_spec_A.HasNoShard() && normalized_spec_B.HasShard()) {
    if (normalized_spec_B.OnlyShardAxis(-1)) {
      // Case 1-1
    } else if (normalized_spec_B.OnlyShardAxis(-2)) {
      // Case 1-2
    } else {
      // Case 1-3
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
        this, context, tensor_shard_A, tensor_reshard_B.get(), 1.0, false, false, false, false, tensor_shard_Y
      ) == Status::OK());

      ORT_ENFORCE(FuncAllReduce(
          nccl_->Comm(), Stream(context), tensor_shard_Y, tensor_shard_Y
      ) == Status::OK());
      return Status::OK();
    } else if (spec_A.OnlyShardAxis(-2) && spec_Y.OnlyShardAxis(-2)) {
      // Case 2-2
      auto tensor_shard_Y = context->Output(0, tensor_shard_shape_Y);
      ORT_ENFORCE(onnxruntime::cuda::FuncMatMul<T>(
        this, context, tensor_shard_A, tensor_shard_B, 1.0, false, false, false, false, tensor_shard_Y
      ) == Status::OK());
      return Status::OK();
    } else if (spec_A.GetPartitionAxis() < tensor_shape_A.NumDimensions() - 2 && spec_A.GetPartitionAxis() == spec_Y.GetPartitionAxis()) {
      // Case 2-3
      ORT_ENFORCE(rank_A == rank_B,
        "Rank of A and B must be the same until we replace spec_* with normalized_* and tensor_* with normalized_*. "
        "It doesn't work now because we cannot easily reset Tensor's shape."
      );

      // Allocate tensor based on shape sharded non-matrix axis.
      // MatMul(SRR, RRR) -> MatMul(SRR, SRR) -> SRR
      auto tensor_shard_Y = context->Output(0, tensor_shard_shape_Y);
      TensorPartitionSpec new_spec_B = CreateTensorShardSpec(spec_B.device_mesh, 0, spec_A.GetPartitionAxis(), rank_B);
      auto tensor_reshard_B = ShardTensor(this, context, new_spec_B, nccl_->Rank(), tensor_shard_B);
      ORT_ENFORCE(onnxruntime::cuda::FuncMatMul<T>(
        this, context, tensor_shard_A, tensor_reshard_B.get(), 1.0, false, false, false, false, tensor_shard_Y
      ) == Status::OK());
      return Status::OK();
    } else {
      ORT_THROW("Not supported yet.");
    }
  }

  // Case 3: A is sharded, B is sharded.
  //  1. shard on (-1, -1): MatMul(RS, RS) -> MatMul(RS, SR) + AllReduce -> RR
  //                                       -> MatMul(RR, RS) -> RS
  //  2. shard on (-1, -2): MatMul(RS, SR) -> MatMul(RS, SR) + AllReduce -> RR
  //  3. shard on (-2, -1)
  //  4. shard on (-2, -2)
  //  5. shard on other axes
  if (spec_A.HasShard() && spec_B.HasShard()) {
    if (spec_A.OnlyShardAxis(-1) && spec_B.OnlyShardAxis(-1)) {
      // Case 3-1
      if (spec_Y.HasNoShard()) {
        // Case 3-1-1
        auto tmp_spec_B = TensorPartitionSpec::Create(spec_B, -2);
        auto tmp_tensor_shard_B = ReshardTensor(this, context, spec_B, tmp_spec_B, nccl_->Rank(), tensor_shard_B);
        auto tensor_shard_Y = context->Output(0, tensor_shard_shape_Y);
        ORT_ENFORCE(onnxruntime::cuda::FuncMatMul<T>(
          this, context, tensor_shard_A, tmp_tensor_shard_B.get(), 1.0, false, false, false, false, tensor_shard_Y
        ) == Status::OK());
        ORT_ENFORCE(FuncAllReduce(
            nccl_->Comm(), Stream(context), tensor_shard_Y, tensor_shard_Y
        ) == Status::OK());
      } else if (spec_Y.OnlyShardAxis(-1)) {
        // Cas 3-1-2
        auto tmp_spec_A = TensorPartitionSpec::CreateAllReplica(spec_A);
        auto tmp_tensor_shard_A = ReshardTensor(this, context, spec_A, tmp_spec_A, nccl_->Rank(), tensor_shard_A);
        auto tensor_shard_Y = context->Output(0, tensor_shard_shape_Y);
        ORT_ENFORCE(onnxruntime::cuda::FuncMatMul<T>(
          this, context, tmp_tensor_shard_A.get(), tensor_shard_B, 1.0, false, false, false, false, tensor_shard_Y
        ) == Status::OK());
      }
    } else if (spec_A.OnlyShardAxis(-1) && spec_B.OnlyShardAxis(-2) && spec_Y.HasNoShard()) {
      // Case 3-2
      auto tensor_shard_Y = context->Output(0, tensor_shard_shape_Y);

      auto status = onnxruntime::cuda::FuncMatMul<T>(
        this, context, tensor_shard_A, tensor_shard_B, 1.0, false, false, false, false, tensor_shard_Y);
      ORT_ENFORCE(status == Status::OK(), status.ErrorMessage());

      status = FuncAllReduce(
          nccl_->Comm(), Stream(context), tensor_shard_Y, tensor_shard_Y
      );
      ORT_ENFORCE(status == Status::OK(), status.ErrorMessage());
    } else if (spec_A.OnlyShardAxis(-2) && spec_B.OnlyShardAxis(-1)) {
      // Case 3-3
    } else if (spec_A.OnlyShardAxis(-2) && spec_B.OnlyShardAxis(-2)) {
      // Case 3-4
    } else {
      // Case 3-5
    }
  }

  // Case 4: A is not sharded, B is not sharded.
  //  - Easy!
  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    AllReduce,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .VariadicAlias(0, 0)  // outputs and inputs are mapped one to one
        .AllocateInputsContiguously()
        .TypeConstraint("T", DataTypeImpl::AllIEEEFloatTensorTypes()),
    AllReduce);

ONNX_OPERATOR_KERNEL_EX(
    AllGather,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .AllocateInputsContiguously()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    AllGather);

ONNX_OPERATOR_KERNEL_EX(
    AllToAll,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .AllocateInputsContiguously()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    AllToAll);

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

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
