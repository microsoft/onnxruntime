// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "nccl_kernels.h"
#include "mpi_include.h"
#include "core/providers/cpu/tensor/slice.h"
#include "core/providers/cuda/tensor/slice.h"
#include "core/providers/cuda/math/matmul.h"
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

Status FuncAllReduce(
    ncclComm_t comm,
    cudaStream_t stream,
    const Tensor* input,
    Tensor* output) {
  const void* input_data = input->DataRaw();
  const auto input_shape = input->Shape();
  int64_t input_count = input_shape.Size();

  void* output_data = output->MutableDataRaw();

  ncclDataType_t dtype = GetNcclDataType(input->DataType());
  NCCL_RETURN_IF_ERROR(ncclAllReduce(input_data, output_data, input_count, dtype, ncclSum, comm, stream));
  return Status::OK();
}

static std::vector<size_t> CalculatePermToSwapAxes(
    const int64_t axis,
    const int64_t another_axis,
    const size_t rank) {
  // This is for swapping axis and another_axis.
  // NCCL's AllGather only gathers along axis 0. If gathering along another axis is needed,
  // we need to call transpose. E.g.,
  // Case 1:
  //  AllGather(axis=0)
  // Case 2:
  //  AllGather(axis=3) = Transpose(perm=[3, 1, 2, 0]) -> AllGather(axis=0) -> Transpose(perm=[3, 1, 2, 0])
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
    Tensor* output) {
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
        nccl_kernel->Stream(ctx));
  } else {
    const auto source_shape = input->Shape();
    TensorShape transposed_shape(source_shape);
    transposed_shape[0] = source_shape[axis];
    transposed_shape[axis] = source_shape[0];

    auto transposed_buffer = Tensor::Create(input->DataType(), transposed_shape, alloc);

    // swap axis 0 and axis axis
    std::vector<size_t> perm = CalculatePermToSwapAxes(0, axis, source_shape.NumDimensions());

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
        nccl_kernel->Stream(ctx));

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
    const int64_t axis) {
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

template <typename T>
DistributedKernel<T>::DistributedKernel(const OpKernelInfo& info) : NcclKernel(info) {
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


}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
