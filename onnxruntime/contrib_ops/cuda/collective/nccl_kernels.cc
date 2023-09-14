// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <unistd.h>
#include <cstring>
#include <ctime>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "nccl_kernels.h"
#include "core/platform/env_var_utils.h"

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

namespace IPC {
#define FLLOG (std::cerr << __FILE__ << ":" << __LINE__ << " ")
#define FLLOGERRNO (std::cerr << __FILE__ << ":" << __LINE__ << " " << strerror(errno) << " ")

static int PortNumber = 18888;
int WriteOnRank0(ncclUniqueId* nccl_id, int word_size) {
  int fd = socket(AF_INET,     /* network versus AF_LOCAL */
                  SOCK_STREAM, /* reliable, bidirectional, arbitrary payload size */
                  0);          /* system picks underlying protocol (TCP) */
  if (fd < 0) {
    FLLOGERRNO << ("socket\n"); /* terminate */
    return -1;
  }

  /* bind the server's local address in memory */
  struct sockaddr_in saddr;
  memset(&saddr, 0, sizeof(saddr));                /* clear the bytes */
  saddr.sin_family = AF_INET;                      /* versus AF_LOCAL */
  saddr.sin_addr.s_addr = inet_addr("127.0.0.1");  // htonl(INADDR_ANY); /* host-to-network endian */
  saddr.sin_port = htons(PortNumber);              /* for listening */

  if (bind(fd, (struct sockaddr*)&saddr, sizeof(saddr)) < 0) {
    FLLOGERRNO << ("bind\n"); /* terminate */
    return -1;
  }
  /* listen to the socket */
  if (listen(fd, word_size) < 0) {
    FLLOGERRNO << ("bind\n"); /* terminate */
    return -1;
  }

  FLLOG << "Listening on port " << PortNumber << " for clients...\n";
  /* a server traditionally listens indefinitely */
  word_size--;  // rank 0 is not in word_size
  while (word_size-- > 0) {
    int client_fd = accept(fd, nullptr, nullptr); /* accept blocks */
    if (client_fd < 0) {
      FLLOGERRNO << ("accept\n"); /* terminate */
      return -1;
    }
    FLLOG << ("Accepted new client\n");
    if (write(client_fd, (nccl_id), sizeof(ncclUniqueId)) != sizeof(ncclUniqueId)) {
      FLLOGERRNO << ("write\n"); /* terminate */
      return -1;
    }
    close(client_fd); /* break connection */
  }
  close(fd); /* break connection */
  return 0;
}

int ReadFromRank0(ncclUniqueId* nccl_id) {
  int sockfd = socket(AF_INET,     /* versus AF_LOCAL */
                      SOCK_STREAM, /* reliable, bidirectional */
                      0);          /* system picks protocol (TCP) */
  if (sockfd < 0) {
    FLLOGERRNO << ("socket");
    return -1;
  }

  /* connect to the server: configure server's address 1st */
  struct sockaddr_in saddr;
  memset(&saddr, 0, sizeof(saddr));
  saddr.sin_family = AF_INET;
  saddr.sin_addr.s_addr = inet_addr("127.0.0.1");
  saddr.sin_port = htons(PortNumber); /* port number in big-endian */
  time_t start_time = time(0);
  int conn_ret = connect(sockfd, (struct sockaddr*)&saddr, sizeof(saddr));
  while (time(0) - start_time < 40 && conn_ret < 0) {
    FLLOGERRNO << (" retry..."); /* terminate */
    // return -1;
    sleep(1);
    conn_ret = connect(sockfd, (struct sockaddr*)&saddr, sizeof(saddr));
  }
  if (conn_ret < 0) {
    FLLOGERRNO << ("connect"); /* terminate */
    return -1;
  }
  /* Write some stuff and read the echoes. */
  FLLOG << ("Connect to server, read ncclUniqueId...");

  if (read(sockfd, (nccl_id), sizeof(ncclUniqueId)) != sizeof(ncclUniqueId)) {
    FLLOGERRNO << ("read"); /* terminate */
    return -1;
  }

  close(sockfd);
  return 0;
}
}  // namespace IPC

int IPC_Bcast(ncclUniqueId* nccl_id, int rank, int world_size) {
  if (rank == 0) {
    if (IPC::WriteOnRank0(nccl_id, world_size) != 0) {
      return (-1);
    }
  } else if (IPC::ReadFromRank0(nccl_id) != 0) {
    return (-1);
  }

  return 0;
}

#ifdef USE_MPI
static Status CreateNcclComm(int world_size, int rank, ncclComm_t* comm, bool is_launched_by_mpi) {
  // Create new NCCL communicator
  ncclUniqueId nccl_id;
  if (rank == 0) {
    NCCL_RETURN_IF_ERROR(ncclGetUniqueId(&nccl_id));
  }
  if (is_launched_by_mpi) {
    MPI_CHECK(MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));
  } else if (IPC_Bcast(&nccl_id, rank, world_size) != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "IPC_Bcast nccl_id failed with :", strerror(errno));
  }

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
  bool is_launched_by_mpi = true;
  if (world_size_ < 1) {
    is_launched_by_mpi = false;
    world_size_ = ParseEnvironmentVariableWithDefault<int32_t>("LOCAL_WORLD_SIZE", -1);
    rank_ = ParseEnvironmentVariableWithDefault<int32_t>("LOCAL_RANK", -1);
    ORT_ENFORCE(world_size_ != -1 && rank_ != -1);
  }

  // Initialize global Parallel Group NCCL Communicator
  auto ret = CreateNcclComm(world_size_, rank_, &comm_, is_launched_by_mpi);
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

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
