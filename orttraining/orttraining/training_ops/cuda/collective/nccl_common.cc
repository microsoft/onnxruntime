// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "nccl_common.h"
#include <mpi.h>

namespace onnxruntime {
namespace cuda {

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


ncclDataType_t GetNcclDataType(onnxruntime::MLDataType type) {
  if (type == DataTypeImpl::GetType<uint8_t>()) {
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
    throw std::logic_error("Tensor type not supported in NCCL.");
  }
}

static Status CreateNcclCommunicator(MPI_Group* mpi_world_group,
                                     const training::WorkerGroupType worker_group_type,
                                     ncclComm_t* group_comm) {
  auto worker_group = training::DistributedRunContext::GetInstance().GetWorkerGroup(worker_group_type);
  if (worker_group.ranks.size() == 1) {
    LOGS_DEFAULT(WARNING) << "Target group size = 1, skip creating nccl communicator. Group info: "
                          << worker_group.ToString();
    return Status::OK();
  }
  std::cout<<"* CreateNcclCommunicator: before MPI_Group_incl"<<std::endl;
  for(auto rank : worker_group.ranks){
    std::cout<<rank<<" ";
  }
  std::cout<<std::endl;
  // Create new group
  MPI_Group mpi_group;
  int result = MPI_Group_incl(*mpi_world_group, worker_group.ranks.size(), worker_group.ranks.data(), &mpi_group);
  if (result != MPI_SUCCESS) {
    ORT_THROW("MPI_Group_incl failed with error code: ", result);
  }

  std::cout<<"* CreateNcclCommunicator: before MPI_Comm_create_group"<<std::endl;
  // Create new MPI communicator
  MPI_Comm mpi_comm;
  static int32_t mpi_group_id = 0;
  result = MPI_Comm_create_group(MPI_COMM_WORLD, mpi_group, ++mpi_group_id, &(mpi_comm));
  if (result != MPI_SUCCESS) {
    ORT_THROW("MPI_Comm_create_group failed with error code: ", result);
  }
  ORT_ENFORCE(mpi_comm != MPI_COMM_NULL, "MPI communicator creation failed.");

  std::cout<<"* CreateNcclCommunicator: before MPI_Bcast"<<std::endl;
  // Create new NCCL communicator
  ncclUniqueId nccl_id;
  if (worker_group.rank_in_group == 0) {
    NCCLCHECK(ncclGetUniqueId(&nccl_id));
  }
  result = MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, mpi_comm);
  if (result != MPI_SUCCESS) {
    ORT_THROW("MPI_Bcast failed with error code: ", result);
  }
  std::cout<<"* CreateNcclCommunicator: before ncclCommInitRank"<<std::endl;
  NCCLCHECK(ncclCommInitRank(group_comm, worker_group.ranks.size(), nccl_id, worker_group.rank_in_group));

  // Clean up
  result = MPI_Group_free(&mpi_group);
  if (result != MPI_SUCCESS) {
    ORT_THROW("MPI_Group_free failed with error code: ", result);
  }
  result = MPI_Comm_free(&mpi_comm);
  if (result != MPI_SUCCESS) {
    ORT_THROW("MPI_Comm_free failed with error code: ", result);
  }
  std::cout<<"* CreateNcclCommunicator: before return"<<std::endl;
  return Status::OK();
}

NcclContext::NcclContext() {
  int is_mpi_initialized = 0;
  MPI_Initialized(&is_mpi_initialized);
  if (!is_mpi_initialized) {
    int mpi_threads_provided = 0;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &mpi_threads_provided);
  }

  // Get the group under MPI_COMM_WORLD
  MPI_Group mpi_world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &mpi_world_group);

  // Initialize Data Parallel Group NCCL Communicator
  auto ret = CreateNcclCommunicator(&mpi_world_group, training::WorkerGroupType::DataParallel,
                                    &data_group_comm_);
  ORT_ENFORCE(ret.IsOK());

  // Initialize Horizontal Model Parallel Group NCCL Communicator
  ret = CreateNcclCommunicator(&mpi_world_group, training::WorkerGroupType::HorizontalParallel,
                               &horizontal_group_comm_);
  ORT_ENFORCE(ret.IsOK());

  MPI_Group_free(&mpi_world_group);
}

ncclComm_t NcclContext::Comm(training::WorkerGroupType group_type) {
  if (training::WorkerGroupType::DataParallel == group_type) {
    return data_group_comm_;
  } else if (training::WorkerGroupType::HorizontalParallel == group_type) {
    return horizontal_group_comm_;
  }

  return nullptr;
}

NcclContext::~NcclContext() {
  if (data_group_comm_ != nullptr) {
    ncclCommDestroy(data_group_comm_);
  }

  if (horizontal_group_comm_ != nullptr) {
    ncclCommDestroy(horizontal_group_comm_);
  }

  int is_mpi_finalized = 0;
  MPI_Finalized(&is_mpi_finalized);
  if (!is_mpi_finalized) {
    MPI_Finalize();
  }
}

NcclKernel::NcclKernel(const OpKernelInfo& info) : CudaKernel(info) {
  static NcclContext context;
  nccl_ = &context;
  int64_t group_type;
  info.GetAttrOrDefault("group_type", &group_type, static_cast<int64_t>(0));
  group_type_ = static_cast<training::WorkerGroupType>(group_type);
}

}  // namespace cuda
}  // namespace onnxruntime
