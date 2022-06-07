// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_MPI
#include "orttraining/core/framework/adasum/adasum_mpi.h"
#include "orttraining/core/framework/communication/mpi/mpi_context.h"

namespace onnxruntime {
namespace training {

bool AdasumMPI::IsAdasumInitialized() {
  return reduction_comms_initialized_;
}

void AdasumMPI::InitializeVHDDReductionComms(WorkerGroupType worker_group) {
  int rank = GetMPIRank(MPIContext::GetInstance()
                        .GetMPIGroup(worker_group)
                        .communicator);
  int size = GetMPISize(MPIContext::GetInstance()
                        .GetMPIGroup(worker_group)
                        .communicator);
  
  // Initialize communication groups for the vector halving, distance doubling
  // (VHDD) Adasum reduction. These are used in computing dot products and
  // norms for tensors whose elements are split across multiple ranks, which
  // is required for implementing the Adasum operation. The first group
  // includes two elements: this rank and it's first VHDD neighbor. The
  // subsequent groups grow to include any ranks the previous group
  // communicates with. Thus the sizes of the groups are 2,4,8... up to the
  // size of MPI_COMM_WORLD. In essence, a reduction group includes all nodes
  // that a tensor may be split across.
  MPI_Group world_group;
  MPI_Comm_group(MPIContext::GetInstance().GetMPIGroup(worker_group).communicator,
                 &world_group);
  int nearest_power_2 = 1;
  int log_size;
  for (nearest_power_2 = 1, log_size = 0; (nearest_power_2 << 1) <= size;
        nearest_power_2 = (nearest_power_2 << 1), log_size++)
    ;
  int shift_val;
  int level;
  reduction_comms_ = std::make_unique<std::vector<MPI_Comm>>();
  reduction_comms_.get()->resize(log_size);

  auto node_rank = std::make_unique<std::vector<int>>();
  node_rank.get()->resize(size);
  for (level = 1, shift_val = 1; level < nearest_power_2;
        level = (level << 1), shift_val++) {
    int base_rank = ((rank >> shift_val) << shift_val);
    for (int i = 0; i < (level << 1); i++) {
      node_rank.get()->at(i) = (base_rank + i);
    }
    MPI_Group red_group;
    MPI_Group_incl(world_group, (level << 1), node_rank.get()->data(), &red_group);
    MPI_Comm_create_group(MPI_COMM_WORLD, red_group, 0,
                          &reduction_comms_.get()->at(shift_val - 1));
    MPI_Group_free(&red_group);
  }
  reduction_comms_initialized_ = true;
}

int AdasumMPI::GetRankWithComm(MPI_Comm comm) {
  return GetMPIRank(comm);
}

int AdasumMPI::GetSizeWithComm(MPI_Comm comm) {
  return GetMPISize(comm);
}

void AdasumMPI::SumAllreduceWithComm(void* data, int num_elements,
                                     MLDataType data_type, MPI_Comm comm) {
  int status;
  status = MPI_Allreduce(MPI_IN_PLACE, data, num_elements,
                         training::GetMPIDataType(data_type),
                         MPI_SUM, comm);
  ORT_ENFORCE(status == MPI_SUCCESS, "MPI_Allreduce failed, see MPI output for details.");
}

void AdasumMPI::PointToPointSendRecv(
    void* input_data_buffer, int64_t input_buffer_bytes,
    void* output_data_buffer, int64_t output_buffer_bytes,
    MLDataType data_type, int dst_src_rank, int tag, MPI_Comm communicator) {
  int status;
  int element_size = data_type->Size();
  int input_count = input_buffer_bytes / element_size;
  int output_count = output_buffer_bytes / element_size;
  int chunk_count =
      std::max((int)(adasum_mpi_chunk_size_ / element_size), 1);


  for (int i = 0; i < std::max(input_count, output_count); i += chunk_count) {
    status = MPI_Sendrecv((char*)input_data_buffer + i * element_size,
                          std::min(chunk_count, std::max(0, input_count - i)),
                          GetMPIDataType(data_type),
                          dst_src_rank, tag,
                          (char*)output_data_buffer + i * element_size,
                          std::min(chunk_count, std::max(0, output_count - i)),
                          GetMPIDataType(data_type),
                          dst_src_rank, tag, communicator, MPI_STATUS_IGNORE);
    if (status != MPI_SUCCESS) {
      ORT_THROW("MPI_SendRecv failed in Adasum reduction, see MPI output for details.");
    }
  }
}

} // namespace training
} // namespace onnxruntime
#endif // USE_MPI
