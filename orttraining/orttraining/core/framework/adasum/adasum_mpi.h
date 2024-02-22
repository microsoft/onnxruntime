// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "orttraining/core/framework/adasum/adasum_interface.h"
#include "orttraining/core/framework/communication/mpi/mpi_utilities.h"

namespace onnxruntime {
namespace training {

#ifdef USE_MPI
class AdasumMPI : public AdasumInterface<MPI_Comm> {
 public:
  bool IsAdasumInitialized() override;

  void InitializeVHDDReductionComms(WorkerGroupType worker_group = WorkerGroupType::GlobalParallel) override;

  const MPI_Comm* GetReductionComms() override { return reduction_comms_.get()->data(); }

 protected:
  void PointToPointSendRecv(void* input_data_buffer,
                            int64_t input_buffer_bytes,
                            void* output_data_buffer,
                            int64_t output_buffer_bytes,
                            MLDataType data_type, int dst_src_rank,
                            int tag, MPI_Comm communicator) override;

  int GetRankWithComm(MPI_Comm local_comm) override;

  int GetSizeWithComm(MPI_Comm comm) override;

  void SumAllreduceWithComm(void* data,
                            int num_elements, MLDataType data_type,
                            MPI_Comm comm) override;

 private:
  // MPI communicators used to do adasum VHDD.
  std::unique_ptr<std::vector<MPI_Comm>> reduction_comms_;
  // Flag to indicate if reduction comms have been initialized.
  bool reduction_comms_initialized_ = false;

  // Chunk size for MPI send/recv in Adasum allreduce. Sometimes MPI
  // benefits from a smaller chunk size.
  static constexpr int64_t adasum_mpi_chunk_size_ = 1 << 30;
};
#endif  // USE_MPI
}  // namespace training
}  // namespace onnxruntime
