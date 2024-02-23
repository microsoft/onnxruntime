// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef USE_MPI
#include "orttraining/core/framework/communication/mpi/mpi_utilities.h"

namespace onnxruntime {
namespace training {

MPI_Datatype GetMPIDataType(MLDataType data_type) {
  if (data_type == DataTypeImpl::GetType<uint8_t>()) {
    return MPI_UINT8_T;
  } else if (data_type == DataTypeImpl::GetType<int8_t>()) {
    return MPI_INT8_T;
  } else if (data_type == DataTypeImpl::GetType<uint16_t>() ||
             data_type == DataTypeImpl::GetType<MLFloat16>()) {
    return MPI_UINT16_T;
  } else if (data_type == DataTypeImpl::GetType<int16_t>()) {
    return MPI_INT16_T;
  } else if (data_type == DataTypeImpl::GetType<int32_t>()) {
    return MPI_INT32_T;
  } else if (data_type == DataTypeImpl::GetType<int64_t>()) {
    return MPI_INT64_T;
  } else if (data_type == DataTypeImpl::GetType<float>()) {
    return MPI_FLOAT;
  } else if (data_type == DataTypeImpl::GetType<double>()) {
    return MPI_DOUBLE;
  } else if (data_type == DataTypeImpl::GetType<bool>()) {
    return MPI_C_BOOL;
  } else if (data_type == DataTypeImpl::GetType<uint8_t>()) {
    return MPI_BYTE;
  } else {
    ORT_THROW("Unsupported MPI data type.");
  }
}

int GetMPIRank(MPI_Comm comm) {
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  return rank;
}

int GetMPISize(MPI_Comm comm) {
  int size = 0;
  MPI_Comm_size(comm, &size);
  return size;
}

}  // namespace training
}  // namespace onnxruntime
#endif  // USE_MPI
