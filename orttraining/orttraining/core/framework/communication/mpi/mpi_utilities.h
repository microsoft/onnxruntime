// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef SHARED_PROVIDER
#include "core/framework/tensor.h"
#include "core/framework/op_kernel.h"
#endif

#include "orttraining/core/framework/communication/mpi/mpi_include.h"

namespace onnxruntime {
namespace training {
#ifdef USE_MPI
MPI_Datatype GetMPIDataType(MLDataType data_type);

int GetMPIRank(MPI_Comm comm);

int GetMPISize(MPI_Comm comm);
#endif
}  // namespace training
}  // namespace onnxruntime
