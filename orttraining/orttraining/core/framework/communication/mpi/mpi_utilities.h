// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/tensor.h"
#include "core/framework/op_kernel.h"

#include <mpi.h>

namespace onnxruntime {
namespace training {

MPI_Datatype GetMPIDataType (MLDataType data_type);

int GetMPIRank(MPI_Comm comm);

int GetMPISize(MPI_Comm comm);

}  // namespace training
}  // namespace onnxruntime
