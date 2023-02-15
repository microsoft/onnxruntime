// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(USE_MPI)
#define OMPI_SKIP_MPICXX 1  // See https://github.com/open-mpi/ompi/issues/5157
#include <mpi.h>
#undef OMPI_SKIP_MPICXX

namespace onnxruntime {

#if defined(USE_MPI)
#define MPI_CHECK(condition)  \
  do {                        \
    int error = (condition);  \
    ORT_ENFORCE(              \
        error == MPI_SUCCESS, \
        "MPI Error at: ",     \
        __FILE__,             \
        ":",                  \
        __LINE__,             \
        ": ",                 \
        error);               \
  } while (0)
#endif
}  // namespace onnxruntime
#endif
