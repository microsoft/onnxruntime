// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(USE_MPI)
#define OMPI_SKIP_MPICXX 1  // See https://github.com/open-mpi/ompi/issues/5157
#include <mpi.h>
#undef OMPI_SKIP_MPICXX

#endif
