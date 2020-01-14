// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
<<<<<<< HEAD
/*
// TODO(weixing):  PER_THREAD_DEFAULT_STREAM is disabled for running BERT-large fast

#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#endif
*/
=======
>>>>>>> c767e264c52c3bac2c319b630d37f541f4d2a677
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <curand.h>
#include <cudnn.h>

#ifdef USE_NCCL
#include <nccl.h>
#endif