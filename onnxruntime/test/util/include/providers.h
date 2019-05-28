// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cpu/cpu_provider_factory.h"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_factory.h"
#endif
#ifdef USE_MKLDNN
#include "core/providers/mkldnn/mkldnn_provider_factory.h"
#endif
#ifdef USE_NGRAPH
#include "core/providers/ngraph/ngraph_provider_factory.h"
#endif
#ifdef USE_NUPHAR
#include "core/providers/nuphar/nuphar_provider_factory.h"
#endif
#if USE_BRAINSLICE
#include "core/providers/brainslice/brainslice_provider_factory.h"
#endif
#ifdef USE_TENSORRT
#include "core/providers/tensorrt/tensorrt_provider_factory.h"
#endif
