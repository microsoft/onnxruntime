// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnxruntime/core/providers/cpu/cpu_provider_factory.h"

#ifdef USE_CUDA
#include "onnxruntime/core/providers/cuda/cuda_provider_factory.h"
#endif
#ifdef USE_DNNL
#include "onnxruntime/core/providers/dnnl/dnnl_provider_factory.h"
#endif
#ifdef USE_NGRAPH
#include "onnxruntime/core/providers/ngraph/ngraph_provider_factory.h"
#endif
#ifdef USE_NUPHAR
#include "onnxruntime/core/providers/nuphar/nuphar_provider_factory.h"
#endif
#ifdef USE_TENSORRT
#include "onnxruntime/core/providers/tensorrt/tensorrt_provider_factory.h"
#endif
#ifdef USE_DML
#include "onnxruntime/core/providers/dml/dml_provider_factory.h"
#endif
#ifdef USE_MIGRAPHX
#include "onnxruntime/core/providers/migraphx/migraphx_provider_factory.h"
#endif
