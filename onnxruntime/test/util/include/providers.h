// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cpu/cpu_provider_factory.h"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_factory.h"
#endif
#ifdef USE_DNNL
#include "core/providers/dnnl/dnnl_provider_factory.h"
#endif
#ifdef USE_NUPHAR
#include "core/providers/nuphar/nuphar_provider_factory.h"
#endif
#ifdef USE_OPENVINO
#include "core/providers/openvino/openvino_provider_factory.h"
#endif
#ifdef USE_NNAPI
#include "core/providers/nnapi/nnapi_provider_factory.h"
#endif
#ifdef USE_COREML
#include "core/providers/coreml/coreml_provider_factory.h"
#endif
#ifdef USE_DML
#include "core/providers/dml/dml_provider_factory.h"
#endif
#ifdef USE_ACL
#include "core/providers/acl/acl_provider_factory.h"
#endif
#ifdef USE_ARMNN
#include "core/providers/armnn/armnn_provider_factory.h"
#endif
#ifdef USE_MIGRAPHX
#include "core/providers/migraphx/migraphx_provider_factory.h"
#endif
