// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Include header with declaration of the function to create the execution provider factory for all enabled
// execution providers.
//
// The functions are typically implemented in
// onnxruntime/core/providers/<provider name>/<provider name>_provider_factory.cc.
//
// For execution providers that are built as separate libraries (CUDA, TensorRT, ROCm, MIGraphX, DNNL, OpenVINO)
// the functions are implemented in provider_bridge_ort.cc.

#include "core/providers/cpu/cpu_provider_factory_creator.h"

#if defined(USE_ACL)
#include "core/providers/acl/acl_provider_factory_creator.h"
#endif

#if defined(USE_ARMNN)
#include "core/providers/armnn/armnn_provider_factory_creator.h"
#endif

#if defined(USE_COREML)
#include "core/providers/coreml/coreml_provider_factory_creator.h"
#endif

#if defined(USE_CUDA)
#include "core/providers/cuda/cuda_provider_factory_creator.h"
#endif

#if defined(USE_DML)
#include "core/providers/dml/dml_provider_factory_creator.h"
#endif

#if defined(USE_DNNL)
#include "core/providers/dnnl/dnnl_provider_factory_creator.h"
#endif

#if defined(USE_MIGRAPHX)
#include "core/providers/migraphx/migraphx_provider_factory_creator.h"
#endif

#if defined(USE_NNAPI)
#include "core/providers/nnapi/nnapi_provider_factory_creator.h"
#endif

#if defined(USE_JSEP)
#include "core/providers/js/js_provider_factory_creator.h"
#endif

#if defined(USE_OPENVINO)
#include "core/providers/openvino/openvino_provider_factory_creator.h"
#endif

#if defined(USE_RKNPU)
#include "core/providers/rknpu/rknpu_provider_factory_creator.h"
#endif

#if defined(USE_ROCM)
#include "core/providers/rocm/rocm_provider_factory_creator.h"
#endif

#if defined(USE_QNN)
#include "core/providers/qnn/qnn_provider_factory_creator.h"
#endif

#if defined(USE_SNPE)
#include "core/providers/snpe/snpe_provider_factory_creator.h"
#endif

#if defined(USE_TENSORRT)
#include "core/providers/tensorrt/tensorrt_provider_factory_creator.h"
#endif

#if defined(USE_TVM)
#include "core/providers/tvm/tvm_provider_factory_creator.h"
#endif

#if !defined(ORT_MINIMAL_BUILD)
#include "core/providers/vitisai/vitisai_provider_factory_creator.h"
#endif

#if defined(USE_XNNPACK)
#include "core/providers/xnnpack/xnnpack_provider_factory_creator.h"
#endif

#if defined(USE_WEBNN)
#include "core/providers/webnn/webnn_provider_factory_creator.h"
#endif

#if defined(USE_CANN)
#include "core/providers/cann/cann_provider_factory_creator.h"
#endif

#if defined(USE_AZURE)
#include "core/providers/azure/azure_provider_factory_creator.h"
#endif
