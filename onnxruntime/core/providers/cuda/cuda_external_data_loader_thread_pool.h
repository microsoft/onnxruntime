// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/external_data_loader_thread_pool.h"

namespace onnxruntime {
namespace cuda {

#if !defined(ORT_MINIMAL_BUILD)
using ::onnxruntime::ExternalDataLoaderThreadPool;
#endif

}  // namespace cuda
}  // namespace onnxruntime
