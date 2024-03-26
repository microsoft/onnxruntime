#pragma once

#define ORT_API_MANUAL_INIT
#include "onnxruntime_c_api.h"
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#define ORT_API_VERSION_SUPPORTED 16
