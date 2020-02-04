// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "common.h"

// Needed to work around the fact that OnnxRuntime defines ERROR
#ifdef ERROR
#undef ERROR
#endif
#include "core/session/inference_session.h"
// Restore ERROR define
#define ERROR 0

#ifdef USE_DML
#include <DirectML.h>
#endif USE_DML

#include "core/framework/customregistry.h"
#include "core/framework/allocatormgr.h"
#include "core/session/environment.h"
#include "core/session/IOBinding.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"