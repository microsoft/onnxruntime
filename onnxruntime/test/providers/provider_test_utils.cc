// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <sstream>

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/data_types_internal.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/graph/model_load_utils.h"
#include "gmock/gmock.h"
#include "test/providers/provider_test_utils.h"
#include "test/providers/run_options_config_keys.h"
#include "test/util/include/default_providers.h"
#include "test/framework/test_utils.h"
#include <csignal>
#include <exception>
#include <memory>

#ifdef ENABLE_TRAINING
#include "orttraining/core/session/training_session.h"
#endif

using namespace ::onnxruntime::logging;

namespace onnxruntime {
namespace test {

}  // namespace test
}  // namespace onnxruntime
