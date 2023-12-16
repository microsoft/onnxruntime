// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common_no_ort_symbol.h"
#include "core/session/onnxruntime_c_api.h"
#include "ortdevice.h"
#include "ortmemoryinfo.h"
#include <map>
#include <string>
#include <sstream>

namespace onnxruntime {
class Stream;
namespace synchronize {
class Notification;
}
using WaitNotificationFn = std::function<void(Stream&, synchronize::Notification&)>;
}  // namespace onnxruntime
