#pragma once

#include "Env.h"
#include <ReactCommon/CallInvoker.h>
#include <jsi/jsi.h>

namespace onnxruntimejsi {

std::shared_ptr<Env>
install(facebook::jsi::Runtime &runtime,
        std::shared_ptr<facebook::react::CallInvoker> jsInvoker = nullptr);

} // namespace onnxruntimejsi
