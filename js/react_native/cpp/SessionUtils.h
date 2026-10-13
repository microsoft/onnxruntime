#pragma once

#include "Env.h"
#include "EpContextDataReadCallback.h"
#include <jsi/jsi.h>
#include <memory>
#include "onnxruntime_cxx_api.h"
#include <vector>

namespace onnxruntimejsi {

extern const std::vector<const char*> supportedBackends;

/**
 * @brief Parse the JavaScript session options into `sessionOptions`.
 *
 * Must be called on the JS thread. When the options register an `epContextDataRead` callback,
 * `epContextDataRead` receives the state that backs it; the caller must keep that state alive for
 * at least as long as the session created from `sessionOptions`.
 */
void parseSessionOptions(facebook::jsi::Runtime& runtime,
                         const facebook::jsi::Value& optionsValue,
                         Ort::SessionOptions& sessionOptions,
                         const std::shared_ptr<Env>& env,
                         std::shared_ptr<EpContextDataReadCallback>& epContextDataRead);

void parseRunOptions(facebook::jsi::Runtime& runtime,
                     const facebook::jsi::Value& optionsValue,
                     Ort::RunOptions& runOptions);

}  // namespace onnxruntimejsi
