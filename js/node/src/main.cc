// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <napi.h>

#include "ep_context_data_read_helper.h"
#include "inference_session_wrap.h"

Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
  InferenceSessionWrap::Init(env, exports);
  exports.Set("__testEpContextDataReadCallback",
              Napi::Function::New(env, TestEpContextDataReadCallback));
  return exports;
}

NODE_API_MODULE(onnxruntime, InitAll)
