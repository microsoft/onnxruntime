// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_experimental_c_api.h"

void OrtEpContextLegacyHeaderCompileTest(OrtReadNamedBufferFunc read_func,
                                         OrtWriteNamedBufferFunc write_func,
                                         OrtEpContextConfig* config) {
  (void)read_func;
  (void)write_func;
  (void)config;
}
