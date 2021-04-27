// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

if (typeof window !== 'undefined') {
  require('./backends/webgl/test_glsl_function_inliner');
  require('./backends/webgl/test_pack_unpack');
}

require('./opset');
