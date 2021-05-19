// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

if (typeof window !== 'undefined') {
  require('./backends/webgl/test-glsl-function-inliner');
  require('./backends/webgl/test-conv-new');
  require('./backends/webgl/test-pack-unpack');
  require('./backends/webgl/test-concat-packed');
  require('./backends/webgl/test-reshape-packed');
}

require('./opset');
