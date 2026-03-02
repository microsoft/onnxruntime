// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { InferenceSession, Tensor } from 'onnxruntime-common';

(async () => {
  // Check that `await using` declarations work with `InferenceSession` (it implements `AsyncDisposable`).
  // {type-tests}|pass
  await using session = await InferenceSession.create(new ArrayBuffer(0));

  // Check that `using` declarations work with `Tensor` (it implements `Disposable`).
  // {type-tests}|pass
  using tensor = new Tensor('float32', [1, 2, 3], [3]);
})();
