// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

it('Browser E2E testing - WebAssembly backend (multiple inference session create calls)', async function() {
  const sessionPromiseA = createSession(ort);
  const sessionPromiseB = createSession(ort);
  await Promise.all([sessionPromiseA, sessionPromiseB]);
});
