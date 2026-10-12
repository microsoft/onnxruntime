// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { expect } from 'chai';

import { validateSessionOptions } from '../../lib/validate-session-options';

describe('EPContext data read callback', () => {
  it('is rejected by ONNX Runtime Web backends', () => {
    expect(() =>
      validateSessionOptions({
        epContextDataRead: { callback: () => new Uint8Array(), maxDataSize: 1 },
      }),
    ).to.throw('session option "epContextDataRead" is not supported by ONNX Runtime Web');
  });

  it('does not affect other session options', () => {
    expect(() => validateSessionOptions({ graphOptimizationLevel: 'all' })).not.to.throw();
  });
});
