// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { expect } from 'chai';
import type { InferenceSession } from 'onnxruntime-common';

import { serializeStorageBufferBucketLimits } from '../../lib/wasm/webgpu-buffer-cache';

describe('WebGPU storage buffer bucket limits', () => {
  const serialize = (
    limits: unknown,
    mode?: InferenceSession.WebGpuExecutionProviderOption['storageBufferCacheMode'],
  ) =>
    serializeStorageBufferBucketLimits({
      name: 'webgpu',
      storageBufferCacheMode: mode,
      storageBufferBucketLimits: limits as Readonly<Record<number, number>>,
    });

  it('encodes a replacement table for the native EP, including zero limits', () => {
    expect(serialize({ 8388608: 24, 64: 0, 5242880: 24 })).to.equal('64:0,5242880:24,8388608:24');
    expect(serialize({ 16: 1 }, 'bucket')).to.equal('16:1');
  });

  for (const limits of [
    undefined,
    null,
    {},
    [],
    '16:1',
    { 0: 1 },
    { 17: 1 },
    { '-16': 1 },
    { '1e3': 1 },
    { '016': 1 },
    { '16.0': 1 },
    { ' 16': 1 },
    { 9007199254740992: 1 },
  ]) {
    it(`rejects invalid bucket sizes or tables: ${JSON.stringify(limits)}`, () => {
      expect(() => serialize(limits)).to.throw('storageBufferBucketLimits');
    });
  }
  for (const limit of [-1, 0.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1, '1', null]) {
    it(`rejects invalid free-buffer limits: ${String(limit)}`, () => {
      expect(() => serialize({ 16: limit })).to.throw('limits must be non-negative safe integers');
    });
  }
  for (const mode of ['simple', 'disabled', 'lazyRelease'] as const) {
    it(`rejects the incompatible ${mode} cache mode`, () => {
      expect(() => serialize({ 16: 1 }, mode)).to.throw("requires storageBufferCacheMode 'bucket'");
    });
  }
});
