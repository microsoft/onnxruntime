// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { expect } from 'chai';

import { shouldRequestSubgroupsFeature } from '../../../../lib/wasm/jsep/backend-webgpu';

describe('#UnitTest# - wasm - webgpu feature selection', () => {
  const originalNavigatorDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'navigator');

  const setUserAgent = (userAgent: string) => {
    Object.defineProperty(globalThis, 'navigator', {
      value: { userAgent },
      configurable: true,
    });
  };

  afterEach(() => {
    if (originalNavigatorDescriptor) {
      Object.defineProperty(globalThis, 'navigator', originalNavigatorDescriptor);
    } else {
      delete (globalThis as { navigator?: unknown }).navigator;
    }
  });

  it('disables subgroups for Electron + Intel', () => {
    setUserAgent('Mozilla/5.0 Electron/1.0');
    expect(shouldRequestSubgroupsFeature({ vendor: 'Intel' } as GPUAdapterInfo)).to.be.false;
  });

  it('disables subgroups for Electron when adapter info is not available', () => {
    setUserAgent('Mozilla/5.0 Electron/1.0');
    expect(shouldRequestSubgroupsFeature()).to.be.false;
  });

  it('keeps subgroups enabled for Electron + non-Intel', () => {
    setUserAgent('Mozilla/5.0 Electron/1.0');
    expect(shouldRequestSubgroupsFeature({ vendor: 'NVIDIA' } as GPUAdapterInfo)).to.be.true;
  });

  it('keeps subgroups enabled outside Electron', () => {
    setUserAgent('Mozilla/5.0 Chrome/1.0');
    expect(shouldRequestSubgroupsFeature({ vendor: 'Intel' } as GPUAdapterInfo)).to.be.true;
  });
});
