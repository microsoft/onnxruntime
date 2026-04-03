// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/* eslint-disable no-underscore-dangle, @typescript-eslint/no-unused-expressions */

import { expect } from 'chai';
import { env } from 'onnxruntime-common';

interface MockWasmModule {
  __ortTelemetryCallback?: (eventName: string, eventData: Record<string, unknown>) => void;
}

// initTelemetry triggers timers and event listeners at module scope,
// so we import it once and let the BUILD_DEFS guard handle no-ops in test builds.
import { initTelemetry } from '../../lib/wasm/telemetry.js';
import type { OrtWasmModule } from '../../lib/wasm/wasm-types.js';

const asModule = (m: MockWasmModule) => m as unknown as OrtWasmModule;

describe('#UnitTest# - Telemetry Bridge', () => {
  let mockModule: MockWasmModule;

  beforeEach(() => {
    mockModule = {};
    env.telemetry.enabled = true;
    env.telemetry.onEvent = undefined;
  });

  afterEach(() => {
    env.telemetry.enabled = true;
    env.telemetry.onEvent = undefined;
  });

  it('should register callback on module', () => {
    initTelemetry(asModule(mockModule));
    expect(mockModule.__ortTelemetryCallback).to.not.be.undefined;
    expect(typeof mockModule.__ortTelemetryCallback).to.equal('function');
  });

  it('should forward events to observer callback', () => {
    initTelemetry(asModule(mockModule));

    const receivedEvents: Array<{ name: string; data: Record<string, unknown> }> = [];
    env.telemetry.onEvent = (name: string, data: Record<string, unknown>) => {
      receivedEvents.push({ name, data });
    };

    mockModule.__ortTelemetryCallback!('SessionCreationStart', { sessionId: 42 });
    mockModule.__ortTelemetryCallback!('ProcessInfo', { runtimeVersion: '1.25.0', platform: 'WebAssembly' });

    expect(receivedEvents).to.have.length(2);
    expect(receivedEvents[0].name).to.equal('SessionCreationStart');
    expect(receivedEvents[0].data.sessionId).to.equal(42);
    expect(receivedEvents[1].name).to.equal('ProcessInfo');
    expect(receivedEvents[1].data.platform).to.equal('WebAssembly');
  });

  it('should not forward events when telemetry is disabled', () => {
    initTelemetry(asModule(mockModule));
    env.telemetry.enabled = false;

    const receivedEvents: Array<{ name: string; data: Record<string, unknown> }> = [];
    env.telemetry.onEvent = (name: string, data: Record<string, unknown>) => {
      receivedEvents.push({ name, data });
    };

    mockModule.__ortTelemetryCallback!('SessionCreationStart', { sessionId: 1 });
    expect(receivedEvents).to.have.length(0);
  });

  it('should handle missing observer callback gracefully', () => {
    initTelemetry(asModule(mockModule));
    expect(() => {
      mockModule.__ortTelemetryCallback!('ProcessInfo', { runtimeVersion: '1.25.0' });
    }).to.not.throw();
  });

  it('should swallow observer callback errors', () => {
    initTelemetry(asModule(mockModule));
    env.telemetry.onEvent = () => {
      throw new Error('Observer error');
    };
    expect(() => {
      mockModule.__ortTelemetryCallback!('ProcessInfo', { runtimeVersion: '1.25.0' });
    }).to.not.throw();
  });
});
