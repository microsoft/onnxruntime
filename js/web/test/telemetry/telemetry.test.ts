// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/**
 * Unit tests for the ONNX Runtime Web telemetry bridge.
 *
 * These tests verify:
 *   - Observer callback receives correct event names and data
 *   - Telemetry is disabled when env.telemetry.enabled = false
 *   - Telemetry bridge handles missing callbacks gracefully
 *   - Observer errors do not propagate
 */

import { expect } from 'chai';
import { env } from 'onnxruntime-common';

// Mock OrtWasmModule for testing
interface MockWasmModule {
  __ortTelemetryCallback?: (eventName: string, eventData: Record<string, unknown>) => void;
}

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

  it('should register callback on module', async () => {
    const { initTelemetry } = await import('../../lib/wasm/telemetry.js');
    initTelemetry(mockModule as unknown as import('../../lib/wasm/wasm-types.js').OrtWasmModule);
    expect(mockModule.__ortTelemetryCallback).to.not.be.undefined;
    expect(typeof mockModule.__ortTelemetryCallback).to.equal('function');
  });

  it('should forward events to observer callback', async () => {
    const { initTelemetry } = await import('../../lib/wasm/telemetry.js');
    initTelemetry(mockModule as unknown as import('../../lib/wasm/wasm-types.js').OrtWasmModule);

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

  it('should not forward events when telemetry is disabled', async () => {
    const { initTelemetry } = await import('../../lib/wasm/telemetry.js');
    initTelemetry(mockModule as unknown as import('../../lib/wasm/wasm-types.js').OrtWasmModule);

    env.telemetry.enabled = false;

    const receivedEvents: Array<{ name: string; data: Record<string, unknown> }> = [];
    env.telemetry.onEvent = (name: string, data: Record<string, unknown>) => {
      receivedEvents.push({ name, data });
    };

    mockModule.__ortTelemetryCallback!('SessionCreationStart', { sessionId: 1 });

    expect(receivedEvents).to.have.length(0);
  });

  it('should handle missing observer callback gracefully', async () => {
    const { initTelemetry } = await import('../../lib/wasm/telemetry.js');
    initTelemetry(mockModule as unknown as import('../../lib/wasm/wasm-types.js').OrtWasmModule);

    // No onEvent callback set — should not throw
    expect(() => {
      mockModule.__ortTelemetryCallback!('ProcessInfo', { runtimeVersion: '1.25.0' });
    }).to.not.throw();
  });

  it('should swallow observer callback errors', async () => {
    const { initTelemetry } = await import('../../lib/wasm/telemetry.js');
    initTelemetry(mockModule as unknown as import('../../lib/wasm/wasm-types.js').OrtWasmModule);

    env.telemetry.onEvent = () => {
      throw new Error('Observer error');
    };

    // Should not throw despite observer error
    expect(() => {
      mockModule.__ortTelemetryCallback!('ProcessInfo', { runtimeVersion: '1.25.0' });
    }).to.not.throw();
  });
});
