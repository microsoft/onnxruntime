// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/* eslint-disable no-underscore-dangle, @typescript-eslint/no-unused-expressions */

import { expect } from 'chai';
import { env } from 'onnxruntime-common';

interface MockWasmModule {
  _OrtIsTelemetrySupported?: () => number;
  __ortTelemetryCallback?: (eventName: string, eventData: Record<string, unknown>) => void;
}

type GlobalPropertyName = 'navigator' | 'location';

// initTelemetry triggers timers and event listeners at module scope,
// so we import it once and let the BUILD_DEFS guard handle no-ops in test builds.
import { initTelemetry, logSessionModelInfo } from '../../lib/wasm/telemetry.js';
import type { OrtWasmModule } from '../../lib/wasm/wasm-types.js';

const asModule = (m: MockWasmModule) => m as unknown as OrtWasmModule;
const getGlobalPropertyDescriptor = (name: GlobalPropertyName): PropertyDescriptor | undefined =>
  Object.getOwnPropertyDescriptor(globalThis, name);

const restoreGlobalProperty = (name: GlobalPropertyName, descriptor: PropertyDescriptor | undefined): void => {
  if (descriptor) {
    Object.defineProperty(globalThis, name, descriptor);
  } else {
    Reflect.deleteProperty(globalThis, name);
  }
};

describe('#UnitTest# - Telemetry Bridge', () => {
  let mockModule: MockWasmModule;
  let navigatorDescriptor: PropertyDescriptor | undefined;
  let locationDescriptor: PropertyDescriptor | undefined;

  before(() => {
    navigatorDescriptor = getGlobalPropertyDescriptor('navigator');
    locationDescriptor = getGlobalPropertyDescriptor('location');
  });

  beforeEach(() => {
    mockModule = { _OrtIsTelemetrySupported: () => 1 };
    env.telemetry.enabled = true;
    env.telemetry.onEvent = undefined;
  });

  afterEach(() => {
    env.telemetry.enabled = true;
    env.telemetry.onEvent = undefined;
    restoreGlobalProperty('navigator', navigatorDescriptor);
    restoreGlobalProperty('location', locationDescriptor);
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

  it('should merge browser metadata into ProcessInfo', () => {
    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: { hardwareConcurrency: 8, language: 'en-US', userAgent: 'test-agent' },
    });
    Object.defineProperty(globalThis, 'location', {
      configurable: true,
      value: { origin: 'https://example.test' },
    });

    initTelemetry(asModule(mockModule));

    const receivedEvents: Array<{ name: string; data: Record<string, unknown> }> = [];
    env.telemetry.onEvent = (name: string, data: Record<string, unknown>) => {
      receivedEvents.push({ name, data });
    };

    mockModule.__ortTelemetryCallback!('ProcessInfo', { runtimeVersion: '1.25.0', projection: 1 });

    expect(receivedEvents).to.have.length(1);
    expect(receivedEvents[0].name).to.equal('ProcessInfo');
    expect(receivedEvents[0].data.runtimeVersion).to.equal('1.25.0');
    expect(receivedEvents[0].data.projection).to.equal(1);
    expect(receivedEvents[0].data.cpuCount).to.equal(8);
    expect(receivedEvents[0].data.locale).to.equal('en-US');
    expect(receivedEvents[0].data.userAgent).to.equal('test-agent');
    expect(receivedEvents[0].data.origin).to.equal('https://example.test');
  });

  it('should not emit a separate startup metadata event during init', () => {
    const receivedEvents: Array<{ name: string; data: Record<string, unknown> }> = [];
    env.telemetry.onEvent = (name: string, data: Record<string, unknown>) => {
      receivedEvents.push({ name, data });
    };

    initTelemetry(asModule(mockModule));

    expect(receivedEvents).to.have.length(0);
  });

  it('should forward JS-generated model events to observer callback', () => {
    const receivedEvents: Array<{ name: string; data: Record<string, unknown> }> = [];
    env.telemetry.onEvent = (name: string, data: Record<string, unknown>) => {
      receivedEvents.push({ name, data });
    };

    initTelemetry(asModule(mockModule));
    logSessionModelInfo(1024, 2, 1);

    expect(receivedEvents).to.have.length(1);
    expect(receivedEvents[0].name).to.equal('modelinfo');
    expect(receivedEvents[0].data.modelSizeBytes).to.equal(1024);
    expect(receivedEvents[0].data.inputCount).to.equal(2);
    expect(receivedEvents[0].data.outputCount).to.equal(1);
  });

  it('should not emit JS-generated events when telemetry is disabled', () => {
    const receivedEvents: Array<{ name: string; data: Record<string, unknown> }> = [];
    env.telemetry.enabled = false;
    env.telemetry.onEvent = (name: string, data: Record<string, unknown>) => {
      receivedEvents.push({ name, data });
    };

    initTelemetry(asModule(mockModule));
    logSessionModelInfo(2048, 3, 2);

    expect(receivedEvents).to.have.length(0);
  });

  it('should skip telemetry when the wasm build does not support it', () => {
    const unsupportedModule: MockWasmModule = { _OrtIsTelemetrySupported: () => 0 };
    const receivedEvents: Array<{ name: string; data: Record<string, unknown> }> = [];
    env.telemetry.onEvent = (name: string, data: Record<string, unknown>) => {
      receivedEvents.push({ name, data });
    };

    initTelemetry(asModule(unsupportedModule));
    logSessionModelInfo(4096, 4, 2);

    expect(unsupportedModule.__ortTelemetryCallback).to.be.undefined;
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
