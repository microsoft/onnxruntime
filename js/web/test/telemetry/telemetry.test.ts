// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/* eslint-disable no-underscore-dangle, @typescript-eslint/no-unused-expressions */

import { expect } from 'chai';
import { env } from 'onnxruntime-common';

interface MockWasmModule {
  _OrtIsTelemetrySupported?: () => number;
  __ortTelemetryCallback?: (eventName: string, eventData: Record<string, unknown>) => void;
}

type GlobalPropertyName = 'fetch' | 'localStorage' | 'navigator' | 'location' | 'setTimeout';

// initTelemetry triggers timers and event listeners at module scope,
// so we import it once and let the BUILD_DEFS guard handle no-ops in test builds.
import { flushTelemetry, initTelemetry, resetTelemetryForTesting } from '../../lib/wasm/telemetry.js';
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

const createMockStorage = (initialValues: Record<string, string> = {}) => {
  const store = new Map<string, string>(Object.entries(initialValues));
  return {
    storage: {
      getItem: (key: string) => store.get(key) ?? null,
      setItem: (key: string, value: string) => {
        store.set(key, value);
      },
      removeItem: (key: string) => {
        store.delete(key);
      },
    },
    getItem: (key: string) => store.get(key) ?? null,
  };
};

const hashDeviceId = async (value: string): Promise<string> => {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value));
  return `c:${Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('')}`;
};

describe('#UnitTest# - Telemetry Bridge', () => {
  let mockModule: MockWasmModule;
  let fetchDescriptor: PropertyDescriptor | undefined;
  let localStorageDescriptor: PropertyDescriptor | undefined;
  let navigatorDescriptor: PropertyDescriptor | undefined;
  let locationDescriptor: PropertyDescriptor | undefined;
  let setTimeoutDescriptor: PropertyDescriptor | undefined;

  before(() => {
    fetchDescriptor = getGlobalPropertyDescriptor('fetch');
    localStorageDescriptor = getGlobalPropertyDescriptor('localStorage');
    navigatorDescriptor = getGlobalPropertyDescriptor('navigator');
    locationDescriptor = getGlobalPropertyDescriptor('location');
    setTimeoutDescriptor = getGlobalPropertyDescriptor('setTimeout');
  });

  beforeEach(() => {
    resetTelemetryForTesting();
    mockModule = { _OrtIsTelemetrySupported: () => 1 };
    env.telemetry.enabled = true;
    env.telemetry.onEvent = undefined;

    Object.defineProperty(globalThis, 'fetch', {
      configurable: true,
      value: async () => ({ ok: true, status: 200 }),
    });
  });

  afterEach(async () => {
    await flushTelemetry();
    resetTelemetryForTesting();
    env.telemetry.enabled = true;
    env.telemetry.onEvent = undefined;
    restoreGlobalProperty('fetch', fetchDescriptor);
    restoreGlobalProperty('localStorage', localStorageDescriptor);
    restoreGlobalProperty('navigator', navigatorDescriptor);
    restoreGlobalProperty('location', locationDescriptor);
    restoreGlobalProperty('setTimeout', setTimeoutDescriptor);
  });

  it('should hash the device id before sending telemetry', async () => {
    const fetchCalls: Array<{ url: string; init?: RequestInit }> = [];
    const localStorageMock = createMockStorage();

    Object.defineProperty(globalThis, 'localStorage', {
      configurable: true,
      value: localStorageMock.storage,
    });
    Object.defineProperty(globalThis, 'fetch', {
      configurable: true,
      value: async (url: string, init?: RequestInit) => {
        fetchCalls.push({ url, init });
        return { ok: true, status: 200 };
      },
    });

    initTelemetry(asModule(mockModule));
    mockModule.__ortTelemetryCallback!('ProcessInfo', { runtimeVersion: '1.25.0' });
    await flushTelemetry();

    const storedDeviceId = localStorageMock.getItem('ort_device_id');
    expect(storedDeviceId).to.be.a('string');
    expect(fetchCalls).to.have.length(1);
    expect(fetchCalls[0].url).to.contain('https://mobile.events.data.microsoft.com/OneCollector/1.0');

    const payload = JSON.parse(fetchCalls[0].init!.body as string);
    expect(payload.ext.device.localId).to.equal(await hashDeviceId(storedDeviceId!));
    expect(payload.ext.device.localId).to.not.equal(storedDeviceId);
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

  it('should respect the runtime telemetry gate for observer callbacks', () => {
    initTelemetry(asModule(mockModule));
    env.telemetry.enabled = false;

    const receivedEvents: Array<{ name: string; data: Record<string, unknown> }> = [];
    env.telemetry.onEvent = (name: string, data: Record<string, unknown>) => {
      receivedEvents.push({ name, data });
    };

    mockModule.__ortTelemetryCallback!('SessionCreationStart', { sessionId: 1 });
    mockModule.__ortTelemetryCallback!('ProcessInfo', { runtimeVersion: '1.25.0' });

    expect(receivedEvents).to.have.length(1);
    expect(receivedEvents[0].name).to.equal('ProcessInfo');
    expect(receivedEvents[0].data.runtimeVersion).to.equal('1.25.0');
  });

  it('should merge browser metadata into ProcessInfo', () => {
    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: { hardwareConcurrency: 8, userAgent: 'test-agent' },
    });
    Object.defineProperty(globalThis, 'location', {
      configurable: true,
      value: { origin: 'https://example.test:8443' },
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
    expect(receivedEvents[0].data.userAgent).to.equal('test-agent');
    expect(receivedEvents[0].data.origin).to.equal('https://example.test:8443');
    expect(receivedEvents[0].data).to.not.have.property('host');
    expect(receivedEvents[0].data).to.not.have.property('locale');
    expect(receivedEvents[0].data).to.not.have.property('timezone');
  });

  it('should not emit a separate startup metadata event during init', () => {
    const receivedEvents: Array<{ name: string; data: Record<string, unknown> }> = [];
    env.telemetry.onEvent = (name: string, data: Record<string, unknown>) => {
      receivedEvents.push({ name, data });
    };

    initTelemetry(asModule(mockModule));

    expect(receivedEvents).to.have.length(0);
  });

  it('should respect the runtime telemetry gate for uploads', async () => {
    const fetchCalls: Array<{ url: string; init?: RequestInit }> = [];
    Object.defineProperty(globalThis, 'fetch', {
      configurable: true,
      value: async (url: string, init?: RequestInit) => {
        fetchCalls.push({ url, init });
        return { ok: true, status: 200 };
      },
    });

    env.telemetry.enabled = false;
    initTelemetry(asModule(mockModule));
    mockModule.__ortTelemetryCallback!('SessionCreationStart', { sessionId: 99 });
    mockModule.__ortTelemetryCallback!('ProcessInfo', { runtimeVersion: '1.25.0' });
    await flushTelemetry();

    expect(fetchCalls).to.have.length(1);

    const payload = JSON.parse(fetchCalls[0].init!.body as string);
    expect(payload.name).to.equal('processinfo');
    expect(payload.data.runtimeVersion).to.equal('1.25.0');
  });

  it('should split oversized batches by byte size', async () => {
    const fetchCalls: Array<{ url: string; init?: RequestInit }> = [];
    Object.defineProperty(globalThis, 'fetch', {
      configurable: true,
      value: async (url: string, init?: RequestInit) => {
        fetchCalls.push({ url, init });
        return { ok: true, status: 200 };
      },
    });

    initTelemetry(asModule(mockModule));
    const details = 'x'.repeat(60 * 1024);
    mockModule.__ortTelemetryCallback!('SessionCreationStart', { sessionId: 1, details });
    mockModule.__ortTelemetryCallback!('SessionCreationStart', { sessionId: 2, details });
    await flushTelemetry();

    expect(fetchCalls).to.have.length(2);
    expect((fetchCalls[0].init!.body as string).split('\n')).to.have.length(1);
    expect((fetchCalls[1].init!.body as string).split('\n')).to.have.length(1);
  });

  it('should split unload batches across multiple beacon sends', async () => {
    const beaconCalls: Array<{ url: string; data: Blob }> = [];
    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: {
        sendBeacon: (url: string, data: Blob) => {
          beaconCalls.push({ url, data });
          return true;
        },
      },
    });

    initTelemetry(asModule(mockModule));
    const details = 'x'.repeat(35 * 1024);
    mockModule.__ortTelemetryCallback!('SessionCreationStart', { sessionId: 1, details });
    mockModule.__ortTelemetryCallback!('SessionCreationStart', { sessionId: 2, details });
    mockModule.__ortTelemetryCallback!('SessionCreationStart', { sessionId: 3, details });
    await flushTelemetry(true);

    expect(beaconCalls.length).to.be.greaterThan(1);
  });

  it('should fall back to fetch when beacon cannot send unload telemetry', async () => {
    const fetchCalls: Array<{ url: string; init?: RequestInit }> = [];
    let beaconCalls = 0;
    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: {
        sendBeacon: () => {
          beaconCalls += 1;
          return false;
        },
      },
    });
    Object.defineProperty(globalThis, 'fetch', {
      configurable: true,
      value: async (url: string, init?: RequestInit) => {
        fetchCalls.push({ url, init });
        return { ok: true, status: 200 };
      },
    });

    initTelemetry(asModule(mockModule));
    mockModule.__ortTelemetryCallback!('SessionCreationStart', { sessionId: 1 });
    await flushTelemetry(true);

    expect(beaconCalls).to.equal(1);
    expect(fetchCalls).to.have.length(1);
  });

  it('should defer non-unload flushes while offline until connectivity returns', async () => {
    const fetchCalls: Array<{ url: string; init?: RequestInit }> = [];
    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: { onLine: false },
    });
    Object.defineProperty(globalThis, 'fetch', {
      configurable: true,
      value: async (url: string, init?: RequestInit) => {
        fetchCalls.push({ url, init });
        return { ok: true, status: 200 };
      },
    });

    initTelemetry(asModule(mockModule));
    mockModule.__ortTelemetryCallback!('SessionCreationStart', { sessionId: 1 });
    await flushTelemetry();
    expect(fetchCalls).to.have.length(0);

    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: { onLine: true },
    });
    await flushTelemetry();
    expect(fetchCalls).to.have.length(1);
  });

  it('should use capped jittered backoff for retries', async () => {
    const retryDelays: number[] = [];
    const originalRandom = Math.random;
    Math.random = () => 1;
    Object.defineProperty(globalThis, 'setTimeout', {
      configurable: true,
      value: (_callback: () => void, delay?: number) => {
        retryDelays.push(delay ?? 0);
        return 0;
      },
    });
    Object.defineProperty(globalThis, 'fetch', {
      configurable: true,
      value: async () => ({ ok: false, status: 429 }),
    });

    try {
      initTelemetry(asModule(mockModule));
      mockModule.__ortTelemetryCallback!('SessionCreationStart', { sessionId: 1 });
      await flushTelemetry();
      await Promise.resolve();

      expect(retryDelays).to.deep.equal([3600]);
    } finally {
      Math.random = originalRandom;
    }
  });

  it('should skip telemetry when the wasm build does not support it', () => {
    const unsupportedModule: MockWasmModule = { _OrtIsTelemetrySupported: () => 0 };

    initTelemetry(asModule(unsupportedModule));

    expect(unsupportedModule.__ortTelemetryCallback).to.be.undefined;
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
