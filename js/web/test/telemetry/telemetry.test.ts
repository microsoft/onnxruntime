// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/* eslint-disable no-underscore-dangle, @typescript-eslint/no-unused-expressions */

import { expect } from 'chai';
import { env } from 'onnxruntime-common';

interface MockWasmModule {
  _OrtIsTelemetrySupported?: () => number;
  __ortTelemetryCallback?: (eventName: string, eventData: Record<string, unknown>) => void;
}

type GlobalPropertyName =
  | 'document'
  | 'fetch'
  | 'localStorage'
  | 'navigator'
  | 'location'
  | 'sessionStorage'
  | 'setTimeout';
type MockCookie = { name: string; value: string; domain: string; hostOnly: boolean };

// initTelemetry triggers timers and event listeners at module scope,
// so we import it once and let the BUILD_DEFS guard handle no-ops in test builds.
import {
  flushTelemetry,
  initTelemetry,
  logSessionModelInfo,
  resetTelemetryForTesting,
} from '../../lib/wasm/telemetry.js';
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

const createMockDocument = (hostname: string, acceptedDomains: readonly string[] = []) => {
  const cookies = new Map<string, MockCookie>();
  const acceptedDomainSet = new Set(acceptedDomains.map((domain) => domain.replace(/^\./, '').toLowerCase()));
  const normalizeDomain = (domain: string): string => domain.trim().replace(/^\./, '').toLowerCase();
  const getCookieKey = (name: string, domain: string, hostOnly: boolean): string =>
    `${name}|${domain}|${hostOnly ? 'host' : 'domain'}`;
  const isAccessible = (cookie: MockCookie): boolean =>
    cookie.hostOnly ? cookie.domain === hostname : cookie.domain === hostname || hostname.endsWith(`.${cookie.domain}`);

  const writeCookie = (cookieString: string): void => {
    const cookieParts = cookieString
      .split(';')
      .map((part) => part.trim())
      .filter(Boolean);
    if (cookieParts.length === 0) {
      return;
    }

    const [nameValue, ...attributeParts] = cookieParts;
    const separatorIndex = nameValue.indexOf('=');
    if (separatorIndex < 0) {
      return;
    }

    const name = decodeURIComponent(nameValue.slice(0, separatorIndex));
    const value = decodeURIComponent(nameValue.slice(separatorIndex + 1));
    let domain: string | null = null;
    let maxAge: number | undefined;
    for (const attribute of attributeParts) {
      const [attributeName, ...attributeValueParts] = attribute.split('=');
      const attributeValue = attributeValueParts.join('=');
      if (attributeName.toLowerCase() === 'domain') {
        domain = normalizeDomain(attributeValue);
      } else if (attributeName.toLowerCase() === 'max-age') {
        maxAge = Number(attributeValue);
      }
    }

    if (domain === null) {
      const cookieKey = getCookieKey(name, hostname, true);
      if (maxAge !== undefined && maxAge <= 0) {
        cookies.delete(cookieKey);
        return;
      }

      cookies.set(cookieKey, { name, value, domain: hostname, hostOnly: true });
      return;
    }

    if (!acceptedDomainSet.has(domain)) {
      return;
    }

    const cookieKey = getCookieKey(name, domain, false);
    if (maxAge !== undefined && maxAge <= 0) {
      cookies.delete(cookieKey);
      return;
    }

    cookies.set(cookieKey, { name, value, domain, hostOnly: false });
  };

  const documentValue = {};
  Object.defineProperty(documentValue, 'cookie', {
    configurable: true,
    get: () =>
      [...cookies.values()]
        .filter(isAccessible)
        .map((cookie) => `${encodeURIComponent(cookie.name)}=${encodeURIComponent(cookie.value)}`)
        .join('; '),
    set: (cookieString: string) => {
      writeCookie(cookieString);
    },
  });

  return {
    document: documentValue as { cookie: string },
    getCookie: (name: string) =>
      [...cookies.values()].find((cookie) => cookie.name === name && isAccessible(cookie))?.value ?? null,
    getStoredCookie: (name: string) => [...cookies.values()].find((cookie) => cookie.name === name) ?? null,
    seedCookie: (name: string, value: string, domain?: string) => {
      const domainAttribute = domain ? `; Domain=${domain}` : '';
      writeCookie(`${encodeURIComponent(name)}=${encodeURIComponent(value)}; Max-Age=60${domainAttribute}`);
    },
  };
};

const hashDeviceId = async (value: string): Promise<string> => {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value));
  return `c:${Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('')}`;
};

describe('#UnitTest# - Telemetry Bridge', () => {
  let mockModule: MockWasmModule;
  let documentDescriptor: PropertyDescriptor | undefined;
  let fetchDescriptor: PropertyDescriptor | undefined;
  let localStorageDescriptor: PropertyDescriptor | undefined;
  let navigatorDescriptor: PropertyDescriptor | undefined;
  let locationDescriptor: PropertyDescriptor | undefined;
  let sessionStorageDescriptor: PropertyDescriptor | undefined;
  let setTimeoutDescriptor: PropertyDescriptor | undefined;

  before(() => {
    documentDescriptor = getGlobalPropertyDescriptor('document');
    fetchDescriptor = getGlobalPropertyDescriptor('fetch');
    localStorageDescriptor = getGlobalPropertyDescriptor('localStorage');
    navigatorDescriptor = getGlobalPropertyDescriptor('navigator');
    locationDescriptor = getGlobalPropertyDescriptor('location');
    sessionStorageDescriptor = getGlobalPropertyDescriptor('sessionStorage');
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
    restoreGlobalProperty('document', documentDescriptor);
    restoreGlobalProperty('fetch', fetchDescriptor);
    restoreGlobalProperty('localStorage', localStorageDescriptor);
    restoreGlobalProperty('navigator', navigatorDescriptor);
    restoreGlobalProperty('location', locationDescriptor);
    restoreGlobalProperty('sessionStorage', sessionStorageDescriptor);
    restoreGlobalProperty('setTimeout', setTimeoutDescriptor);
  });

  it('should store a new device id in both localStorage and a shared-domain cookie before sending telemetry', async () => {
    const fetchCalls: Array<{ url: string; init?: RequestInit }> = [];
    const localStorageMock = createMockStorage();
    const mockDocument = createMockDocument('app.example.test', ['app.example.test', 'example.test']);

    Object.defineProperty(globalThis, 'document', {
      configurable: true,
      value: mockDocument.document,
    });
    Object.defineProperty(globalThis, 'localStorage', {
      configurable: true,
      value: localStorageMock.storage,
    });
    Object.defineProperty(globalThis, 'location', {
      configurable: true,
      value: { hostname: 'app.example.test', origin: 'https://app.example.test', protocol: 'https:' },
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
    expect(mockDocument.getCookie('ort_device_id')).to.equal(storedDeviceId);
    expect(mockDocument.getStoredCookie('ort_device_id')?.domain).to.equal('example.test');
    expect(fetchCalls).to.have.length(1);
    expect(fetchCalls[0].url).to.contain('https://mobile.events.data.microsoft.com/OneCollector/1.0');

    const payload = JSON.parse(fetchCalls[0].init!.body as string);
    expect(payload.ext.device.localId).to.equal(await hashDeviceId(storedDeviceId!));
    expect(payload.ext.device.localId).to.not.equal(storedDeviceId);
  });

  it('should prefer the cookie device id and repair localStorage from it', async () => {
    const fetchCalls: Array<{ url: string; init?: RequestInit }> = [];
    const localStorageMock = createMockStorage({ ort_device_id: 'local-storage-id' });
    const mockDocument = createMockDocument('app.example.test', ['app.example.test', 'example.test']);
    mockDocument.seedCookie('ort_device_id', 'cookie-device-id', 'example.test');

    Object.defineProperty(globalThis, 'document', {
      configurable: true,
      value: mockDocument.document,
    });
    Object.defineProperty(globalThis, 'localStorage', {
      configurable: true,
      value: localStorageMock.storage,
    });
    Object.defineProperty(globalThis, 'location', {
      configurable: true,
      value: { hostname: 'app.example.test', origin: 'https://app.example.test', protocol: 'https:' },
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

    expect(localStorageMock.getItem('ort_device_id')).to.equal('cookie-device-id');
    expect(fetchCalls).to.have.length(1);

    const payload = JSON.parse(fetchCalls[0].init!.body as string);
    expect(payload.ext.device.localId).to.equal(await hashDeviceId('cookie-device-id'));
  });

  it('should promote the localStorage device id into a host-only cookie when a shared-domain cookie is unavailable', async () => {
    const fetchCalls: Array<{ url: string; init?: RequestInit }> = [];
    const localStorageMock = createMockStorage({ ort_device_id: 'local-storage-id' });
    const mockDocument = createMockDocument('localhost');

    Object.defineProperty(globalThis, 'document', {
      configurable: true,
      value: mockDocument.document,
    });
    Object.defineProperty(globalThis, 'localStorage', {
      configurable: true,
      value: localStorageMock.storage,
    });
    Object.defineProperty(globalThis, 'location', {
      configurable: true,
      value: { hostname: 'localhost', origin: 'http://localhost:8080', protocol: 'http:' },
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

    expect(mockDocument.getCookie('ort_device_id')).to.equal('local-storage-id');
    expect(mockDocument.getStoredCookie('ort_device_id')?.hostOnly).to.equal(true);
    expect(fetchCalls).to.have.length(1);

    const payload = JSON.parse(fetchCalls[0].init!.body as string);
    expect(payload.ext.device.localId).to.equal(await hashDeviceId('local-storage-id'));
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

  it('should cache pending telemetry events in sessionStorage until they are delivered', async () => {
    const sessionStore = new Map<string, string>();
    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: { onLine: false },
    });
    Object.defineProperty(globalThis, 'sessionStorage', {
      configurable: true,
      value: {
        getItem: (key: string) => sessionStore.get(key) ?? null,
        setItem: (key: string, value: string) => {
          sessionStore.set(key, value);
        },
        removeItem: (key: string) => {
          sessionStore.delete(key);
        },
      },
    });

    initTelemetry(asModule(mockModule));
    mockModule.__ortTelemetryCallback!('SessionCreationStart', { sessionId: 1 });
    await flushTelemetry();

    const storedPayload = sessionStore.get('ort_pending_telemetry');
    expect(storedPayload).to.be.a('string');

    const storedEntries = JSON.parse(storedPayload!) as Array<{ eventName: string }>;
    expect(storedEntries).to.have.length(1);
    expect(storedEntries[0].eventName).to.equal('SessionCreationStart');
  });

  it('should restore buffered telemetry events from sessionStorage and clear them after a successful send', async () => {
    const fetchCalls: Array<{ url: string; init?: RequestInit }> = [];
    const sessionStore = new Map<string, string>();
    const storedEntries = [
      {
        id: 'entry-1',
        eventName: 'SessionCreationStart',
        payload: JSON.stringify({ name: 'sessioncreationstart', data: { sessionId: 7 } }),
      },
      {
        id: 'entry-2',
        eventName: 'ProcessInfo',
        payload: JSON.stringify({ name: 'processinfo', data: { runtimeVersion: '1.25.0' } }),
      },
    ];
    sessionStore.set('ort_pending_telemetry', JSON.stringify(storedEntries));

    Object.defineProperty(globalThis, 'sessionStorage', {
      configurable: true,
      value: {
        getItem: (key: string) => sessionStore.get(key) ?? null,
        setItem: (key: string, value: string) => {
          sessionStore.set(key, value);
        },
        removeItem: (key: string) => {
          sessionStore.delete(key);
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
    await flushTelemetry();

    expect(fetchCalls).to.have.length(1);
    expect(fetchCalls[0].init!.body).to.equal(storedEntries.map((entry) => entry.payload).join('\n'));
    expect(sessionStore.has('ort_pending_telemetry')).to.equal(false);
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
