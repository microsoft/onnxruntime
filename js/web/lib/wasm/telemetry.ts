// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { env } from 'onnxruntime-common';

import type { OrtWasmModule } from './wasm-types';

const IKEY = atob(
  'NWFkOTYzYmQ0YjNhNDExOGE0ODE0MDFjYzAyMTE4NzUtMGNiNDUxNTktNDZmNS00NDk1LWI0ZTUtZDA4OTAzNTVlOTY0LTY3OTc=',
);
const COLLECTOR_URL = `https://mobile.events.data.microsoft.com/OneCollector/1.0?cors=true&content-type=application/x-json-stream&apikey=${IKEY}`;
const IKEY_PREFIX = `o:${IKEY.split('-')[0]}`;
const FLUSH_INTERVAL_MS = 30_000;
const MAX_QUEUE_SIZE = 500;
const MAX_RETRIES = 6;
const DEVICE_ID_KEY = 'ort_device_id';

// eslint-disable-next-line prefer-const
let queue: string[] = [];
let flushTimer: ReturnType<typeof setInterval> | null = null;
let deviceId: string | null = null;
let wasmTelemetrySupported = false;

const generateUUID = (): string => {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  // Fallback for older browsers (Safari < 15.4)
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  // eslint-disable-next-line no-bitwise
  bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
  // eslint-disable-next-line no-bitwise
  bytes[8] = (bytes[8] & 0x3f) | 0x80; // variant 1
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
};

const getDeviceId = (): string => {
  if (deviceId) {
    return deviceId;
  }
  try {
    deviceId = localStorage.getItem(DEVICE_ID_KEY);
    if (!deviceId) {
      deviceId = generateUUID();
      localStorage.setItem(DEVICE_ID_KEY, deviceId);
    }
  } catch {
    deviceId = generateUUID();
  }
  return deviceId;
};

const getBrowserMetadata = (): Record<string, unknown> => {
  const nav = typeof navigator !== 'undefined' ? navigator : undefined;
  const metadata: Record<string, unknown> = {};
  if (nav) {
    if (nav.hardwareConcurrency) {
      metadata.cpuCount = nav.hardwareConcurrency;
    }
    if ((nav as unknown as Record<string, unknown>).deviceMemory) {
      metadata.deviceMemoryGB = (nav as unknown as Record<string, unknown>).deviceMemory;
    }
    if (nav.language) {
      metadata.locale = nav.language;
    }
    if (nav.userAgent) {
      metadata.userAgent = nav.userAgent;
    }
    const uaData = (nav as unknown as Record<string, unknown>).userAgentData as
      | { platform?: string; architecture?: string; mobile?: boolean }
      | undefined;
    if (uaData) {
      if (uaData.platform) {
        metadata.osPlatform = uaData.platform;
      }
      if (uaData.architecture) {
        metadata.architecture = uaData.architecture;
      }
      if (uaData.mobile !== undefined) {
        metadata.isMobile = uaData.mobile;
      }
    }
    const conn = (nav as unknown as Record<string, unknown>).connection as { effectiveType?: string } | undefined;
    if (conn?.effectiveType) {
      metadata.connectionType = conn.effectiveType;
    }
    if ('gpu' in nav) {
      metadata.webgpuAvailable = true;
    }
    if ((nav as unknown as Record<string, unknown>).ml) {
      metadata.webnnAvailable = true;
    }
  }
  if (typeof location !== 'undefined' && location.origin) {
    metadata.origin = location.origin;
  }
  try {
    metadata.timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
  } catch {
    // Intl may not be available in all environments.
  }
  const perfMemory = (performance as unknown as Record<string, unknown>).memory as
    | { jsHeapSizeLimit?: number; totalJSHeapSize?: number; usedJSHeapSize?: number }
    | undefined;
  if (perfMemory) {
    if (perfMemory.jsHeapSizeLimit) {
      metadata.jsHeapSizeLimitMB = Math.round(perfMemory.jsHeapSizeLimit / 1048576);
    }
    if (perfMemory.totalJSHeapSize) {
      metadata.totalJSHeapSizeMB = Math.round(perfMemory.totalJSHeapSize / 1048576);
    }
    if (perfMemory.usedJSHeapSize) {
      metadata.usedJSHeapSizeMB = Math.round(perfMemory.usedJSHeapSize / 1048576);
    }
  }
  return metadata;
};

const enqueue = (eventName: string, eventData: Record<string, unknown>): void => {
  if (queue.length >= MAX_QUEUE_SIZE) {
    return;
  }
  queue.push(
    JSON.stringify({
      name: eventName.toLowerCase(),
      time: new Date().toISOString(),
      ver: '4.0',
      iKey: IKEY_PREFIX,
      ext: {
        sdk: { ver: 'ORT-Web/1.0' },
        device: { localId: getDeviceId() },
      },
      data: eventData,
    }),
  );
};

const isRuntimeTelemetryEnabled = (): boolean => env.telemetry?.enabled !== false;

const isTelemetrySupported = (): boolean => !BUILD_DEFS.DISABLE_TELEMETRY && wasmTelemetrySupported;

const shouldEmitTelemetry = (): boolean => isTelemetrySupported() && isRuntimeTelemetryEnabled();

const emitTelemetryEvent = (eventName: string, eventData: Record<string, unknown>): void => {
  if (!shouldEmitTelemetry()) {
    return;
  }

  try {
    env.telemetry?.onEvent?.(eventName, eventData);
  } catch {
    // Observer errors must not disrupt the application.
  }

  try {
    enqueue(eventName, eventData);
  } catch {
    // Telemetry must not disrupt the application.
  }
};

const sendWithRetry = (payload: string, headers: Record<string, string>, attempt = 0): void => {
  if (!shouldEmitTelemetry()) {
    return;
  }

  fetch(COLLECTOR_URL, {
    method: 'POST',
    headers,
    body: payload,
    keepalive: true,
  }).then(
    (res) => {
      if (!res.ok && attempt < MAX_RETRIES && (res.status === 429 || res.status >= 500)) {
        const delay = 3000 * Math.pow(2, attempt) + Math.random() * 1000;
        setTimeout(() => {
          if (shouldEmitTelemetry()) {
            sendWithRetry(payload, headers, attempt + 1);
          }
        }, delay);
      }
    },
    () => {
      if (attempt < MAX_RETRIES) {
        const delay = 3000 * Math.pow(2, attempt) + Math.random() * 1000;
        setTimeout(() => {
          if (shouldEmitTelemetry()) {
            sendWithRetry(payload, headers, attempt + 1);
          }
        }, delay);
      }
    },
  );
};

/* eslint-disable @typescript-eslint/naming-convention */
const FETCH_HEADERS: Record<string, string> = {
  'Content-Type': 'application/x-json-stream',
  apikey: IKEY,
  'Client-Id': 'NO_AUTH',
  'cache-control': 'no-cache, no-store',
};
/* eslint-enable @typescript-eslint/naming-convention */

const flush = (useBeacon = false): void => {
  if (!isTelemetrySupported()) {
    return;
  }

  if (!isRuntimeTelemetryEnabled()) {
    queue = [];
    return;
  }

  if (queue.length === 0) {
    return;
  }
  const batch = queue.splice(0);
  const payload = batch.join('\n');

  if (useBeacon && typeof navigator !== 'undefined' && 'sendBeacon' in navigator) {
    const blob = new Blob([payload], { type: 'application/x-json-stream' });
    if (
      (navigator as unknown as { sendBeacon: (url: string, data: Blob) => boolean }).sendBeacon(COLLECTOR_URL, blob)
    ) {
      return;
    }
  }

  // eslint-disable-next-line @typescript-eslint/naming-convention
  sendWithRetry(payload, { ...FETCH_HEADERS, 'upload-time': Date.now().toString() });
};

const startFlushTimer = (): void => {
  if (flushTimer) {
    return;
  }
  flushTimer = setInterval(() => flush(), FLUSH_INTERVAL_MS);

  if (typeof globalThis !== 'undefined' && typeof globalThis.addEventListener === 'function') {
    globalThis.addEventListener('beforeunload', () => flush(true));
    globalThis.addEventListener('pagehide', () => flush(true));
    globalThis.addEventListener('visibilitychange', () => {
      if (typeof document !== 'undefined' && 'visibilityState' in document && document.visibilityState === 'hidden') {
        flush(true);
      }
    });
  }
};

export const initTelemetry = (module: OrtWasmModule): void => {
  wasmTelemetrySupported = false;

  if (BUILD_DEFS.DISABLE_TELEMETRY) {
    return;
  }

  try {
    wasmTelemetrySupported = !!module._OrtIsTelemetrySupported?.();
  } catch {
    wasmTelemetrySupported = false;
  }

  if (!wasmTelemetrySupported) {
    return;
  }

  startFlushTimer();

  // eslint-disable-next-line dot-notation
  (module as unknown as Record<string, unknown>)['__ortTelemetryCallback'] = (
    eventName: string,
    eventData: Record<string, unknown>,
  ) => {
    emitTelemetryEvent(eventName, eventData);
  };

  // Device info event — fires from JS before WASM loads, so it works even on incompatible browsers
  emitTelemetryEvent('deviceinfo', getBrowserMetadata());
};

export const logSessionModelInfo = (modelSizeBytes: number, inputCount?: number, outputCount?: number): void => {
  if (!isTelemetrySupported()) {
    return;
  }

  try {
    const info: Record<string, unknown> = { modelSizeBytes };
    if (inputCount !== undefined) {
      info.inputCount = inputCount;
    }
    if (outputCount !== undefined) {
      info.outputCount = outputCount;
    }
    emitTelemetryEvent('modelinfo', info);
  } catch {
    // Telemetry must not disrupt the application.
  }
};
