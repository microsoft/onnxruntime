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
const MAX_BATCH_SIZE_BYTES = 100 * 1024;
const MAX_UNLOAD_BATCH_SIZE_BYTES = 60 * 1024;
const MAX_RETRIES = 6;
const BASE_BACKOFF_MS = 3000;
const MAX_BACKOFF_MS = 600_000;
const RETRY_RANDOMIZATION_LOWER_THRESHOLD = 0.8;
const RETRY_RANDOMIZATION_UPPER_THRESHOLD = 1.2;
const OFFLINE_RETRY_MULTIPLIER = 10;
const DEVICE_ID_KEY = 'ort_device_id';
const DEVICE_ID_COOKIE_NAME = DEVICE_ID_KEY;
const DEVICE_ID_COOKIE_SCOPE_TEST_NAME = 'ort_device_id_scope_test';
const DEVICE_ID_COOKIE_MAX_AGE_SECONDS = 365 * 24 * 60 * 60;
const DEVICE_ID_HASH_PREFIX = 'c:';
const PENDING_TELEMETRY_SESSION_STORAGE_KEY = 'ort_pending_telemetry';
const LEGACY_PROCESS_INFO_SESSION_STORAGE_KEY = 'ort_pending_processinfo';
const BYTES_PER_MEBIBYTE = 1024 * 1024;
const PROCESS_INFO_EVENT = 'ProcessInfo';

type QueuedTelemetryEvent = {
  id: string;
  eventName: string;
  payload: string;
  payloadBytes: number;
};

type StoredQueuedTelemetryEvent = Pick<QueuedTelemetryEvent, 'id' | 'eventName' | 'payload'>;

const queue: QueuedTelemetryEvent[] = [];
const pendingCachedEvents: QueuedTelemetryEvent[] = [];
let flushTimer: ReturnType<typeof setInterval> | null = null;
let deviceId: string | null = null;
let protectedDeviceId: string | null = null;
let protectedDeviceIdInitialized = false;
let protectedDeviceIdPromise: Promise<string | null> | null = null;
let wasmTelemetrySupported = false;
let deviceIdCookieDomain: string | null = null;
let deviceIdCookieAvailable = false;
let deviceIdCookieResolved = false;
let nextQueuedTelemetryEventId = 0;
const pendingEnqueues = new Set<Promise<void>>();

const getPayloadSize = (payload: string): number => {
  if (typeof TextEncoder !== 'undefined') {
    return new TextEncoder().encode(payload).byteLength;
  }

  if (typeof Blob !== 'undefined') {
    return new Blob([payload]).size;
  }

  return payload.length;
};

const createQueuedTelemetryEvent = (
  eventName: string,
  payload: string,
  id = `${Date.now()}-${nextQueuedTelemetryEventId++}`,
): QueuedTelemetryEvent => ({
  id,
  eventName,
  payload,
  payloadBytes: getPayloadSize(payload),
});

const isStoredQueuedTelemetryEvent = (value: unknown): value is StoredQueuedTelemetryEvent => {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const entry = value as Record<string, unknown>;
  return typeof entry.id === 'string' && typeof entry.eventName === 'string' && typeof entry.payload === 'string';
};

const persistPendingTelemetryCache = (): void => {
  if (typeof sessionStorage === 'undefined') {
    return;
  }

  try {
    if (pendingCachedEvents.length === 0) {
      sessionStorage.removeItem(PENDING_TELEMETRY_SESSION_STORAGE_KEY);
      sessionStorage.removeItem(LEGACY_PROCESS_INFO_SESSION_STORAGE_KEY);
      return;
    }

    const storedEntries: StoredQueuedTelemetryEvent[] = pendingCachedEvents.map(({ id, eventName, payload }) => ({
      id,
      eventName,
      payload,
    }));

    sessionStorage.setItem(PENDING_TELEMETRY_SESSION_STORAGE_KEY, JSON.stringify(storedEntries));
    sessionStorage.removeItem(LEGACY_PROCESS_INFO_SESSION_STORAGE_KEY);
  } catch {
    // Ignore storage failures so telemetry does not disrupt the application.
  }
};

const removePendingTelemetryEntries = (entries: readonly QueuedTelemetryEvent[]): void => {
  if (entries.length === 0 || pendingCachedEvents.length === 0) {
    return;
  }

  const entryIds = new Set(entries.map((entry) => entry.id));
  let removed = false;
  for (let i = pendingCachedEvents.length - 1; i >= 0; i--) {
    if (entryIds.has(pendingCachedEvents[i].id)) {
      pendingCachedEvents.splice(i, 1);
      removed = true;
    }
  }

  if (removed) {
    persistPendingTelemetryCache();
  }
};

const restorePendingTelemetryCache = (): void => {
  if (typeof sessionStorage === 'undefined') {
    return;
  }

  try {
    const storedTelemetryPayload = sessionStorage.getItem(PENDING_TELEMETRY_SESSION_STORAGE_KEY);
    const legacyProcessInfoPayload =
      storedTelemetryPayload === null ? sessionStorage.getItem(LEGACY_PROCESS_INFO_SESSION_STORAGE_KEY) : null;

    const restoredEntries: StoredQueuedTelemetryEvent[] = storedTelemetryPayload
      ? (JSON.parse(storedTelemetryPayload) as unknown[]).filter(isStoredQueuedTelemetryEvent)
      : legacyProcessInfoPayload
        ? [{ id: `${Date.now()}-legacy-processinfo`, eventName: PROCESS_INFO_EVENT, payload: legacyProcessInfoPayload }]
        : [];

    if (restoredEntries.length === 0) {
      return;
    }

    const existingIds = new Set([...queue, ...pendingCachedEvents].map((entry) => entry.id));
    let restoredAny = false;
    for (const entry of restoredEntries) {
      if (existingIds.has(entry.id)) {
        continue;
      }

      const restoredEntry = createQueuedTelemetryEvent(entry.eventName, entry.payload, entry.id);
      queue.push(restoredEntry);
      pendingCachedEvents.push(restoredEntry);
      existingIds.add(entry.id);
      restoredAny = true;
    }

    if (restoredAny || legacyProcessInfoPayload) {
      persistPendingTelemetryCache();
    }
  } catch {
    // Ignore storage failures so telemetry does not disrupt the application.
  }
};

const getLocalStorageDeviceId = (): string | null => {
  if (typeof localStorage === 'undefined') {
    return null;
  }

  try {
    return localStorage.getItem(DEVICE_ID_KEY);
  } catch {
    return null;
  }
};

const persistLocalStorageDeviceId = (rawDeviceId: string): void => {
  if (typeof localStorage === 'undefined') {
    return;
  }

  try {
    localStorage.setItem(DEVICE_ID_KEY, rawDeviceId);
  } catch {
    // Ignore storage failures so telemetry does not disrupt the application.
  }
};

const getCookieValue = (name: string): string | null => {
  if (typeof document === 'undefined') {
    return null;
  }

  try {
    const encodedName = `${encodeURIComponent(name)}=`;
    for (const cookie of document.cookie.split(';')) {
      const trimmedCookie = cookie.trim();
      if (trimmedCookie.startsWith(encodedName)) {
        return decodeURIComponent(trimmedCookie.slice(encodedName.length));
      }
    }
  } catch {
    // Ignore cookie failures so telemetry does not disrupt the application.
  }

  return null;
};

const isIpAddress = (hostname: string): boolean => /^\d{1,3}(?:\.\d{1,3}){3}$/.test(hostname) || hostname.includes(':');

const shouldUseSecureCookies = (): boolean =>
  typeof location !== 'undefined' && (location.protocol === 'https:' || location.origin?.startsWith('https://'));

const setCookieValue = (name: string, value: string, maxAgeSeconds: number, domain?: string): boolean => {
  if (typeof document === 'undefined') {
    return false;
  }

  try {
    const attributes = [
      `${encodeURIComponent(name)}=${encodeURIComponent(value)}`,
      `Max-Age=${maxAgeSeconds}`,
      'Path=/',
      'SameSite=Lax',
    ];
    if (domain) {
      attributes.push(`Domain=${domain}`);
    }
    if (shouldUseSecureCookies()) {
      attributes.push('Secure');
    }

    document.cookie = attributes.join('; ');
    return getCookieValue(name) === value;
  } catch {
    return false;
  }
};

const clearCookieValue = (name: string, domain?: string): void => {
  if (typeof document === 'undefined') {
    return;
  }

  try {
    const attributes = [`${encodeURIComponent(name)}=`, 'Max-Age=0', 'Path=/', 'SameSite=Lax'];
    if (domain) {
      attributes.push(`Domain=${domain}`);
    }
    if (shouldUseSecureCookies()) {
      attributes.push('Secure');
    }
    document.cookie = attributes.join('; ');
  } catch {
    // Ignore cookie failures so telemetry does not disrupt the application.
  }
};

const getCookieDomainCandidates = (): string[] => {
  if (typeof location === 'undefined' || !location.hostname || isIpAddress(location.hostname)) {
    return [];
  }

  const hostnameLabels = location.hostname.split('.').filter(Boolean);
  if (hostnameLabels.length < 2) {
    return [];
  }

  const candidates: string[] = [];
  for (let i = hostnameLabels.length - 2; i >= 0; i--) {
    candidates.push(hostnameLabels.slice(i).join('.'));
  }

  return candidates;
};

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

const resolveDeviceIdCookieStorage = (): { available: boolean; domain: string | null } => {
  if (deviceIdCookieResolved) {
    return { available: deviceIdCookieAvailable, domain: deviceIdCookieDomain };
  }

  const testValue = generateUUID();
  for (const domain of getCookieDomainCandidates()) {
    if (setCookieValue(DEVICE_ID_COOKIE_SCOPE_TEST_NAME, testValue, 1, domain)) {
      clearCookieValue(DEVICE_ID_COOKIE_SCOPE_TEST_NAME, domain);
      deviceIdCookieDomain = domain;
      deviceIdCookieAvailable = true;
      deviceIdCookieResolved = true;
      return { available: true, domain };
    }
  }

  if (setCookieValue(DEVICE_ID_COOKIE_SCOPE_TEST_NAME, testValue, 1)) {
    clearCookieValue(DEVICE_ID_COOKIE_SCOPE_TEST_NAME);
    deviceIdCookieDomain = null;
    deviceIdCookieAvailable = true;
    deviceIdCookieResolved = true;
    return { available: true, domain: null };
  }

  deviceIdCookieDomain = null;
  deviceIdCookieAvailable = false;
  deviceIdCookieResolved = true;
  return { available: false, domain: null };
};

const getCookieDeviceId = (): string | null => getCookieValue(DEVICE_ID_COOKIE_NAME);

const persistCookieDeviceId = (rawDeviceId: string): void => {
  const cookieStorage = resolveDeviceIdCookieStorage();
  if (!cookieStorage.available) {
    return;
  }

  if (
    !setCookieValue(
      DEVICE_ID_COOKIE_NAME,
      rawDeviceId,
      DEVICE_ID_COOKIE_MAX_AGE_SECONDS,
      cookieStorage.domain ?? undefined,
    )
  ) {
    deviceIdCookieAvailable = false;
  }
};

const getDeviceId = (): string => {
  if (deviceId) {
    return deviceId;
  }

  const cookieDeviceId = getCookieDeviceId();
  if (cookieDeviceId) {
    deviceId = cookieDeviceId;
    persistCookieDeviceId(deviceId);
    persistLocalStorageDeviceId(deviceId);
    return deviceId;
  }

  const localStorageDeviceId = getLocalStorageDeviceId();
  if (localStorageDeviceId) {
    deviceId = localStorageDeviceId;
    persistCookieDeviceId(deviceId);
    persistLocalStorageDeviceId(deviceId);
    return deviceId;
  }

  deviceId = generateUUID();
  persistCookieDeviceId(deviceId);
  persistLocalStorageDeviceId(deviceId);
  return deviceId;
};

const toHex = (bytes: Uint8Array): string => Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');

const hashDeviceId = async (rawDeviceId: string): Promise<string | null> => {
  if (typeof crypto === 'undefined' || !crypto.subtle?.digest || typeof TextEncoder === 'undefined') {
    return null;
  }

  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(rawDeviceId));
  return `${DEVICE_ID_HASH_PREFIX}${toHex(new Uint8Array(digest))}`;
};

const getProtectedDeviceId = async (): Promise<string | null> => {
  if (protectedDeviceIdInitialized) {
    return protectedDeviceId;
  }

  if (!protectedDeviceIdPromise) {
    const rawDeviceId = getDeviceId();
    protectedDeviceIdPromise = hashDeviceId(rawDeviceId)
      .then((hashedDeviceId) => {
        protectedDeviceId = hashedDeviceId;
        protectedDeviceIdInitialized = true;
        return hashedDeviceId;
      })
      .catch(() => {
        protectedDeviceId = null;
        protectedDeviceIdInitialized = true;
        return null;
      })
      .finally(() => {
        protectedDeviceIdPromise = null;
      });
  }

  return protectedDeviceIdPromise;
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
  if (typeof location !== 'undefined') {
    if (location.origin) {
      metadata.origin = location.origin;
    }
  }
  const perfMemory = (performance as unknown as Record<string, unknown>).memory as
    | { jsHeapSizeLimit?: number; totalJSHeapSize?: number; usedJSHeapSize?: number }
    | undefined;
  if (perfMemory) {
    if (perfMemory.jsHeapSizeLimit) {
      metadata.jsHeapSizeLimitMB = Math.round(perfMemory.jsHeapSizeLimit / BYTES_PER_MEBIBYTE);
    }
    if (perfMemory.totalJSHeapSize) {
      metadata.totalJSHeapSizeMB = Math.round(perfMemory.totalJSHeapSize / BYTES_PER_MEBIBYTE);
    }
    if (perfMemory.usedJSHeapSize) {
      metadata.usedJSHeapSizeMB = Math.round(perfMemory.usedJSHeapSize / BYTES_PER_MEBIBYTE);
    }
  }
  return metadata;
};

const isRuntimeTelemetryEnabled = (): boolean => env.telemetry?.enabled !== false;

const isTelemetrySupported = (): boolean => !BUILD_DEFS.DISABLE_TELEMETRY && wasmTelemetrySupported;

const shouldAlwaysEmitEvent = (eventName: string): boolean => eventName === PROCESS_INFO_EVENT;

const shouldEmitEvent = (eventName: string): boolean =>
  isTelemetrySupported() && (isRuntimeTelemetryEnabled() || shouldAlwaysEmitEvent(eventName));

const canSendBatch = (allowWhenRuntimeTelemetryDisabled: boolean): boolean =>
  isTelemetrySupported() && (isRuntimeTelemetryEnabled() || allowWhenRuntimeTelemetryDisabled);

const isBrowserOffline = (): boolean =>
  typeof navigator !== 'undefined' && 'onLine' in navigator && navigator.onLine === false;

const shouldRetryStatus = (statusCode: number): boolean =>
  statusCode === 408 || statusCode === 429 || statusCode >= 500;

const getRetryDelayMs = (attempt: number, multiplier = 1): number => {
  const randomizationFactor =
    RETRY_RANDOMIZATION_LOWER_THRESHOLD +
    Math.random() * (RETRY_RANDOMIZATION_UPPER_THRESHOLD - RETRY_RANDOMIZATION_LOWER_THRESHOLD);

  return Math.min(BASE_BACKOFF_MS * Math.pow(2, attempt) * randomizationFactor * multiplier, MAX_BACKOFF_MS);
};

const enqueue = async (eventName: string, eventData: Record<string, unknown>): Promise<void> => {
  if (!shouldEmitEvent(eventName) || pendingCachedEvents.length >= MAX_QUEUE_SIZE) {
    return;
  }

  const localId = await getProtectedDeviceId();
  if (!shouldEmitEvent(eventName) || pendingCachedEvents.length >= MAX_QUEUE_SIZE) {
    return;
  }

  const ext: Record<string, unknown> = {
    sdk: { ver: 'ORT-Web/1.0' },
  };
  if (localId) {
    ext.device = { localId };
  }

  const payload = JSON.stringify({
    name: eventName.toLowerCase(),
    time: new Date().toISOString(),
    ver: '4.0',
    iKey: IKEY_PREFIX,
    ext,
    data: eventData,
  });

  const queuedEvent = createQueuedTelemetryEvent(eventName, payload);
  queue.push(queuedEvent);
  pendingCachedEvents.push(queuedEvent);
  persistPendingTelemetryCache();
};

const getEventData = (eventName: string, eventData: Record<string, unknown>): Record<string, unknown> =>
  eventName === PROCESS_INFO_EVENT ? { ...getBrowserMetadata(), ...eventData } : eventData;

const emitTelemetryEvent = (eventName: string, eventData: Record<string, unknown>): void => {
  if (!shouldEmitEvent(eventName)) {
    return;
  }

  try {
    env.telemetry?.onEvent?.(eventName, eventData);
  } catch {
    // Observer errors must not disrupt the application.
  }

  const enqueuePromise = enqueue(eventName, eventData).catch(() => {
    // Telemetry must not disrupt the application.
  });
  pendingEnqueues.add(enqueuePromise);
  void enqueuePromise.finally(() => pendingEnqueues.delete(enqueuePromise));
};

/* eslint-disable @typescript-eslint/naming-convention */
const FETCH_HEADERS: Record<string, string> = {
  'Content-Type': 'application/x-json-stream',
  apikey: IKEY,
  'Client-Id': 'NO_AUTH',
  'cache-control': 'no-cache, no-store',
};
/* eslint-enable @typescript-eslint/naming-convention */

const buildBatchPayload = (batch: readonly QueuedTelemetryEvent[]): string =>
  batch.map((entry) => entry.payload).join('\n');

const getFetchHeaders = (): Record<string, string> => ({
  ...FETCH_HEADERS,
  // eslint-disable-next-line @typescript-eslint/naming-convention
  'upload-time': Date.now().toString(),
});

type BatchRetrySender = (
  batch: readonly QueuedTelemetryEvent[],
  allowWhenRuntimeTelemetryDisabled: boolean,
  attempt?: number,
) => void;

const sendBatchWithRetry: BatchRetrySender = (
  batch: readonly QueuedTelemetryEvent[],
  allowWhenRuntimeTelemetryDisabled: boolean,
  attempt = 0,
): void => {
  if (!canSendBatch(allowWhenRuntimeTelemetryDisabled)) {
    return;
  }

  if (isBrowserOffline()) {
    // eslint-disable-next-line @typescript-eslint/no-use-before-define
    scheduleBatchRetry(batch, allowWhenRuntimeTelemetryDisabled, attempt, OFFLINE_RETRY_MULTIPLIER);
    return;
  }

  if (typeof fetch !== 'function') {
    return;
  }

  fetch(COLLECTOR_URL, {
    method: 'POST',
    headers: getFetchHeaders(),
    body: buildBatchPayload(batch),
    keepalive: true,
  }).then(
    (res) => {
      if (res.ok) {
        removePendingTelemetryEntries(batch);
        return;
      }

      if (shouldRetryStatus(res.status)) {
        // eslint-disable-next-line @typescript-eslint/no-use-before-define
        scheduleBatchRetry(batch, allowWhenRuntimeTelemetryDisabled, attempt);
      } else {
        removePendingTelemetryEntries(batch);
      }
    },
    () => {
      // eslint-disable-next-line @typescript-eslint/no-use-before-define
      scheduleBatchRetry(batch, allowWhenRuntimeTelemetryDisabled, attempt);
    },
  );
};

const scheduleBatchRetry = (
  batch: readonly QueuedTelemetryEvent[],
  allowWhenRuntimeTelemetryDisabled: boolean,
  attempt: number,
  multiplier = 1,
): void => {
  if (attempt >= MAX_RETRIES) {
    removePendingTelemetryEntries(batch);
    return;
  }

  const delay = getRetryDelayMs(attempt, multiplier);
  setTimeout(() => {
    if (canSendBatch(allowWhenRuntimeTelemetryDisabled)) {
      sendBatchWithRetry(batch, allowWhenRuntimeTelemetryDisabled, attempt + 1);
    }
  }, delay);
};

const takeBatch = (maxBatchSizeBytes: number): QueuedTelemetryEvent[] => {
  const batch: QueuedTelemetryEvent[] = [];
  let batchSizeBytes = 0;

  while (queue.length > 0) {
    const nextEntry = queue[0];
    if (!shouldEmitEvent(nextEntry.eventName)) {
      queue.shift();
      removePendingTelemetryEntries([nextEntry]);
      continue;
    }

    const entrySizeBytes = nextEntry.payloadBytes + (batch.length > 0 ? 1 : 0);
    if (batch.length > 0 && batchSizeBytes + entrySizeBytes > maxBatchSizeBytes) {
      break;
    }

    queue.shift();
    batch.push(nextEntry);
    batchSizeBytes += entrySizeBytes;
  }

  return batch;
};

const flushBatch = (batch: readonly QueuedTelemetryEvent[], useBeacon: boolean): void => {
  const allowWhenRuntimeTelemetryDisabled = batch.every((entry) => shouldAlwaysEmitEvent(entry.eventName));
  const payload = buildBatchPayload(batch);

  if (useBeacon && typeof navigator !== 'undefined' && 'sendBeacon' in navigator) {
    const blob = new Blob([payload], { type: 'application/x-json-stream' });
    if (
      (navigator as unknown as { sendBeacon: (url: string, data: Blob) => boolean }).sendBeacon(COLLECTOR_URL, blob)
    ) {
      removePendingTelemetryEntries(batch);
      return;
    }
  }

  sendBatchWithRetry(batch, allowWhenRuntimeTelemetryDisabled);
};

const flush = async (useBeacon = false): Promise<void> => {
  if (!isTelemetrySupported()) {
    return;
  }

  if (pendingEnqueues.size > 0) {
    await Promise.allSettled([...pendingEnqueues]);
  }

  if (!useBeacon && isBrowserOffline()) {
    return;
  }

  const maxBatchSizeBytes = useBeacon ? MAX_UNLOAD_BATCH_SIZE_BYTES : MAX_BATCH_SIZE_BYTES;
  while (queue.length > 0) {
    const batch = takeBatch(maxBatchSizeBytes);
    if (batch.length === 0) {
      return;
    }

    flushBatch(batch, useBeacon);
  }
};

const startFlushTimer = (): void => {
  if (flushTimer) {
    return;
  }
  flushTimer = setInterval(() => {
    void flush();
  }, FLUSH_INTERVAL_MS);

  if (typeof globalThis !== 'undefined' && typeof globalThis.addEventListener === 'function') {
    globalThis.addEventListener('beforeunload', () => {
      void flush(true);
    });
    globalThis.addEventListener('pagehide', () => {
      void flush(true);
    });
    globalThis.addEventListener('visibilitychange', () => {
      if (typeof document !== 'undefined' && 'visibilityState' in document && document.visibilityState === 'hidden') {
        void flush(true);
      }
    });
    globalThis.addEventListener('online', () => {
      void flush();
    });
  }
};

export const flushTelemetry = async (useBeacon = false): Promise<void> => {
  await flush(useBeacon);
};

// Internal test hook to reset module state between unit tests.
export const resetTelemetryForTesting = (): void => {
  queue.length = 0;
  pendingCachedEvents.length = 0;
  if (flushTimer) {
    clearInterval(flushTimer);
    flushTimer = null;
  }
  deviceId = null;
  protectedDeviceId = null;
  protectedDeviceIdInitialized = false;
  protectedDeviceIdPromise = null;
  wasmTelemetrySupported = false;
  deviceIdCookieDomain = null;
  deviceIdCookieAvailable = false;
  deviceIdCookieResolved = false;
  nextQueuedTelemetryEventId = 0;
  pendingEnqueues.clear();
};

export const initTelemetry = (module: OrtWasmModule): void => {
  wasmTelemetrySupported = false;

  if (BUILD_DEFS.DISABLE_TELEMETRY) {
    return;
  }

  try {
    // eslint-disable-next-line no-underscore-dangle
    wasmTelemetrySupported = !!module._OrtIsTelemetrySupported?.();
  } catch {
    wasmTelemetrySupported = false;
  }

  if (!wasmTelemetrySupported) {
    return;
  }

  restorePendingTelemetryCache();
  startFlushTimer();
  if (isRuntimeTelemetryEnabled()) {
    void getProtectedDeviceId();
  }

  // eslint-disable-next-line dot-notation
  (module as unknown as Record<string, unknown>)['__ortTelemetryCallback'] = (
    eventName: string,
    eventData: Record<string, unknown>,
  ) => {
    emitTelemetryEvent(eventName, getEventData(eventName, eventData));
  };
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
