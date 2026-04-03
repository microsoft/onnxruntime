// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { env } from 'onnxruntime-common';

import type { OrtWasmModule } from './wasm-types';

export interface TelemetryTransport {
  sendEvent(eventName: string, eventData: Record<string, unknown>): void;
  flush(): Promise<void>;
  shutdown(): Promise<void>;
}

let transport: TelemetryTransport | null = null;
let transportInitPromise: Promise<void> | null = null;
let pendingEvents: Array<{ name: string; data: Record<string, unknown> }> = [];

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
      if (uaData.platform) metadata.osPlatform = uaData.platform;
      if (uaData.architecture) metadata.architecture = uaData.architecture;
      if (uaData.mobile !== undefined) metadata.isMobile = uaData.mobile;
    }
  }
  return metadata;
};

export const initTelemetry = (module: OrtWasmModule): void => {
  const browserMetadata = getBrowserMetadata();

  (module as unknown as Record<string, unknown>)['__ortTelemetryCallback'] = (eventName: string, eventData: Record<string, unknown>) => {
    if (env.telemetry?.enabled === false) {
      return;
    }

    const enrichedData = eventName === 'ProcessInfo' ? { ...eventData, ...browserMetadata } : eventData;

    try {
      env.telemetry?.onEvent?.(eventName, enrichedData);
    } catch {
      // Observer errors must not disrupt the application.
    }

    if (transport) {
      transport.sendEvent(eventName, enrichedData);
    } else {
      pendingEvents.push({ name: eventName, data: enrichedData });
    }
  };

  if (env.telemetry?.enabled !== false) {
    transportInitPromise = initTransport().catch(() => {
      transport = null;
    });
  }
};

const initTransport = async (): Promise<void> => {
  const { OneDSTransport } = await import('./telemetry-1ds-transport.js');
  transport = new OneDSTransport();

  for (const event of pendingEvents) {
    transport.sendEvent(event.name, event.data);
  }
  pendingEvents = [];
};

export const shutdownTelemetry = async (): Promise<void> => {
  if (transportInitPromise) {
    await transportInitPromise;
  }
  if (transport) {
    await transport.flush();
    await transport.shutdown();
    transport = null;
  }
  pendingEvents = [];
};
