// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import type { TelemetryTransport } from './telemetry';
import type { AppInsightsCore } from '@microsoft/1ds-core-js';

const TENANT_TOKEN = atob('NWFkOTYzYmQ0YjNhNDExOGE0ODE0MDFjYzAyMTE4NzUtMGNiNDUxNTktNDZmNS00NDk1LWI0ZTUtZDA4OTAzNTVlOTY0LTY3OTc=');
const COLLECTOR_URL = 'https://mobile.events.data.microsoft.com/OneCollector/1.0';

export class OneDSTransport implements TelemetryTransport {
  private core: AppInsightsCore | null = null;
  private initialized = false;

  constructor() {
    this.init();
  }

  private async init(): Promise<void> {
    try {
      // Dynamic import so bundle tools can tree-shake / code-split
      const [{ AppInsightsCore }, { PostChannel }] = await Promise.all([
        import('@microsoft/1ds-core-js'),
        import('@microsoft/1ds-post-js'),
      ]);

      this.core = new AppInsightsCore();
      const channel = new PostChannel();

      this.core.initialize(
        {
          instrumentationKey: TENANT_TOKEN,
          endpointUrl: COLLECTOR_URL,
          extensions: [channel],
          extensionConfig: {
            [channel.identifier]: {
              // Batch events and send at regular intervals
              eventsLimitInMem: 500,
            },
          },
        },
        [],
      );

      this.initialized = true;
    } catch {
      // If the 1DS SDK packages are not installed, this transport is a no-op.
      // Telemetry observation via env.telemetry.onEvent still works.
      this.initialized = false;
    }
  }

  sendEvent(eventName: string, eventData: Record<string, unknown>): void {
    if (!this.initialized || !this.core) {
      return;
    }

    this.core.track({
      name: eventName.toLowerCase(),
      data: {
        ...eventData,
        platform: 'WebAssembly',
      },
    });
  }

  async flush(): Promise<void> {
    if (this.initialized && this.core) {
      this.core.flush();
    }
  }

  async shutdown(): Promise<void> {
    if (this.initialized && this.core) {
      this.core.unload();
      this.core = null;
      this.initialized = false;
    }
  }
}
