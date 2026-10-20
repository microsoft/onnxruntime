// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { resolveBackendForLoraAdapter } from './backend-impl.js';
import { Backend, LoraAdapterHandler } from './backend.js';
import { LoraAdapter as LoraAdapterInterface } from './lora-adapter.js';
import { TRACE_FUNC_BEGIN, TRACE_FUNC_END } from './trace.js';

export class LoraAdapter implements LoraAdapterInterface {
  private constructor(backend: Backend, handler: LoraAdapterHandler) {
    this.backend = backend;
    this.handler = handler;
  }

  static async create(uriOrBuffer: string | Uint8Array): Promise<LoraAdapterInterface> {
    TRACE_FUNC_BEGIN();
    if (typeof uriOrBuffer !== 'string' && !(uriOrBuffer instanceof Uint8Array)) {
      throw new TypeError("Unexpected argument[0]: must be 'path' or 'buffer'.");
    }

    const backend = await resolveBackendForLoraAdapter();
    const handler = await backend.createLoraAdapterHandler!(uriOrBuffer);
    TRACE_FUNC_END();
    return new LoraAdapter(backend, handler);
  }

  async release(): Promise<void> {
    return this.handler.dispose();
  }

  /**
   * The backend that created this adapter. An adapter can only be used by sessions of the same backend.
   */
  readonly backend: Backend;
  readonly handler: LoraAdapterHandler;
}
