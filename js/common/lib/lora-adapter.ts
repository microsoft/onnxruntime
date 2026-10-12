// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { LoraAdapter as LoraAdapterImpl } from './lora-adapter-impl.js';

/* eslint-disable @typescript-eslint/no-redeclare */

/**
 * Represent a LoRA adapter loaded from a file in the ONNX Runtime LoRA adapter format (.onnx_adapter).
 *
 * An adapter is independent of any inference session. It can be activated for a single run by passing it in
 * `InferenceSession.RunOptions.activeLoraAdapters`.
 *
 * This feature is available only in WebAssembly backend, and is not supported yet in proxy mode (`env.wasm.proxy`).
 * See `InferenceSession.RunOptions.activeLoraAdapters` for other limitations.
 */
export interface LoraAdapter {
  /**
   * Release the LoRA adapter and the underlying resources.
   *
   * The adapter must not be released while a run that uses it is still in progress.
   */
  release(): Promise<void>;
}

export interface LoraAdapterFactory {
  /**
   * Create a new LoRA adapter and load it asynchronously from a LoRA adapter file.
   *
   * @param uri - The URI or file path of the LoRA adapter file to load.
   * @returns A promise that resolves to a LoraAdapter object.
   */
  create(uri: string): Promise<LoraAdapter>;

  /**
   * Create a new LoRA adapter and load it asynchronously from a Uint8Array.
   *
   * @param buffer - A Uint8Array representation of a LoRA adapter file.
   * @returns A promise that resolves to a LoraAdapter object.
   */
  create(buffer: Uint8Array): Promise<LoraAdapter>;
}

// eslint-disable-next-line @typescript-eslint/naming-convention
export const LoraAdapter: LoraAdapterFactory = LoraAdapterImpl;
