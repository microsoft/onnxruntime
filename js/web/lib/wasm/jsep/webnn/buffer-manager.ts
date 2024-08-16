// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { WebNNBackend } from '../backend-webnn';

// WebNN API currently does not have a TypeScript definition file. This file is a workaround with types generated from
// WebNN API specification.
// https://github.com/webmachinelearning/webnn/issues/677
/// <reference path="webnn.d.ts" />

export type BufferId = number;

/**
 * Manages BufferId to MLBuffer mapping.
 */
export interface BufferManager {
  /**
   * Reserve a new BufferId.
   */
  reserveBufferId(): BufferId;
  /**
   * Release a BufferId.
   */
  releaseBufferId(bufferId: BufferId): void;
  /**
   * Ensure a MLBuffer is created for the BufferId.
   */
  ensureBuffer(bufferId: BufferId, dataType: MLOperandDataType, dimensions: number[]): Promise<MLBuffer>;
  /**
   * Upload data to a MLBuffer.
   */
  upload(bufferId: BufferId, data: Uint8Array): void;
  /**
   * Download data from a MLBuffer.
   */
  download(bufferId: BufferId): Promise<ArrayBuffer>;
  /**
   * Release all buffers for a MLContext.
   */
  releaseBuffersForContext(mlContext: MLContext): void;
  /**
   * Register an externally created MLBuffer with a given MLContext and return a BufferId.
   */
  registerBuffer(mlContext: MLContext, mlBuffer: MLBuffer): BufferId;
}

let bufferGuid = 1;
const createNewBufferId = (): BufferId => bufferGuid++;

/**
 * BufferTracker tracks the MLBuffer and pending upload data.
 *
 * We need to track the MLBuffer and pending upload data because we delay the creation of MLBuffer until
 * we know the data type and dimensions. This is because future implementations of WebNN will only support creating
 * MLBuffers with dataTypes and dimensions.
 */
class BufferTracker {
  private mlBuffer?: MLBuffer;
  private activeUpload?: Uint8Array;

  constructor(
    private mlContext?: MLContext,
    buffer?: MLBuffer,
  ) {
    this.mlBuffer = buffer;
  }

  public get buffer(): MLBuffer | undefined {
    return this.mlBuffer;
  }

  public get context(): MLContext {
    if (!this.mlContext) {
      throw new Error('MLContext has not been set.');
    }
    return this.mlContext;
  }

  public set context(mlContext: MLContext) {
    if (this.mlContext && this.mlContext !== mlContext) {
      throw new Error('MLBuffer in use in a different MLContext.');
    }
    this.mlContext = mlContext;
  }

  public destroy(): void {
    this.mlBuffer?.destroy();
    this.mlBuffer = undefined;
  }

  public async ensureBuffer(dataType: MLOperandDataType, dimensions: number[]): Promise<MLBuffer> {
    if (this.mlBuffer) {
      return this.mlBuffer;
    }

    const buffer = await this.context.createBuffer({ dataType, dimensions });
    this.mlBuffer = buffer;

    if (this.activeUpload) {
      this.mlContext?.writeBuffer(buffer, this.activeUpload);
      this.activeUpload = undefined;
    }

    return buffer;
  }

  public upload(data: Uint8Array): void {
    if (!this.mlBuffer) {
      this.activeUpload = new Uint8Array(data);
      return;
    }

    this.mlContext?.writeBuffer(this.mlBuffer, data);
  }

  public async download(): Promise<ArrayBuffer> {
    if (this.activeUpload) {
      return this.activeUpload.buffer;
    }
    if (!this.mlBuffer) {
      throw new Error('Buffer has not been created.');
    }
    return this.context.readBuffer(this.mlBuffer);
  }
}

class BufferManagerImpl implements BufferManager {
  private buffersById = new Map<BufferId, BufferTracker>();
  private bufferIdsByContext = new Map<MLContext, Set<BufferId>>();

  constructor(private backend: WebNNBackend) {}

  public reserveBufferId(): BufferId {
    const bufferId = createNewBufferId();
    this.buffersById.set(bufferId, new BufferTracker());
    return bufferId;
  }

  public releaseBufferId(bufferId: BufferId): void {
    const bufferTracker = this.buffersById.get(bufferId);
    if (!bufferTracker) {
      return;
    }
    bufferTracker.destroy();
    this.buffersById.delete(bufferId);
    for (const [mlContext, buffers] of this.bufferIdsByContext) {
      if (buffers.has(bufferId)) {
        buffers.delete(bufferId);
        if (buffers.size === 0) {
          this.bufferIdsByContext.delete(mlContext);
        }
        break;
      }
    }
  }

  public async ensureBuffer(bufferId: BufferId, dataType: MLOperandDataType, dimensions: number[]): Promise<MLBuffer> {
    const buffer = this.buffersById.get(bufferId);
    if (!buffer) {
      throw new Error('Buffer not found.');
    }
    buffer.context = this.backend.currentContext;
    if (!this.bufferIdsByContext.has(this.backend.currentContext)) {
      this.bufferIdsByContext.set(this.backend.currentContext, new Set());
    }
    this.bufferIdsByContext.get(this.backend.currentContext)?.add(bufferId);
    return buffer.ensureBuffer(dataType, dimensions);
  }

  public upload(bufferId: BufferId, data: Uint8Array): void {
    this.buffersById.get(bufferId)!.upload(data);
  }

  public async download(bufferId: BufferId): Promise<ArrayBuffer> {
    return this.buffersById.get(bufferId)!.download();
  }

  public releaseBuffersForContext(mlContext: MLContext): void {
    const buffers = this.bufferIdsByContext.get(mlContext);
    if (!buffers) {
      return;
    }
    for (const bufferId of buffers) {
      this.buffersById.get(bufferId)!.destroy();
      this.buffersById.delete(bufferId);
    }
    this.bufferIdsByContext.delete(mlContext);
  }

  public registerBuffer(mlContext: MLContext, mlBuffer: MLBuffer): BufferId {
    for (const [bufferId, bufferTracker] of this.buffersById) {
      if (bufferTracker.buffer === mlBuffer) {
        if (bufferTracker.context !== mlContext) {
          throw new Error('MLBuffer cannot be registered with a different MLContext.');
        }
        return bufferId;
      }
    }
    const bufferId = createNewBufferId();
    this.buffersById.set(bufferId, new BufferTracker(mlContext, mlBuffer));
    let buffers = this.bufferIdsByContext.get(mlContext);
    if (!buffers) {
      buffers = new Set();
      this.bufferIdsByContext.set(mlContext, buffers);
    }
    buffers.add(bufferId);
    return bufferId;
  }
}

export const createBufferManager = (...args: ConstructorParameters<typeof BufferManagerImpl>): BufferManager =>
  new BufferManagerImpl(...args);
