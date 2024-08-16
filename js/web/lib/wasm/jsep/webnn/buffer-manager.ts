// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { WebNNBackend } from '../backend-webnn';
import { LOG_DEBUG } from '../log';

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
  ensureBuffer(
    bufferId: BufferId,
    dataType: MLOperandDataType,
    dimensions: readonly number[],
    copyOld: boolean,
  ): Promise<MLBuffer>;
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
  registerBuffer(mlContext: MLContext, mlBuffer: MLBuffer, dataType: MLOperandDataType, dimensions: number[]): BufferId;
}

let bufferGuid = 1;
const createNewBufferId = (): BufferId => bufferGuid++;

export type MLBufferEntry = [MLBuffer, MLOperandDataType, readonly number[]];

/**
 * BufferTracker tracks the MLBuffer and pending upload data.
 *
 * We need to track the MLBuffer and pending upload data because we delay the creation of MLBuffer until
 * we know the data type and dimensions. This is because future implementations of WebNN will only support creating
 * MLBuffers with dataTypes and dimensions.
 */
class BufferTracker {
  private bufferEntry?: MLBufferEntry;
  private activeUpload?: Uint8Array;
  private bufferCache: MLBufferEntry[];

  constructor(
    private mlContext?: MLContext,
    bufferEntry?: MLBufferEntry,
  ) {
    this.bufferEntry = bufferEntry;
    this.bufferCache = bufferEntry ? [bufferEntry] : [];
  }

  public get buffer(): MLBuffer | undefined {
    return this.bufferEntry?.[0];
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
    for (const [mlBuffer] of this.bufferCache) {
      mlBuffer.destroy();
    }
    this.bufferCache = [];
    this.bufferEntry = undefined;
  }

  public trySelectBuffer(context: MLContext, tryMlBuffer: MLBuffer): boolean {
    for (const [mlBuffer, dataType, dimensions] of this.bufferCache) {
      if (tryMlBuffer === mlBuffer) {
        if (this.context !== context) {
          throw new Error('MLBuffer cannot be registered with a different MLContext.');
        }
        this.bufferEntry = [mlBuffer, dataType, dimensions];
        return true;
      }
    }
    return false;
  }

  public async ensureBuffer(
    dataType: MLOperandDataType,
    dimensions: readonly number[],
    copyOld: boolean,
  ): Promise<MLBuffer> {
    if (this.bufferEntry) {
      const [mlBuffer, existingDataType, existingDimensions] = this.bufferEntry;
      if (existingDataType === dataType && existingDimensions.every((v, i) => v === dimensions[i])) {
        return mlBuffer;
      }
    }

    for (const [mlBuffer, existingDataType, existingDimensions] of this.bufferCache) {
      if (existingDataType === dataType && existingDimensions.every((v, i) => v === dimensions[i])) {
        if (copyOld && this.bufferEntry) {
          // WebNN does not support copyBufferToBuffer, so we need to read and write the buffers.
          LOG_DEBUG(
            'verbose',
            () =>
              `[WebNN] Slowdown may occur, having to copy existing buffer {dataType: ${
                dataType
              }, dimensions: ${dimensions}}`,
          );
          const data = await this.context.readBuffer(this.bufferEntry[0]);
          this.context.writeBuffer(mlBuffer, data);
        }
        this.bufferEntry = [mlBuffer, existingDataType, existingDimensions];
        return mlBuffer;
      }
    }
    LOG_DEBUG('verbose', () => `[WebNN] createBuffer {dataType: ${dataType}, dimensions: ${dimensions}}`);
    const buffer = await this.context.createBuffer({ dataType, dimensions });
    this.bufferEntry = [buffer, dataType, dimensions];
    this.bufferCache.push(this.bufferEntry);

    if (this.activeUpload) {
      this.mlContext?.writeBuffer(buffer, this.activeUpload);
      this.activeUpload = undefined;
    }

    return buffer;
  }

  public upload(data: Uint8Array): void {
    if (!this.bufferEntry) {
      this.activeUpload = new Uint8Array(data);
      return;
    }

    this.mlContext?.writeBuffer(this.bufferEntry[0], data);
  }

  public async download(): Promise<ArrayBuffer> {
    if (this.activeUpload) {
      return this.activeUpload.buffer;
    }
    if (!this.bufferEntry) {
      throw new Error('Buffer has not been created.');
    }
    return this.context.readBuffer(this.bufferEntry[0]);
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

  public async ensureBuffer(
    bufferId: BufferId,
    dataType: MLOperandDataType,
    dimensions: number[],
    copyOld: boolean,
  ): Promise<MLBuffer> {
    LOG_DEBUG(
      'verbose',
      () =>
        `[WebNN] BufferManager.ensureBuffer {bufferId: ${bufferId}, dataType: ${
          dataType
        }, dimensions: ${dimensions}}, copyOld: ${copyOld}`,
    );
    const buffer = this.buffersById.get(bufferId);
    if (!buffer) {
      throw new Error('Buffer not found.');
    }
    buffer.context = this.backend.currentContext;
    if (!this.bufferIdsByContext.has(this.backend.currentContext)) {
      this.bufferIdsByContext.set(this.backend.currentContext, new Set());
    }
    this.bufferIdsByContext.get(this.backend.currentContext)?.add(bufferId);
    return buffer.ensureBuffer(dataType, dimensions, copyOld);
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

  public registerBuffer(
    mlContext: MLContext,
    mlBuffer: MLBuffer,
    dataType: MLOperandDataType,
    dimensions: readonly number[],
  ): BufferId {
    for (const [bufferId, bufferTracker] of this.buffersById) {
      if (bufferTracker.trySelectBuffer(mlContext, mlBuffer)) {
        return bufferId;
      }
    }
    const bufferId = createNewBufferId();
    this.buffersById.set(bufferId, new BufferTracker(mlContext, [mlBuffer, dataType, dimensions]));
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
