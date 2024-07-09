// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// WebNN API currently does not have a TypeScript definition file. This file is a workaround with types generated from
// WebNN API specification.
// https://github.com/webmachinelearning/webnn/issues/677
/// <reference path="webnn/webnn.d.ts" />

import {Tensor} from 'onnxruntime-common';

import {createView} from './tensor-view';
import {BufferId, BufferManager, createBufferManager} from './webnn/buffer-manager';

/*
 * TensorProto::data_type from the ONNX specification.
 */
enum TensorProtoDataType {
  float = 1,
  uint8 = 2,
  int8 = 3,
  int32 = 6,
  int64 = 7,
  float16 = 10,
  uint32 = 12,
  uint64 = 13,
}

/*
 * TensorProto::data_type to WebNN OperandType mapping.
 */
const onnxDataTypeToWebnnDataType = new Map<TensorProtoDataType, MLOperandDataType>([
  [TensorProtoDataType.float, 'float32'],
  [TensorProtoDataType.float16, 'float16'],
  [TensorProtoDataType.int32, 'int32'],
  [TensorProtoDataType.uint32, 'uint32'],
  [TensorProtoDataType.int64, 'int64'],
  [TensorProtoDataType.uint64, 'uint64'],
  [TensorProtoDataType.int8, 'int8'],
  [TensorProtoDataType.uint8, 'uint8'],
]);

/**
 * WebNN backend implementation. This class is used to keep track of the MLBuffers created by the backend and keep track
 * of the current MLContext being used by the sessions.
 */
export class WebNNBackend {
  private bufferManager: BufferManager = createBufferManager(this);
  /**
   * Maps from session id to MLContexts.
   */
  private mlContextBySessionId = new Map<number, MLContext>();
  /**
   * Maps from MLContext to session ids.
   */
  private sessionIdsByMlContext = new Map<MLContext, Set<number>>();
  /**
   * Current session id.
   */
  currentSessionId?: number;

  public onRunStart(sessionId: number): void {
    this.currentSessionId = sessionId;
  }

  public get currentContext(): MLContext {
    if (this.currentSessionId === undefined) {
      throw new Error('No active session');
    }
    return this.getMlContext(this.currentSessionId);
  }

  public registerMlContext(sessionId: number, mlContext: MLContext): void {
    this.mlContextBySessionId.set(sessionId, mlContext);
    let sessionIds = this.sessionIdsByMlContext.get(mlContext);
    if (!sessionIds) {
      sessionIds = new Set();
      this.sessionIdsByMlContext.set(mlContext, sessionIds);
    }
    sessionIds.add(sessionId);
  }

  public unregisterMlContext(sessionId: number): void {
    this.mlContextBySessionId.delete(sessionId);
    const mlContext = this.mlContextBySessionId.get(sessionId)!;
    const sessionIds = this.sessionIdsByMlContext.get(mlContext)!;
    sessionIds.delete(sessionId);
    if (sessionIds.size === 0) {
      this.sessionIdsByMlContext.delete(mlContext);
    }
  }

  public onReleaseSession(sessionId: number): void {
    this.unregisterMlContext(sessionId);
    this.bufferManager.releaseBuffersForContext(this.getMlContext(sessionId));
  }

  public getMlContext(sessionId: number): MLContext {
    return this.mlContextBySessionId.get(sessionId)!;
  }

  public reserveBufferId(): BufferId {
    return this.bufferManager.reserveBufferId();
  }

  public releaseBufferId(bufferId: BufferId): void {
    this.bufferManager.releaseBufferId(bufferId);
  }

  public getBuffer(bufferId: BufferId): MLBuffer {
    return this.bufferManager.getBuffer(bufferId);
  }

  public ensureBuffer(bufferId: BufferId, onnxDataType: number|MLOperandDataType, dimensions: number[]): MLBuffer {
    let dataType: MLOperandDataType;
    if (typeof onnxDataType === 'number') {
      const webnnDataType = onnxDataTypeToWebnnDataType.get(onnxDataType)!;
      if (!webnnDataType) {
        throw new Error(`Unsupported ONNX data type: ${onnxDataType}`);
      }
      dataType = webnnDataType;
    } else {
      dataType = onnxDataType;
    }
    return this.bufferManager.ensureBuffer(bufferId, dataType, dimensions);
  }

  public uploadBuffer(bufferId: BufferId, data: Uint8Array): void {
    this.bufferManager.upload(bufferId, data);
  }

  public async downloadBuffer(bufferId: BufferId): Promise<ArrayBuffer> {
    return this.bufferManager.download(bufferId);
  }

  public createMlBufferDownloader(bufferId: BufferId, type: Tensor.GpuBufferDataTypes): () => Promise<Tensor.DataType> {
    return async () => {
      const data = await this.bufferManager.download(bufferId);
      return createView(data, type);
    };
  }

  public registerMlBuffer(buffer: MLBuffer): BufferId {
    return this.bufferManager.registerBuffer(this.currentContext, buffer);
  }

  public flush(): void {
    // Unlike the WebGPU backend, the WebNN backend does not need to flush any pending operations.
  }
}
