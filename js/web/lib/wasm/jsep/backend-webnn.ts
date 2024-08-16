// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// WebNN API currently does not have a TypeScript definition file. This file is a workaround with types generated from
// WebNN API specification.
// https://github.com/webmachinelearning/webnn/issues/677
/// <reference path="webnn/webnn.d.ts" />

import { Tensor } from 'onnxruntime-common';

import { DataType } from '../wasm-common';
import { getInstance } from '../wasm-factory';

import { createView } from './tensor-view';
import { BufferId, BufferManager, createBufferManager } from './webnn/buffer-manager';

/*
 * TensorProto::data_type to WebNN OperandType mapping.
 */
const onnxDataTypeToWebnnDataType = new Map<DataType, MLOperandDataType>([
  [DataType.float, 'float32'],
  [DataType.float16, 'float16'],
  [DataType.int32, 'int32'],
  [DataType.uint32, 'uint32'],
  [DataType.int64, 'int64'],
  [DataType.uint64, 'uint64'],
  [DataType.int8, 'int8'],
  [DataType.uint8, 'uint8'],
  [DataType.bool, 'uint8'],
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
  private sessionIdsByMLContext = new Map<MLContext, Set<number>>();
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
    return this.getMLContext(this.currentSessionId);
  }

  public registerMLContext(sessionId: number, mlContext: MLContext): void {
    this.mlContextBySessionId.set(sessionId, mlContext);
    let sessionIds = this.sessionIdsByMLContext.get(mlContext);
    if (!sessionIds) {
      sessionIds = new Set();
      this.sessionIdsByMLContext.set(mlContext, sessionIds);
    }
    sessionIds.add(sessionId);
  }

  public unregisterMLContext(sessionId: number): void {
    const mlContext = this.mlContextBySessionId.get(sessionId)!;
    if (!mlContext) {
      throw new Error(`No MLContext found for session ${sessionId}`);
    }
    this.mlContextBySessionId.delete(sessionId);
    const sessionIds = this.sessionIdsByMLContext.get(mlContext)!;
    sessionIds.delete(sessionId);
    if (sessionIds.size === 0) {
      this.sessionIdsByMLContext.delete(mlContext);
    }
  }

  public onReleaseSession(sessionId: number): void {
    this.unregisterMLContext(sessionId);
    this.bufferManager.releaseBuffersForContext(this.getMLContext(sessionId));
  }

  public getMLContext(sessionId: number): MLContext {
    return this.mlContextBySessionId.get(sessionId)!;
  }

  public reserveBufferId(): BufferId {
    return this.bufferManager.reserveBufferId();
  }

  public releaseBufferId(bufferId: BufferId): void {
    this.bufferManager.releaseBufferId(bufferId);
  }

  public async ensureBuffer(
    bufferId: BufferId,
    onnxDataType: number | MLOperandDataType,
    dimensions: number[],
  ): Promise<MLBuffer> {
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
    const wasm = getInstance();
    if (!wasm.shouldTransferToMLBuffer) {
      throw new Error('Trying to upload to a MLBuffer while shouldTransferToMLBuffer is false');
    }
    this.bufferManager.upload(bufferId, data);
  }

  public async downloadBuffer(bufferId: BufferId): Promise<ArrayBuffer> {
    return this.bufferManager.download(bufferId);
  }

  public createMLBufferDownloader(bufferId: BufferId, type: Tensor.MLBufferDataTypes): () => Promise<Tensor.DataType> {
    return async () => {
      const data = await this.bufferManager.download(bufferId);
      return createView(data, type);
    };
  }

  public registerMLBuffer(buffer: MLBuffer): BufferId {
    return this.bufferManager.registerBuffer(this.currentContext, buffer);
  }

  public flush(): void {
    // Unlike the WebGPU backend, the WebNN backend does not need to flush any pending operations.
  }
}
