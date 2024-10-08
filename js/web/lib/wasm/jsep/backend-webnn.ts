// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// WebNN API currently does not have a TypeScript definition file. This file is a workaround with types generated from
// WebNN API specification.
// https://github.com/webmachinelearning/webnn/issues/677
/// <reference path="webnn/webnn.d.ts" />

import { Env, Tensor } from 'onnxruntime-common';

import { DataType } from '../wasm-common';
import { getInstance } from '../wasm-factory';

import { createView } from './tensor-view';
import { TensorId, createTensorManager } from './webnn/tensor-manager';
import { configureLogger, LOG_DEBUG } from './log';

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
 * WebNN backend implementation. This class is used to keep track of the MLTensors created by the backend and keep track
 * of the current MLContext being used by the sessions.
 */
export class WebNNBackend {
  /**
   * Tensor managers for each session.
   */
  private tensorManager = createTensorManager(this);
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
  private activeSessionId?: number;

  constructor(env: Env) {
    configureLogger(env.logLevel!, !!env.debug);
  }

  public get currentSessionId(): number {
    if (this.activeSessionId === undefined) {
      throw new Error('No active session');
    }
    return this.activeSessionId;
  }

  public onRunStart(sessionId: number): void {
    this.activeSessionId = sessionId;
  }

  public get currentContext(): MLContext {
    const mlContext = this.getMLContext(this.currentSessionId);
    if (!mlContext) {
      throw new Error(`No MLContext found for session ${this.currentSessionId}`);
    }
    return mlContext;
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

  public onReleaseSession(sessionId: number): void {
    const mlContext = this.mlContextBySessionId.get(sessionId)!;
    if (!mlContext) {
      // Current session is not a WebNN session.
      return;
    }
    this.mlContextBySessionId.delete(sessionId);
    const sessionIds = this.sessionIdsByMLContext.get(mlContext)!;
    sessionIds.delete(sessionId);
    if (sessionIds.size === 0) {
      this.sessionIdsByMLContext.delete(mlContext);
      this.tensorManager.releaseTensorsForContext(mlContext);
    }
  }

  public getMLContext(sessionId: number): MLContext | undefined {
    return this.mlContextBySessionId.get(sessionId);
  }

  public reserveTensorId(): TensorId {
    return this.tensorManager.reserveTensorId();
  }

  public releaseTensorId(tensorId: TensorId): void {
    LOG_DEBUG('verbose', () => `[WebNN] releaseTensorId {tensorId: ${tensorId}}`);
    this.tensorManager.releaseTensorId(tensorId);
  }

  public async ensureTensor(
    tensorId: TensorId,
    onnxDataType: DataType,
    dimensions: number[],
    copyOld: boolean,
  ): Promise<MLTensor> {
    const webnnDataType = onnxDataTypeToWebnnDataType.get(onnxDataType);
    if (!webnnDataType) {
      throw new Error(`Unsupported ONNX data type: ${onnxDataType}`);
    }
    return this.tensorManager.ensureTensor(tensorId, webnnDataType, dimensions, copyOld);
  }

  public uploadTensor(tensorId: TensorId, data: Uint8Array): void {
    const wasm = getInstance();
    if (!wasm.shouldTransferToMLTensor) {
      throw new Error('Trying to upload to a MLTensor while shouldTransferToMLTensor is false');
    }
    LOG_DEBUG('verbose', () => `[WebNN] uploadTensor {tensorId: ${tensorId}, data: ${data.byteLength}}`);
    this.tensorManager.upload(tensorId, data);
  }

  public async downloadTensor(tensorId: TensorId, dstBuffer: ArrayBufferView | ArrayBuffer): Promise<undefined> {
    return this.tensorManager.download(tensorId, dstBuffer);
  }

  public createMLTensorDownloader(tensorId: TensorId, type: Tensor.MLTensorDataTypes): () => Promise<Tensor.DataType> {
    return async () => {
      const data = await this.tensorManager.download(tensorId);
      return createView(data, type);
    };
  }

  public registerMLTensor(tensor: MLTensor, onnxDataType: DataType, dimensions: number[]): TensorId {
    const webnnDataType = onnxDataTypeToWebnnDataType.get(onnxDataType);
    if (!webnnDataType) {
      throw new Error(`Unsupported ONNX data type: ${onnxDataType}`);
    }

    const id = this.tensorManager.registerTensor(this.currentContext, tensor, webnnDataType, dimensions);
    LOG_DEBUG(
      'verbose',
      () =>
        `[WebNN] registerMLTensor {tensor: ${tensor}, dataType: ${webnnDataType}, dimensions: ${
          dimensions
        }} -> {tensorId: ${id}}`,
    );
    return id;
  }

  public flush(): void {
    // Unlike the WebGPU backend, the WebNN backend does not need to flush any pending operations.
  }
}
