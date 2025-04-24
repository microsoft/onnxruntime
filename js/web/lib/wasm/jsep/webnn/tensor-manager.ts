// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { WebNNBackend } from '../backend-webnn';
import { tensorTypeToTypedArrayConstructor } from '../../wasm-common';
import { LOG_DEBUG } from '../log';

// WebNN API currently does not have a TypeScript definition file. This file is a workaround with types generated from
// WebNN API specification.
// https://github.com/webmachinelearning/webnn/issues/677
/// <reference path="webnn.d.ts" />

/**
 * Map from MLOperandDataType to size in bits. Using bits instead of bytes to avoid possible precision loss on int4 and uint4.
 */
const webnnDataTypeToSize = new Map<MLOperandDataType, number>([
  ['float32', 32],
  ['float16', 16],
  ['int32', 32],
  ['uint32', 32],
  ['int64', 64],
  ['uint64', 64],
  ['int8', 8],
  ['uint8', 8],
  ['int4', 4],
  ['uint4', 4],
]);

// Convert integer data to an Int32Array buffer.
// Supports conversion from int64, uint64, uint32, int8 and uint8 to int32.
export const convertDataToInt32 = (data: Uint8Array, dataType: MLOperandDataType): Uint8Array => {
  if (dataType === 'int32') {
    return data;
  }

  const dataTypeSize = webnnDataTypeToSize.get(dataType);
  if (!dataTypeSize) {
    throw new Error(`WebNN backend does not support data type: ${dataType}`);
  }
  const bytesPerElement = dataTypeSize / 8;
  // Make sure the data length is a multiple of the data type size.
  if (data.byteLength % bytesPerElement !== 0) {
    throw new Error(`Invalid Uint8Array length - must be a multiple of ${bytesPerElement}.`);
  }

  // Convert Uint8Array to original typed array.
  const numElements = data.byteLength / bytesPerElement;
  const originalArray = new (tensorTypeToTypedArrayConstructor(dataType))(data.buffer, data.byteOffset, numElements);

  switch (dataType) {
    case 'int64':
    case 'uint64': {
      // Convert original typed array to Int32Array.
      const int32Array = new Int32Array(numElements);
      for (let i = 0; i < numElements; i++) {
        const value = originalArray[i];

        // Check for overflow.
        if (value > 2147483647n || value < -2147483648n) {
          throw new Error(`Can not convert int64 data to int32 - value out of range.`);
        }

        int32Array[i] = Number(value);
      }

      return new Uint8Array(int32Array.buffer);
    }
    case 'int8':
    case 'uint8':
    case 'uint32': {
      // Check for overflow.
      if (dataType === 'uint32') {
        if (originalArray.some((value) => value > 2147483647)) {
          throw new Error(`Can not convert uint32 data to int32 - value out of range.`);
        }
      }
      // Convert original typed array to Int32Array.
      const int32Array = Int32Array.from(originalArray, Number);
      return new Uint8Array(int32Array.buffer);
    }
    default:
      throw new Error(`Unsupported data conversion from ${dataType} to 'int32'`);
  }
};

// Convert Int32Array data to original integer data buffer.
// Supports conversion from int32 to int64, uint64, uint32, int8 and uint8.
export const convertInt32ToData = (data: Uint8Array, dataType: MLOperandDataType): Uint8Array => {
  if (dataType === 'int32') {
    return data;
  }

  // Make sure the data length is a multiple of 4 bytes (Int32Array).
  if (data.byteLength % 4 !== 0) {
    throw new Error('Invalid Uint8Array length - must be a multiple of 4 (int32).');
  }

  // Convert Uint8Array to Int32Array.
  const numElements = data.byteLength / 4;
  const int32Array = new Int32Array(data.buffer, data.byteOffset, numElements);

  switch (dataType) {
    case 'int64': {
      const bigInt64Array = BigInt64Array.from(int32Array, BigInt);
      return new Uint8Array(bigInt64Array.buffer);
    }
    case 'uint64': {
      if (int32Array.some((value) => value < 0)) {
        throw new Error('Can not convert int32 data to uin64 - negative value found.');
      }
      const bigUint64Array = BigUint64Array.from(int32Array, BigInt);
      return new Uint8Array(bigUint64Array.buffer);
    }
    case 'int8': {
      if (int32Array.some((value) => value < -128 || value > 127)) {
        throw new Error('Can not convert int32 data to int8 - value out of range.');
      }
      const int8Array = Int8Array.from(int32Array, Number);
      return new Uint8Array(int8Array.buffer);
    }
    case 'uint8': {
      if (int32Array.some((value) => value < 0 || value > 255)) {
        throw new Error('Can not convert int32 data to uint8 - value out of range.');
      }
      return Uint8Array.from(int32Array, Number);
    }
    case 'uint32': {
      if (int32Array.some((value) => value < 0)) {
        throw new Error('Can not convert int32 data to uint32 - negative value found.');
      }
      const uint32Array = Uint32Array.from(int32Array, Number);
      return new Uint8Array(uint32Array.buffer);
    }
    default:
      throw new Error(`Unsupported data conversion from 'int32' to ${dataType}`);
  }
};

export type TensorId = number;

/**
 * Manages TensorId to MLTensor mapping.
 */
export interface TensorManager {
  /**
   * Reserve a new TensorId.
   */
  reserveTensorId(): TensorId;
  /**
   * Release a TensorId.
   */
  releaseTensorId(tensorId: TensorId): void;
  /**
   * Ensure a MLTensor is created for the TensorId.
   */
  ensureTensor(
    sessionId: number,
    tensorId: TensorId,
    dataType: MLOperandDataType,
    shape: readonly number[],
    copyOld: boolean,
  ): Promise<MLTensor>;
  /**
   * Upload data to a MLTensor.
   */
  upload(tensorId: TensorId, data: Uint8Array): void;
  /**
   * Download data from a MLTensor.
   */
  download(tensorId: TensorId): Promise<ArrayBuffer>;
  download(tensorId: TensorId, dstTensor: ArrayBufferView | ArrayBuffer): Promise<undefined>;
  /**
   * Release all tensors for a given session.
   */
  releaseTensorsForSession(session: number): void;
  /**
   * Register an externally created MLTensor with a given session id and return a TensorId.
   */
  registerTensor(sessionId: number, mlTensor: MLTensor, dataType: MLOperandDataType, shape: number[]): TensorId;
}

let tensorGuid = 1;
const createNewTensorId = (): TensorId => tensorGuid++;

/**
 * Map from data type to fallback data type.
 * When the context does not support the original data type, use fallback data type as workaround.
 * Note: Currently, we only support fallback to int32 for certain integer data types.
 */
const webnnDataTypeToFallback = new Map<MLOperandDataType, MLOperandDataType>([
  ['int8', 'int32'],
  ['uint8', 'int32'],
  ['uint32', 'int32'],
  ['int64', 'int32'],
]);

/**
 * Calculate the byte length of a tensor with the given data type and shape.
 */
const calculateByteLength = (dataType: MLOperandDataType, shape: readonly number[]): number => {
  const dataTypeSize = webnnDataTypeToSize.get(dataType);
  if (!dataTypeSize) {
    throw new Error(`WebNN backend does not support data type: ${dataType}`);
  }
  return shape.length > 0 ? Math.ceil((shape.reduce((a, b) => a * b) * dataTypeSize) / 8) : 0;
};

/**
 * TensorWrapper wraps an MLTensor and provides a way to track the last session that used it.
 */
class TensorWrapper {
  // The id of the last session that used this tensor.
  public sessionId: number;
  // This flag is used to indicate whether the data has been converted to fallback data type.
  public isDataConverted = false;

  private mlContext: MLContext;
  private mlTensor: MLTensor;
  private dataType: MLOperandDataType;
  // Fallback data type to use when the context does not support the original data type.
  private fallbackDataType: MLOperandDataType | undefined;
  private tensorShape: readonly number[];

  constructor(descriptor: {
    sessionId: number;
    context: MLContext;
    tensor: MLTensor;
    dataType: MLOperandDataType;
    shape: readonly number[];
    fallbackDataType?: MLOperandDataType;
  }) {
    const { sessionId, context, tensor, dataType, shape, fallbackDataType } = descriptor;
    this.sessionId = sessionId;
    this.mlContext = context;
    this.mlTensor = tensor;
    this.dataType = dataType;
    this.tensorShape = shape;
    this.fallbackDataType = fallbackDataType;
  }

  public get tensor(): MLTensor {
    return this.mlTensor;
  }

  public get type(): MLOperandDataType {
    return this.dataType;
  }

  public get fallbackType(): MLOperandDataType | undefined {
    return this.fallbackDataType;
  }

  public get shape(): readonly number[] {
    return this.tensorShape;
  }

  public get byteLength(): number {
    return calculateByteLength(this.dataType, this.tensorShape);
  }

  public destroy(): void {
    LOG_DEBUG('verbose', () => '[WebNN] TensorWrapper.destroy');
    this.mlTensor.destroy();
  }

  public write(data: Uint8Array): void {
    this.mlContext.writeTensor(this.mlTensor, data);
  }

  public async read(): Promise<ArrayBuffer>;
  public async read(dstBuffer?: ArrayBufferView | ArrayBuffer): Promise<ArrayBuffer | undefined>;
  public async read(dstBuffer?: ArrayBufferView | ArrayBuffer): Promise<ArrayBuffer | undefined> {
    if (this.fallbackDataType) {
      // This tensor has been fallback to int32 as workaround, we need to read it as its original integer data type.
      const data = await this.mlContext.readTensor(this.mlTensor);
      const originalData = convertInt32ToData(new Uint8Array(data), this.dataType);

      if (dstBuffer) {
        const targetBuffer =
          dstBuffer instanceof ArrayBuffer
            ? new Uint8Array(dstBuffer)
            : new Uint8Array(dstBuffer.buffer, dstBuffer.byteOffset, dstBuffer.byteLength);
        targetBuffer.set(originalData);
        return undefined;
      } else {
        return originalData.buffer;
      }
    } else {
      return dstBuffer ? this.mlContext.readTensor(this.mlTensor, dstBuffer) : this.mlContext.readTensor(this.mlTensor);
    }
  }

  public canReuseTensor(context: MLContext, dataType: MLOperandDataType, shape: readonly number[]): boolean {
    return (
      this.mlContext === context &&
      this.dataType === dataType &&
      this.tensorShape.length === shape.length &&
      this.tensorShape.every((v, i) => v === shape[i])
    );
  }

  public setIsDataConverted(isConverted: boolean): void {
    this.isDataConverted = isConverted;
  }
}

/**
 * TensorTracker tracks the MLTensor and pending upload data.
 *
 * We need to track the MLTensor and pending upload data because we delay the creation of MLTensor until
 * we know the data type and shape. This is because WebNN only support creating MLTensors with dataTypes and shape.
 */
class TensorIdTracker {
  private activeUpload?: Uint8Array;

  constructor(
    private tensorManager: TensorManagerImpl,
    private wrapper?: TensorWrapper,
  ) {}

  public get tensorWrapper(): TensorWrapper | undefined {
    return this.wrapper;
  }

  public releaseTensor(): void {
    if (this.tensorWrapper) {
      this.tensorManager.releaseTensor(this.tensorWrapper);
      this.wrapper = undefined;
    }
  }

  public async ensureTensor(
    sessionId: number,
    dataType: MLOperandDataType,
    shape: readonly number[],
    copyOld: boolean,
  ): Promise<MLTensor> {
    const context = this.tensorManager.getMLContext(sessionId);
    let fallbackDataType: MLOperandDataType | undefined;
    // Check if the context supports the data type. If not, try to use the fallback data type.
    if (!context.opSupportLimits().input.dataTypes.includes(dataType)) {
      fallbackDataType = webnnDataTypeToFallback.get(dataType);
      if (!fallbackDataType || !context.opSupportLimits().input.dataTypes.includes(fallbackDataType)) {
        throw new Error(`WebNN backend does not support data type: ${dataType}`);
      }
      LOG_DEBUG(
        'verbose',
        () => `[WebNN] TensorIdTracker.ensureTensor: fallback dataType from ${dataType} to ${fallbackDataType}`,
      );
    }

    if (this.wrapper) {
      if (this.wrapper.canReuseTensor(context, dataType, shape)) {
        return this.wrapper.tensor;
      } else {
        if (copyOld) {
          if (this.wrapper.byteLength !== calculateByteLength(dataType, shape)) {
            throw new Error('Unable to copy data to tensor with different size.');
          }
          this.activeUpload = new Uint8Array(await this.wrapper.read());
        }
        this.tensorManager.releaseTensor(this.wrapper);
      }
    }

    // eslint-disable-next-line no-bitwise
    const usage = typeof MLTensorUsage == 'undefined' ? undefined : MLTensorUsage.READ | MLTensorUsage.WRITE;
    this.wrapper = await this.tensorManager.getCachedTensor(
      sessionId,
      dataType,
      shape,
      usage,
      true,
      true,
      fallbackDataType,
    );

    if (copyOld && this.activeUpload) {
      // We don't need to convert the original integer data to int32,
      // because it has been converted when it was uploaded.
      this.wrapper.write(this.activeUpload);
      this.activeUpload = undefined;
    }

    return this.wrapper.tensor;
  }

  public upload(data: Uint8Array): void {
    let newData = data;
    if (this.wrapper) {
      if (this.wrapper.fallbackType) {
        if (this.wrapper.fallbackType === 'int32') {
          // Convert original integer data to int32.
          newData = convertDataToInt32(data, this.wrapper.type);
          this.wrapper.setIsDataConverted(true);
        } else {
          throw new Error(`Unsupported fallback data type: ${this.wrapper.fallbackType}`);
        }
      }

      // Check if the data size matches the tensor size.
      if (data.byteLength === this.wrapper.byteLength) {
        // Write the newData to the tensor.
        this.wrapper.write(newData);
        return;
      } else {
        LOG_DEBUG('verbose', () => 'Data size does not match tensor size. Releasing tensor.');
        this.releaseTensor();
      }
    }

    if (this.activeUpload) {
      this.activeUpload.set(newData);
    } else {
      this.activeUpload = new Uint8Array(newData);
    }
  }

  public async download(dstBuffer?: ArrayBufferView | ArrayBuffer): Promise<ArrayBuffer | undefined> {
    if (this.activeUpload) {
      // If this.activeUpload has been converted to int32, we need to convert it back to original integer data type.
      const dstData = this.wrapper?.isDataConverted
        ? convertInt32ToData(this.activeUpload, this.wrapper?.type)
        : this.activeUpload;

      if (dstBuffer) {
        if (dstBuffer instanceof ArrayBuffer) {
          new Uint8Array(dstBuffer).set(dstData);
        } else {
          new Uint8Array(dstBuffer.buffer, dstBuffer.byteOffset, dstBuffer.byteLength).set(dstData);
        }
        return;
      } else {
        return dstData.buffer;
      }
    }
    if (!this.wrapper) {
      throw new Error('Tensor has not been created.');
    }

    if (!dstBuffer) {
      return this.wrapper.read();
    }
    return this.wrapper.read(dstBuffer);
  }
}

class TensorManagerImpl implements TensorManager {
  private tensorTrackersById: Map<TensorId, TensorIdTracker> = new Map();
  private freeTensors: TensorWrapper[] = [];
  private externalTensors: Set<TensorWrapper> = new Set();

  constructor(private backend: WebNNBackend) {}

  public getMLContext(sessionId: number): MLContext {
    const context = this.backend.getMLContext(sessionId);
    if (!context) {
      throw new Error('MLContext not found for session.');
    }
    return context;
  }

  public reserveTensorId(): TensorId {
    const tensorId = createNewTensorId();
    this.tensorTrackersById.set(tensorId, new TensorIdTracker(this));
    return tensorId;
  }

  public releaseTensorId(tensorId: TensorId): void {
    const tensorTracker = this.tensorTrackersById.get(tensorId);
    if (!tensorTracker) {
      return;
    }
    this.tensorTrackersById.delete(tensorId);
    if (tensorTracker.tensorWrapper) {
      this.releaseTensor(tensorTracker.tensorWrapper);
    }
  }

  public async ensureTensor(
    sessionId: number,
    tensorId: TensorId,
    dataType: MLOperandDataType,
    shape: number[],
    copyOld: boolean,
  ): Promise<MLTensor> {
    LOG_DEBUG(
      'verbose',
      () =>
        `[WebNN] TensorManager.ensureTensor {tensorId: ${tensorId}, dataType: ${
          dataType
        }, shape: ${shape}, copyOld: ${copyOld}}`,
    );
    const tensor = this.tensorTrackersById.get(tensorId);
    if (!tensor) {
      throw new Error('Tensor not found.');
    }
    return tensor.ensureTensor(sessionId, dataType, shape, copyOld);
  }

  public upload(tensorId: TensorId, data: Uint8Array): void {
    const tensor = this.tensorTrackersById.get(tensorId);
    if (!tensor) {
      throw new Error('Tensor not found.');
    }
    tensor.upload(data);
  }

  public async download(tensorId: TensorId): Promise<ArrayBuffer>;
  public async download(tensorId: TensorId, dstBuffer: ArrayBufferView | ArrayBuffer): Promise<undefined>;
  async download(tensorId: TensorId, dstBuffer?: ArrayBufferView | ArrayBuffer): Promise<ArrayBuffer | undefined> {
    LOG_DEBUG(
      'verbose',
      () => `[WebNN] TensorManager.download {tensorId: ${tensorId}, dstBuffer: ${dstBuffer?.byteLength}}`,
    );
    const tensorTracker = this.tensorTrackersById.get(tensorId);
    if (!tensorTracker) {
      throw new Error('Tensor not found.');
    }
    return tensorTracker.download(dstBuffer);
  }

  public releaseTensorsForSession(sessionId: number): void {
    for (const tensor of this.freeTensors) {
      if (tensor.sessionId === sessionId) {
        tensor.destroy();
      }
    }
    this.freeTensors = this.freeTensors.filter((tensor) => tensor.sessionId !== sessionId);
  }

  public registerTensor(
    sessionId: number,
    mlTensor: MLTensor,
    dataType: MLOperandDataType,
    shape: readonly number[],
  ): TensorId {
    const context = this.getMLContext(sessionId);
    const tensorId = createNewTensorId();
    // Defaulting to READ | WRITE if usage is not provided.
    // eslint-disable-next-line no-bitwise
    const wrapper = new TensorWrapper({
      sessionId,
      context,
      tensor: mlTensor,
      dataType,
      shape,
    });
    this.tensorTrackersById.set(tensorId, new TensorIdTracker(this, wrapper));
    this.externalTensors.add(wrapper);
    return tensorId;
  }

  /**
   * Get or create an MLTensor with the given data type and shape.
   */
  public async getCachedTensor(
    sessionId: number,
    dataType: MLOperandDataType,
    shape: readonly number[],
    usage: MLTensorUsageFlags | undefined,
    writable: boolean,
    readable: boolean,
    fallbackDataType?: MLOperandDataType,
  ): Promise<TensorWrapper> {
    const context = this.getMLContext(sessionId);
    for (const [index, tensor] of this.freeTensors.entries()) {
      if (tensor.canReuseTensor(context, dataType, shape)) {
        LOG_DEBUG(
          'verbose',
          () =>
            `[WebNN] Reusing tensor {dataType: ${dataType}, ${
              fallbackDataType ? `fallbackDataType: ${fallbackDataType},` : ''
            } shape: ${shape}`,
        );
        const wrapper = this.freeTensors.splice(index, 1)[0];
        wrapper.sessionId = sessionId;
        return wrapper;
      }
    }
    LOG_DEBUG(
      'verbose',
      () =>
        `[WebNN] MLContext.createTensor {dataType: ${dataType}, ${
          fallbackDataType ? `fallbackDataType: ${fallbackDataType},` : ''
        } shape: ${shape}}`,
    );
    const tensor = await context.createTensor({
      dataType: fallbackDataType ?? dataType, // If fallback data type is provided, use it.
      shape,
      dimensions: shape,
      usage,
      writable,
      readable,
    });
    return new TensorWrapper({ sessionId, context, tensor, dataType, shape, fallbackDataType });
  }

  /**
   * Release tensor for reuse unless external.
   */
  public releaseTensor(tensorWrapper: TensorWrapper) {
    if (this.externalTensors.has(tensorWrapper)) {
      this.externalTensors.delete(tensorWrapper);
    }
    this.freeTensors.push(tensorWrapper);
  }
}

export const createTensorManager = (...args: ConstructorParameters<typeof TensorManagerImpl>): TensorManager =>
  new TensorManagerImpl(...args);
