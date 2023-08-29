// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {tensorToDataURL, tensorToImageData} from './tensor-conversion-impl.js';
import {TensorToDataUrlOptions, TensorToImageDataOptions} from './tensor-conversion.js';
import {tensorFromGpuBuffer, tensorFromImage, tensorFromPinnedBuffer, tensorFromTexture} from './tensor-factory-impl.js';
import {CpuPinnedConstructorParameters, CpuPinnedDataTypes, GpuBufferConstructorParameters, GpuBufferDataTypes, TensorFromGpuBufferOptions, TensorFromImageBitmapOptions, TensorFromImageDataOptions, TensorFromImageElementOptions, TensorFromTextureOptions, TensorFromUrlOptions, TextureConstructorParameters} from './tensor-factory.js';
import {checkBigInt, NUMERIC_TENSOR_TYPE_TO_TYPEDARRAY_MAP, NUMERIC_TENSOR_TYPEDARRAY_TO_TYPE_MAP, SupportedTypedArray, SupportedTypedArrayConstructors} from './tensor-impl-type-mapping.js';
import {calculateSize, tensorReshape} from './tensor-utils-impl.js';
import {Tensor as TensorInterface} from './tensor.js';

// type aliases for those exported from Tensor interface

type TensorType = TensorInterface.Type;
type TensorDataType = TensorInterface.DataType;
type TensorDataLocation = TensorInterface.DataLocation;
type TensorTextureType = TensorInterface.TextureType;
type TensorGpuBufferType = TensorInterface.GpuBufferType;

/**
 * the implementation of Tensor interface.
 *
 * @internal
 */
export class Tensor implements TensorInterface {
  // #region constructors

  /**
   * Construct a new CPU tensor object from the given type, data and dims.
   */
  constructor(
      type: TensorType, data: TensorDataType|readonly string[]|readonly number[]|readonly boolean[],
      dims?: readonly number[]);
  /**
   * Construct a new CPU tensor object from the given data and dims. Type is inferred from data.
   */
  constructor(data: TensorDataType|readonly string[]|readonly boolean[], dims?: readonly number[]);
  /**
   * Construct a new tensor object from the pinned CPU data with the given type and dims.
   *
   * Tensor's location will be set to 'cpu-pinned'.
   *
   * @param params - Specify the parameters to construct the tensor.
   */
  constructor(params: CpuPinnedConstructorParameters);
  /**
   * Construct a new tensor object from the WebGL texture with the given type and dims.
   *
   * Tensor's location will be set to 'texture'.
   *
   * @param params - Specify the parameters to construct the tensor.
   */
  constructor(params: TextureConstructorParameters);
  /**
   * Construct a new tensor object from the WebGPU buffer with the given type and dims.
   *
   * Tensor's location will be set to 'gpu-buffer'.
   *
   * @param params - Specify the parameters to construct the tensor.
   */
  constructor(params: GpuBufferConstructorParameters);

  /**
   * implementation.
   */
  constructor(
      arg0: TensorType|TensorDataType|readonly string[]|readonly boolean[]|CpuPinnedConstructorParameters|
      TextureConstructorParameters|GpuBufferConstructorParameters,
      arg1?: TensorDataType|readonly number[]|readonly string[]|readonly boolean[], arg2?: readonly number[]) {
    // perform one-time check for BigInt support
    checkBigInt();

    let type: TensorType;
    let dims: readonly number[];

    if (typeof arg0 === 'object' && 'location' in arg0) {
      //
      // constructing tensor from specific location
      //
      this.dataLocation = arg0.location;
      type = arg0.type;
      dims = arg0.dims;
      switch (arg0.location) {
        case 'cpu-pinned': {
          const expectedTypedArrayConstructor = NUMERIC_TENSOR_TYPE_TO_TYPEDARRAY_MAP.get(type);
          if (!expectedTypedArrayConstructor) {
            throw new TypeError(`unsupported type "${type}" to create tensor from pinned buffer`);
          }
          if (!(arg0.data instanceof expectedTypedArrayConstructor)) {
            throw new TypeError(`buffer should be of type ${expectedTypedArrayConstructor.name}`);
          }
          this.cpuData = arg0.data;
          break;
        }
        case 'texture': {
          if (type !== 'float32') {
            throw new TypeError(`unsupported type "${type}" to create tensor from texture`);
          }
          this.gpuTextureData = arg0.texture;
          this.downloader = arg0.download;
          this.disposer = arg0.dispose;
          break;
        }
        case 'gpu-buffer': {
          if (type !== 'float32' && type !== 'int32') {
            throw new TypeError(`unsupported type "${type}" to create tensor from gpu buffer`);
          }
          this.gpuBufferData = arg0.gpuBuffer;
          this.downloader = arg0.download;
          this.disposer = arg0.dispose;
          break;
        }
        default:
          throw new Error(`Tensor constructor: unsupported location '${this.dataLocation}'`);
      }
    } else {
      //
      // constructing tensor of location 'cpu'
      //
      let data: TensorDataType;
      let maybeDims: typeof arg1|typeof arg2;
      // check whether arg0 is type or data
      if (typeof arg0 === 'string') {
        //
        // Override: constructor(type, data, ...)
        //
        type = arg0;
        maybeDims = arg2;
        if (arg0 === 'string') {
          // string tensor
          if (!Array.isArray(arg1)) {
            throw new TypeError('A string tensor\'s data must be a string array.');
          }
          // we don't check whether every element in the array is string; this is too slow. we assume it's correct and
          // error will be populated at inference
          data = arg1;
        } else {
          // numeric tensor
          const typedArrayConstructor = NUMERIC_TENSOR_TYPE_TO_TYPEDARRAY_MAP.get(arg0);
          if (typedArrayConstructor === undefined) {
            throw new TypeError(`Unsupported tensor type: ${arg0}.`);
          }
          if (Array.isArray(arg1)) {
            if (arg0 === 'float16') {
              // Throw error here because when user try to use number array as data,
              // e.g. new Tensor('float16', [1, 2, 3, 4], dims)), it will actually call
              // Uint16Array.from(arg1) which generates wrong data.
              throw new TypeError(
                  'Creating a float16 tensor from number array is not supported. Please use Uint16Array as data.');
            } else if (arg0 === 'uint64' || arg0 === 'int64') {
              // use 'as any' here because:
              // 1. TypeScript's check on type of 'Array.isArray()' does not work with readonly arrays.
              // see https://github.com/microsoft/TypeScript/issues/17002
              // 2. TypeScript's check on union type of '(BigInt64ArrayConstructor|BigUint64ArrayConstructor).from()'
              // does not accept parameter mapFn.
              // 3. parameters of 'SupportedTypedArrayConstructors.from()' does not match the requirement of the union
              // type.

              // assume 'arg1' is of type "readonly number[]|readonly bigint[]" here.

              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              data = (typedArrayConstructor as any).from(arg1, BigInt);
            } else {
              // assume 'arg1' is of type "readonly number[]" here.
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              data = (typedArrayConstructor as any).from(arg1);
            }
          } else if (arg1 instanceof typedArrayConstructor) {
            data = arg1;
          } else {
            throw new TypeError(`A ${type} tensor's data must be type of ${typedArrayConstructor}`);
          }
        }
      } else {
        //
        // Override: constructor(data, ...)
        //
        maybeDims = arg1;
        if (Array.isArray(arg0)) {
          // only boolean[] and string[] is supported
          if (arg0.length === 0) {
            throw new TypeError('Tensor type cannot be inferred from an empty array.');
          }
          const firstElementType = typeof arg0[0];
          if (firstElementType === 'string') {
            type = 'string';
            data = arg0;
          } else if (firstElementType === 'boolean') {
            type = 'bool';
            // 'arg0' is of type 'boolean[]'. Uint8Array.from(boolean[]) actually works, but typescript thinks this is
            // wrong type. We use 'as any' to make it happy.
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            data = Uint8Array.from(arg0 as any[]);
          } else {
            throw new TypeError(`Invalid element type of data array: ${firstElementType}.`);
          }
        } else {
          // get tensor type from TypedArray
          const mappedType =
              NUMERIC_TENSOR_TYPEDARRAY_TO_TYPE_MAP.get(arg0.constructor as SupportedTypedArrayConstructors);
          if (mappedType === undefined) {
            throw new TypeError(`Unsupported type for tensor data: ${arg0.constructor}.`);
          }
          type = mappedType;
          data = arg0 as SupportedTypedArray;
        }
      }

      // type and data is processed, now processing dims
      if (maybeDims === undefined) {
        // assume 1-D tensor if dims omitted
        maybeDims = [data.length];
      } else if (!Array.isArray(maybeDims)) {
        throw new TypeError('A tensor\'s dims must be a number array');
      }
      dims = maybeDims as readonly number[];

      this.cpuData = data;
      this.dataLocation = 'cpu';
    }

    // perform check on dims
    const size = calculateSize(dims);
    // if data is on CPU, check whether data length matches tensor size
    if (this.cpuData && size !== this.cpuData.length) {
      throw new Error(`Tensor's size(${size}) does not match data length(${this.cpuData.length}).`);
    }

    this.type = type;
    this.dims = dims;
    this.size = size;
  }
  // #endregion

  // #region factory
  static async fromImage(
      image: ImageData|HTMLImageElement|ImageBitmap|string,
      options?: TensorFromImageDataOptions|TensorFromImageElementOptions|TensorFromImageBitmapOptions|
      TensorFromUrlOptions): Promise<TensorInterface> {
    return tensorFromImage(image, options);
  }

  static fromTexture(texture: TensorTextureType, options: TensorFromTextureOptions<'float32'>): TensorInterface {
    return tensorFromTexture(texture, options);
  }

  static fromGpuBuffer<T extends GpuBufferDataTypes>(
      gpuBuffer: TensorGpuBufferType, options: TensorFromGpuBufferOptions<T>): TensorInterface {
    return tensorFromGpuBuffer(gpuBuffer, options);
  }

  static fromPinnedBuffer<T extends CpuPinnedDataTypes>(
      type: T, buffer: TensorInterface.DataTypeMap[T], dims?: readonly number[]): Tensor {
    return tensorFromPinnedBuffer(type, buffer, dims);
  }

  // #endregion

  // #region conversions
  toDataURL(options?: TensorToDataUrlOptions): string {
    return tensorToDataURL(this, options);
  }

  toImageData(options?: TensorToImageDataOptions): ImageData {
    return tensorToImageData(this, options);
  }
  // #endregion

  // #region public fields
  readonly dims: readonly number[];
  readonly type: TensorType;
  readonly size: number;
  // #endregion

  // #region private fields

  /**
   * stores the location of the data.
   */
  private dataLocation: TensorDataLocation;

  /**
   * stores the data on CPU, if location is 'cpu' or 'cpu-pinned'. otherwise empty.
   */
  private cpuData?: TensorDataType;

  /**
   * stores the underlying texture when location is 'texture'. otherwise empty.
   */
  private gpuTextureData?: TensorTextureType;

  /**
   * stores the underlying GPU buffer when location is 'gpu-buffer'. otherwise empty.
   */
  private gpuBufferData?: TensorGpuBufferType;

  /**
   * stores an optional downloader function to download data from GPU to CPU.
   */
  private downloader?(): Promise<TensorDataType>;

  /**
   * a flag indicating whether the data is being downloaded from GPU to CPU.
   */
  private isDownloading?: boolean;

  /**
   * stores an optional disposer function to dispose the underlying data.
   */
  private disposer?(): void;
  // #endregion

  // #region properties
  get data(): TensorDataType {
    this.ensureValid();
    if (!this.cpuData) {
      throw new Error(
          'The data is not on CPU. Use `getData()` to download GPU data to CPU, ' +
          'or use `texture` property to access the GPU data directly.');
    }
    return this.cpuData;
  }

  get location(): TensorDataLocation {
    return this.dataLocation;
  }

  get texture(): TensorTextureType {
    this.ensureValid();
    if (!this.gpuTextureData) {
      throw new Error('The data is not stored as a WebGL texture.');
    }
    return this.gpuTextureData;
  }

  get gpuBuffer(): TensorGpuBufferType {
    this.ensureValid();
    if (!this.gpuBufferData) {
      throw new Error('The data is not stored as a WebGPU buffer.');
    }
    return this.gpuBufferData;
  }
  // #endregion

  // #region methods

  async getData(releaseData?: boolean): Promise<TensorDataType> {
    this.ensureValid();
    switch (this.dataLocation) {
      case 'cpu':
      case 'cpu-pinned':
        return this.data;
      case 'texture':
      case 'gpu-buffer': {
        if (!this.downloader) {
          throw new Error('The current tensor is not created with a specified data downloader.');
        }
        if (this.isDownloading) {
          throw new Error('The current tensor is being downloaded.');
        }
        try {
          this.isDownloading = true;
          const data = await this.downloader();
          this.downloader = undefined;
          this.dataLocation = 'cpu';
          this.cpuData = data;

          if (releaseData && this.disposer) {
            this.disposer();
            this.disposer = undefined;
          }

          return data;

        } finally {
          this.isDownloading = false;
        }
      }
      default:
        throw new Error(`cannot get data from location: ${this.dataLocation}`);
    }
  }

  dispose(): void {
    if (this.isDownloading) {
      throw new Error('The current tensor is being downloaded.');
    }

    if (this.disposer) {
      this.disposer();
      this.disposer = undefined;
    }
    this.cpuData = undefined;
    this.gpuTextureData = undefined;
    this.gpuBufferData = undefined;
    this.downloader = undefined;
    this.isDownloading = undefined;

    this.dataLocation = 'none';
  }

  // #endregion

  // #region tensor utilities
  private ensureValid(): void {
    if (this.dataLocation === 'none') {
      throw new Error('The tensor is disposed.');
    }
  }

  reshape(dims: readonly number[]): TensorInterface {
    this.ensureValid();
    if (this.downloader || this.disposer) {
      throw new Error('Cannot reshape a tensor that owns GPU resource.');
    }
    return tensorReshape(this, dims);
  }
  // #endregion
}
