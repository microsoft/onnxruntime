// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tensor as TensorInterface, TensorFromImageOptions, TensorToImageDataOptions} from './tensor';

type TensorType = TensorInterface.Type;
type TensorDataType = TensorInterface.DataType;

type SupportedTypedArrayConstructors = Float32ArrayConstructor|Uint8ArrayConstructor|Int8ArrayConstructor|
    Uint16ArrayConstructor|Int16ArrayConstructor|Int32ArrayConstructor|BigInt64ArrayConstructor|Uint8ArrayConstructor|
    Float64ArrayConstructor|Uint32ArrayConstructor|BigUint64ArrayConstructor;
type SupportedTypedArray = InstanceType<SupportedTypedArrayConstructors>;

// a runtime map that maps type string to TypedArray constructor. Should match Tensor.DataTypeMap.
const NUMERIC_TENSOR_TYPE_TO_TYPEDARRAY_MAP = new Map<string, SupportedTypedArrayConstructors>([
  ['float32', Float32Array],
  ['uint8', Uint8Array],
  ['int8', Int8Array],
  ['uint16', Uint16Array],
  ['int16', Int16Array],
  ['int32', Int32Array],
  ['bool', Uint8Array],
  ['float64', Float64Array],
  ['uint32', Uint32Array],
]);

// a runtime map that maps type string to TypedArray constructor. Should match Tensor.DataTypeMap.
const NUMERIC_TENSOR_TYPEDARRAY_TO_TYPE_MAP = new Map<SupportedTypedArrayConstructors, TensorType>([
  [Float32Array, 'float32'],
  [Uint8Array, 'uint8'],
  [Int8Array, 'int8'],
  [Uint16Array, 'uint16'],
  [Int16Array, 'int16'],
  [Int32Array, 'int32'],
  [Float64Array, 'float64'],
  [Uint32Array, 'uint32'],
]);

// the following code allows delaying execution of BigInt checking. This allows lazy initialization for
// NUMERIC_TENSOR_TYPE_TO_TYPEDARRAY_MAP and NUMERIC_TENSOR_TYPEDARRAY_TO_TYPE_MAP, which allows BigInt polyfill
// if available.
let isBigIntChecked = false;
const checkBigInt = () => {
  if (!isBigIntChecked) {
    isBigIntChecked = true;
    const isBigInt64ArrayAvailable = typeof BigInt64Array !== 'undefined' && typeof BigInt64Array.from === 'function';
    const isBigUint64ArrayAvailable =
        typeof BigUint64Array !== 'undefined' && typeof BigUint64Array.from === 'function';

    if (isBigInt64ArrayAvailable) {
      NUMERIC_TENSOR_TYPE_TO_TYPEDARRAY_MAP.set('int64', BigInt64Array);
      NUMERIC_TENSOR_TYPEDARRAY_TO_TYPE_MAP.set(BigInt64Array, 'int64');
    }
    if (isBigUint64ArrayAvailable) {
      NUMERIC_TENSOR_TYPE_TO_TYPEDARRAY_MAP.set('uint64', BigUint64Array);
      NUMERIC_TENSOR_TYPEDARRAY_TO_TYPE_MAP.set(BigUint64Array, 'uint64');
    }
  }
};

/**
 * calculate size from dims.
 *
 * @param dims the dims array. May be an illegal input.
 */
const calculateSize = (dims: readonly unknown[]): number => {
  let size = 1;
  for (let i = 0; i < dims.length; i++) {
    const dim = dims[i];
    if (typeof dim !== 'number' || !Number.isSafeInteger(dim)) {
      throw new TypeError(`dims[${i}] must be an integer, got: ${dim}`);
    }
    if (dim < 0) {
      throw new RangeError(`dims[${i}] must be a non-negative integer, got: ${dim}`);
    }
    size *= dim;
  }
  return size;
};

export class Tensor implements TensorInterface {
  // #region constructors
  constructor(type: TensorType, data: TensorDataType|readonly number[]|readonly boolean[], dims?: readonly number[]);
  constructor(data: TensorDataType|readonly boolean[], dims?: readonly number[]);
  constructor(
      arg0: TensorType|TensorDataType|readonly boolean[], arg1?: TensorDataType|readonly number[]|readonly boolean[],
      arg2?: readonly number[]) {
    checkBigInt();

    let type: TensorType;
    let data: TensorDataType;
    let dims: typeof arg1|typeof arg2;
    // check whether arg0 is type or data
    if (typeof arg0 === 'string') {
      //
      // Override: constructor(type, data, ...)
      //
      type = arg0;
      dims = arg2;
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
          // use 'as any' here because TypeScript's check on type of 'SupportedTypedArrayConstructors.from()' produces
          // incorrect results.
          // 'typedArrayConstructor' should be one of the typed array prototype objects.
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          data = (typedArrayConstructor as any).from(arg1);
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
      dims = arg1;
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
    if (dims === undefined) {
      // assume 1-D tensor if dims omitted
      dims = [data.length];
    } else if (!Array.isArray(dims)) {
      throw new TypeError('A tensor\'s dims must be a number array');
    }

    // perform check
    const size = calculateSize(dims);
    if (size !== data.length) {
      throw new Error(`Tensor's size(${size}) does not match data length(${data.length}).`);
    }

    this.dims = dims as readonly number[];
    this.type = type;
    this.data = data;
    this.size = size;
  }
  // #endregion
  /**
   * Create a new tensor object from image object
   *
   * @param buffer - Extracted image buffer data - assuming RGBA format
   * @param imageFormat - input image configuration - required configurations height, width, format
   * @param tensorFormat - output tensor configuration - Default is RGB format
   */
  private static bufferToTensor(buffer: Uint8ClampedArray|undefined, options: TensorFromImageOptions): Tensor {
    if (buffer === undefined) {
      throw new Error('Image buffer must be defined');
    }
    if (options.height === undefined || options.width === undefined) {
      throw new Error('Image height and width must be defined');
    }
    if (options.tensorLayout === 'NHWC') {
      throw new Error('NHWC Tensor layout is not supported yet');
    }

    const {height, width} = options;

    const norm = options.norm ?? {mean: 255, bias: 0};
    let normMean: [number, number, number, number];
    let normBias: [number, number, number, number];

    if (typeof (norm.mean) === 'number') {
      normMean = [norm.mean, norm.mean, norm.mean, norm.mean];
    } else {
      normMean = [norm.mean![0], norm.mean![1], norm.mean![2], norm.mean![3] ?? 255];
    }

    if (typeof (norm.bias) === 'number') {
      normBias = [norm.bias, norm.bias, norm.bias, norm.bias];
    } else {
      normBias = [norm.bias![0], norm.bias![1], norm.bias![2], norm.bias![3] ?? 0];
    }

    const inputformat = options.bitmapFormat !== undefined ? options.bitmapFormat : 'RGBA';
    // default value is RGBA since imagedata and HTMLImageElement uses it

    const outputformat = options.tensorFormat !== undefined ?
        (options.tensorFormat !== undefined ? options.tensorFormat : 'RGB') :
        'RGB';
    const stride = height * width;
    const float32Data = outputformat === 'RGBA' ? new Float32Array(stride * 4) : new Float32Array(stride * 3);

    // Default pointer assignments
    let step = 4, rImagePointer = 0, gImagePointer = 1, bImagePointer = 2, aImagePointer = 3;
    let rTensorPointer = 0, gTensorPointer = stride, bTensorPointer = stride * 2, aTensorPointer = -1;

    // Updating the pointer assignments based on the input image format
    if (inputformat === 'RGB') {
      step = 3;
      rImagePointer = 0;
      gImagePointer = 1;
      bImagePointer = 2;
      aImagePointer = -1;
    }

    // Updating the pointer assignments based on the output tensor format
    if (outputformat === 'RGBA') {
      aTensorPointer = stride * 3;
    } else if (outputformat === 'RBG') {
      rTensorPointer = 0;
      bTensorPointer = stride;
      gTensorPointer = stride * 2;
    } else if (outputformat === 'BGR') {
      bTensorPointer = 0;
      gTensorPointer = stride;
      rTensorPointer = stride * 2;
    }

    for (let i = 0; i < stride;
         i++, rImagePointer += step, bImagePointer += step, gImagePointer += step, aImagePointer += step) {
      float32Data[rTensorPointer++] = (buffer[rImagePointer] + normBias[0]) / normMean[0];
      float32Data[gTensorPointer++] = (buffer[gImagePointer] + normBias[1]) / normMean[1];
      float32Data[bTensorPointer++] = (buffer[bImagePointer] + normBias[2]) / normMean[2];
      if (aTensorPointer !== -1 && aImagePointer !== -1) {
        float32Data[aTensorPointer++] = (buffer[aImagePointer] + normBias[3]) / normMean[3];
      }
    }

    // Float32Array -> ort.Tensor
    const outputTensor = outputformat === 'RGBA' ? new Tensor('float32', float32Data, [1, 4, height, width]) :
                                                   new Tensor('float32', float32Data, [1, 3, height, width]);
    return outputTensor;
  }

  // #region factory
  static async fromImage(imageData: ImageData, options?: TensorFromImageOptions): Promise<Tensor>;
  static async fromImage(imageElement: HTMLImageElement, options?: TensorFromImageOptions): Promise<Tensor>;
  static async fromImage(bitmap: ImageBitmap, options: TensorFromImageOptions): Promise<Tensor>;
  static async fromImage(urlSource: string, options?: TensorFromImageOptions): Promise<Tensor>;

  static async fromImage(image: ImageData|HTMLImageElement|ImageBitmap|string, options?: TensorFromImageOptions):
      Promise<Tensor> {
    // checking the type of image object
    const isHTMLImageEle = typeof (HTMLImageElement) !== 'undefined' && image instanceof HTMLImageElement;
    const isImageDataEle = typeof (ImageData) !== 'undefined' && image instanceof ImageData;
    const isImageBitmap = typeof (ImageBitmap) !== 'undefined' && image instanceof ImageBitmap;
    const isString = typeof image === 'string';

    let data: Uint8ClampedArray|undefined;
    let tensorConfig: TensorFromImageOptions = options ?? {};

    // filling and checking image configuration options
    if (isHTMLImageEle) {
      // HTMLImageElement - image object - format is RGBA by default
      const canvas = document.createElement('canvas');
      canvas.width = image.width;
      canvas.height = image.height;
      const pixels2DContext = canvas.getContext('2d');

      if (pixels2DContext != null) {
        let height = image.height;
        let width = image.width;
        if (options !== undefined && options.resizedHeight !== undefined && options.resizedWidth !== undefined) {
          height = options.resizedHeight;
          width = options.resizedWidth;
        }

        if (options !== undefined) {
          tensorConfig = options;
          if (options.tensorFormat !== undefined) {
            throw new Error('Image input config format must be RGBA for HTMLImageElement');
          } else {
            tensorConfig.tensorFormat = 'RGBA';
          }
          if (options.height !== undefined && options.height !== height) {
            throw new Error('Image input config height doesn\'t match HTMLImageElement height');
          } else {
            tensorConfig.height = height;
          }
          if (options.width !== undefined && options.width !== width) {
            throw new Error('Image input config width doesn\'t match HTMLImageElement width');
          } else {
            tensorConfig.width = width;
          }
        } else {
          tensorConfig.tensorFormat = 'RGBA';
          tensorConfig.height = height;
          tensorConfig.width = width;
        }

        pixels2DContext.drawImage(image, 0, 0);
        data = pixels2DContext.getImageData(0, 0, width, height).data;
      } else {
        throw new Error('Can not access image data');
      }

    } else if (isImageDataEle) {
      // ImageData - image object - format is RGBA by default
      const format = 'RGBA';
      let height: number;
      let width: number;

      if (options !== undefined && options.resizedWidth !== undefined && options.resizedHeight !== undefined) {
        height = options.resizedHeight;
        width = options.resizedWidth;
      } else {
        height = image.height;
        width = image.width;
      }

      if (options !== undefined) {
        tensorConfig = options;
        if (options.bitmapFormat !== undefined && options.bitmapFormat !== format) {
          throw new Error('Image input config format must be RGBA for ImageData');
        } else {
          tensorConfig.bitmapFormat = 'RGBA';
        }
      } else {
        tensorConfig.bitmapFormat = 'RGBA';
      }

      tensorConfig.height = height;
      tensorConfig.width = width;

      if (options !== undefined) {
        const tempCanvas = document.createElement('canvas');

        tempCanvas.width = width;
        tempCanvas.height = height;

        const pixels2DContext = tempCanvas.getContext('2d');

        if (pixels2DContext != null) {
          pixels2DContext.putImageData(image, 0, 0);
          data = pixels2DContext.getImageData(0, 0, width, height).data;
        } else {
          throw new Error('Can not access image data');
        }
      } else {
        data = image.data;
      }

    } else if (isImageBitmap) {
      // ImageBitmap - image object - format must be provided by user
      if (options === undefined) {
        throw new Error('Please provide image config with format for Imagebitmap');
      }
      if (options.bitmapFormat !== undefined) {
        throw new Error('Image input config format must be defined for ImageBitmap');
      }

      const pixels2DContext = document.createElement('canvas').getContext('2d');

      if (pixels2DContext != null) {
        const height = image.height;
        const width = image.width;
        pixels2DContext.drawImage(image, 0, 0, width, height);
        data = pixels2DContext.getImageData(0, 0, width, height).data;
        if (options !== undefined) {
          // using square brackets to avoid TS error - type 'never'
          if (options.height !== undefined && options.height !== height) {
            throw new Error('Image input config height doesn\'t match ImageBitmap height');
          } else {
            tensorConfig.height = height;
          }
          // using square brackets to avoid TS error - type 'never'
          if (options.width !== undefined && options.width !== width) {
            throw new Error('Image input config width doesn\'t match ImageBitmap width');
          } else {
            tensorConfig.width = width;
          }
        } else {
          tensorConfig.height = height;
          tensorConfig.width = width;
        }
        return Tensor.bufferToTensor(data, tensorConfig);
      } else {
        throw new Error('Can not access image data');
      }

    } else if (isString) {
      return new Promise((resolve, reject) => {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        if (!image || !context) {
          return reject();
        }
        const newImage = new Image();
        newImage.crossOrigin = 'Anonymous';
        newImage.src = image;
        newImage.onload = () => {
          canvas.width = newImage.width;
          canvas.height = newImage.height;
          context.drawImage(newImage, 0, 0, canvas.width, canvas.height);
          const img = context.getImageData(0, 0, canvas.width, canvas.height);
          if (options !== undefined) {
            if (options.height !== undefined && options.height !== canvas.height) {
              throw new Error('Image input config height doesn\'t match height');
            } else {
              tensorConfig.height = canvas.height;
            }
            if (options.width !== undefined && options.width !== canvas.width) {
              throw new Error('Image input config width doesn\'t match width');
            } else {
              tensorConfig.width = canvas.width;
            }
          } else {
            tensorConfig.height = canvas.height;
            tensorConfig.width = canvas.width;
          }
          resolve(Tensor.bufferToTensor(img.data, tensorConfig));
        };
      });
    } else {
      throw new Error('Input data provided is not supported - aborted tensor creation');
    }

    if (data !== undefined) {
      return Tensor.bufferToTensor(data, tensorConfig);
    } else {
      throw new Error('Input data provided is not supported - aborted tensor creation');
    }
  }

  toDataURL(options?: TensorToImageDataOptions): string {
    const canvas = document.createElement('canvas');
    canvas.width = this.dims[3];
    canvas.height = this.dims[2];
    const pixels2DContext = canvas.getContext('2d');

    if (pixels2DContext != null) {
      // Default values for height and width & format
      let width: number;
      let height: number;
      if (options?.tensorLayout !== undefined && options.tensorLayout === 'NHWC') {
        width = this.dims[2];
        height = this.dims[3];
      } else {  // Default layout is NCWH
        width = this.dims[3];
        height = this.dims[2];
      }

      const inputformat = options?.format !== undefined ? options.format : 'RGB';

      const norm = options?.norm;
      let normMean: [number, number, number, number];
      let normBias: [number, number, number, number];
      if (norm === undefined || norm.mean === undefined) {
        normMean = [255, 255, 255, 255];
      } else {
        if (typeof (norm.mean) === 'number') {
          normMean = [norm.mean, norm.mean, norm.mean, norm.mean];
        } else {
          normMean = [norm.mean[0], norm.mean[1], norm.mean[2], 0];
          if (norm.mean[3] !== undefined) {
            normMean[3] = norm.mean[3];
          }
        }
      }
      if (norm === undefined || norm.bias === undefined) {
        normBias = [0, 0, 0, 0];
      } else {
        if (typeof (norm.bias) === 'number') {
          normBias = [norm.bias, norm.bias, norm.bias, norm.bias];
        } else {
          normBias = [norm.bias[0], norm.bias[1], norm.bias[2], 0];
          if (norm.bias[3] !== undefined) {
            normBias[3] = norm.bias[3];
          }
        }
      }

      const stride = height * width;
      // Default pointer assignments
      let rTensorPointer = 0, gTensorPointer = stride, bTensorPointer = stride * 2, aTensorPointer = -1;

      // Updating the pointer assignments based on the input image format
      if (inputformat === 'RGBA') {
        rTensorPointer = 0;
        gTensorPointer = stride;
        bTensorPointer = stride * 2;
        aTensorPointer = stride * 3;
      } else if (inputformat === 'RGB') {
        rTensorPointer = 0;
        gTensorPointer = stride;
        bTensorPointer = stride * 2;
      } else if (inputformat === 'RBG') {
        rTensorPointer = 0;
        bTensorPointer = stride;
        gTensorPointer = stride * 2;
      }

      for (let i = 0; i < height; i++) {
        for (let j = 0; j < width; j++) {
          const R = ((this.data[rTensorPointer++] as number) - normBias[0]) * normMean[0];  // R value
          const G = ((this.data[gTensorPointer++] as number) - normBias[1]) * normMean[1];  // G value
          const B = ((this.data[bTensorPointer++] as number) - normBias[2]) * normMean[2];  // B value
          const A = aTensorPointer === -1 ?
              255 :
              ((this.data[aTensorPointer++] as number) - normBias[3]) * normMean[3];  // A value
          // eslint-disable-next-line @typescript-eslint/restrict-plus-operands
          pixels2DContext.fillStyle = 'rgba(' + R + ',' + G + ',' + B + ',' + A + ')';
          pixels2DContext.fillRect(j, i, 1, 1);
        }
      }
      return canvas.toDataURL();
    } else {
      throw new Error('Can not access image data');
    }
  }

  toImageData(options?: TensorToImageDataOptions): ImageData {
    const pixels2DContext = document.createElement('canvas').getContext('2d');
    let image: ImageData;
    if (pixels2DContext != null) {
      // Default values for height and width & format
      let width: number;
      let height: number;
      let channels: number;
      if (options?.tensorLayout !== undefined && options.tensorLayout === 'NHWC') {
        width = this.dims[2];
        height = this.dims[1];
        channels = this.dims[3];
      } else {  // Default layout is NCWH
        width = this.dims[3];
        height = this.dims[2];
        channels = this.dims[1];
      }
      const inputformat = options !== undefined ? (options.format !== undefined ? options.format : 'RGB') : 'RGB';

      const norm = options?.norm;
      let normMean: [number, number, number, number];
      let normBias: [number, number, number, number];
      if (norm === undefined || norm.mean === undefined) {
        normMean = [255, 255, 255, 255];
      } else {
        if (typeof (norm.mean) === 'number') {
          normMean = [norm.mean, norm.mean, norm.mean, norm.mean];
        } else {
          normMean = [norm.mean[0], norm.mean[1], norm.mean[2], 255];
          if (norm.mean[3] !== undefined) {
            normMean[3] = norm.mean[3];
          }
        }
      }
      if (norm === undefined || norm.bias === undefined) {
        normBias = [0, 0, 0, 0];
      } else {
        if (typeof (norm.bias) === 'number') {
          normBias = [norm.bias, norm.bias, norm.bias, norm.bias];
        } else {
          normBias = [norm.bias[0], norm.bias[1], norm.bias[2], 0];
          if (norm.bias[3] !== undefined) {
            normBias[3] = norm.bias[3];
          }
        }
      }

      const stride = height * width;
      if (options !== undefined) {
        if (options.height !== undefined && options.height !== height) {
          throw new Error('Image output config height doesn\'t match tensor height');
        }
        if (options.width !== undefined && options.width !== width) {
          throw new Error('Image output config width doesn\'t match tensor width');
        }
        if (options.format !== undefined && (channels === 4 && options.format !== 'RGBA') ||
            (channels === 3 && (options.format !== 'RGB' && options.format !== 'BGR'))) {
          throw new Error('Tensor format doesn\'t match input tensor dims');
        }
      }

      // Default pointer assignments
      const step = 4;
      let rImagePointer = 0, gImagePointer = 1, bImagePointer = 2, aImagePointer = 3;
      let rTensorPointer = 0, gTensorPointer = stride, bTensorPointer = stride * 2, aTensorPointer = -1;

      // Updating the pointer assignments based on the input image format
      if (inputformat === 'RGBA') {
        rTensorPointer = 0;
        gTensorPointer = stride;
        bTensorPointer = stride * 2;
        aTensorPointer = stride * 3;
      } else if (inputformat === 'RGB') {
        rTensorPointer = 0;
        gTensorPointer = stride;
        bTensorPointer = stride * 2;
      } else if (inputformat === 'RBG') {
        rTensorPointer = 0;
        bTensorPointer = stride;
        gTensorPointer = stride * 2;
      }

      image = pixels2DContext.createImageData(width, height);

      for (let i = 0; i < height * width;
           rImagePointer += step, gImagePointer += step, bImagePointer += step, aImagePointer += step, i++) {
        image.data[rImagePointer] = ((this.data[rTensorPointer++] as number) - normBias[0]) * normMean[0];  // R value
        image.data[gImagePointer] = ((this.data[gTensorPointer++] as number) - normBias[1]) * normMean[1];  // G value
        image.data[bImagePointer] = ((this.data[bTensorPointer++] as number) - normBias[2]) * normMean[2];  // B value
        image.data[aImagePointer] = aTensorPointer === -1 ?
            255 :
            ((this.data[aTensorPointer++] as number) - normBias[3]) * normMean[3];  // A value
      }

    } else {
      throw new Error('Can not access image data');
    }
    return image;
  }

  // #region fields
  readonly dims: readonly number[];
  readonly type: TensorType;
  readonly data: TensorDataType;
  readonly size: number;
  // #endregion

  // #region tensor utilities
  reshape(dims: readonly number[]): Tensor {
    return new Tensor(this.type, this.data, dims);
  }
  // #endregion
}
