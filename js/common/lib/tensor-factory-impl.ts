// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {OptionsDimensions, OptionsFormat, OptionsNormalizationParameters, OptionsTensorFormat, OptionsTensorLayout, TensorFromImageBitmapOptions, TensorFromImageDataOptions, TensorFromImageElementOptions, TensorFromUrlOptions} from './tensor-factory.js';
import {Tensor, TypedTensor} from './tensor.js';

interface BufferToTensorOptions extends OptionsDimensions, OptionsTensorLayout, OptionsNormalizationParameters,
                                        OptionsFormat, OptionsTensorFormat {}

/**
 * Create a new tensor object from image object
 *
 * @param buffer - Extracted image buffer data - assuming RGBA format
 * @param imageFormat - input image configuration - required configurations height, width, format
 * @param tensorFormat - output tensor configuration - Default is RGB format
 */
export const bufferToTensor =
    (buffer: Uint8ClampedArray|undefined, options: BufferToTensorOptions): TypedTensor<'float32'>|
    TypedTensor<'uint8'> => {
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

      const inputformat = options.format !== undefined ? options.format : 'RGBA';
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
    };

/**
 * implementation of Tensor.fromImage().
 */
export const tensorFromImage = async(
    image: ImageData|HTMLImageElement|ImageBitmap|string,
    options?: TensorFromImageDataOptions|TensorFromImageElementOptions|TensorFromImageBitmapOptions|
    TensorFromUrlOptions): Promise<TypedTensor<'float32'>|TypedTensor<'uint8'>> => {
  // checking the type of image object
  const isHTMLImageEle = typeof (HTMLImageElement) !== 'undefined' && image instanceof HTMLImageElement;
  const isImageDataEle = typeof (ImageData) !== 'undefined' && image instanceof ImageData;
  const isImageBitmap = typeof (ImageBitmap) !== 'undefined' && image instanceof ImageBitmap;
  const isString = typeof image === 'string';

  let data: Uint8ClampedArray|undefined;
  let bufferToTensorOptions: BufferToTensorOptions = options ?? {};

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
        bufferToTensorOptions = options;
        if (options.tensorFormat !== undefined) {
          throw new Error('Image input config format must be RGBA for HTMLImageElement');
        } else {
          bufferToTensorOptions.tensorFormat = 'RGBA';
        }
        bufferToTensorOptions.height = height;
        bufferToTensorOptions.width = width;
      } else {
        bufferToTensorOptions.tensorFormat = 'RGBA';
        bufferToTensorOptions.height = height;
        bufferToTensorOptions.width = width;
      }

      pixels2DContext.drawImage(image, 0, 0);
      data = pixels2DContext.getImageData(0, 0, width, height).data;
    } else {
      throw new Error('Can not access image data');
    }
  } else if (isImageDataEle) {
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
      bufferToTensorOptions = options;
    }
    bufferToTensorOptions.format = 'RGBA';
    bufferToTensorOptions.height = height;
    bufferToTensorOptions.width = width;

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

    const canvas = document.createElement('canvas');
    canvas.width = image.width;
    canvas.height = image.height;
    const pixels2DContext = canvas.getContext('2d');

    if (pixels2DContext != null) {
      const height = image.height;
      const width = image.width;
      pixels2DContext.drawImage(image, 0, 0, width, height);
      data = pixels2DContext.getImageData(0, 0, width, height).data;
      bufferToTensorOptions.height = height;
      bufferToTensorOptions.width = width;
      return bufferToTensor(data, bufferToTensorOptions);
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

        bufferToTensorOptions.height = canvas.height;
        bufferToTensorOptions.width = canvas.width;
        resolve(bufferToTensor(img.data, bufferToTensorOptions));
      };
    });
  } else {
    throw new Error('Input data provided is not supported - aborted tensor creation');
  }

  if (data !== undefined) {
    return bufferToTensor(data, bufferToTensorOptions);
  } else {
    throw new Error('Input data provided is not supported - aborted tensor creation');
  }
};
