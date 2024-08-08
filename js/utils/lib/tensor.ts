// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tensor as BasicTensor, TensorConstructor} from 'onnxruntime-common';

import type {TensorFromImageExtension} from './tensor-from-image-type.js';
import {tensorFromImage} from './tensor-from-image-impl.js';

/**
 * The Tensor class with additional methods for constructing tensor from image.
 */
// TODO: when add more extension, use a TypeScript interface to merge them.
//
// eslint-disable-next-line @typescript-eslint/naming-convention
export const Tensor: TensorConstructor&TensorFromImageExtension = new Proxy(BasicTensor, {
  get: (target, prop) => {
    if (prop === 'fromImage') {
      return tensorFromImage;
    }
    return Reflect.get(target, prop);
  }
});
