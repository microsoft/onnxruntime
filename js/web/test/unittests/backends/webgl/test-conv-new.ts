// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Attribute} from '../../../../lib/onnxjs/attribute';
import {Backend, InferenceHandler, resolveBackend, SessionHandler} from '../../../../lib/onnxjs/backend';
import {Profiler} from '../../../../lib/onnxjs/instrument';
import {Tensor} from '../../../../lib/onnxjs/tensor';
import {PoolConvUtil} from '../../../../lib/onnxjs/util';
import {TensorResultValidator} from '../../../test-runner';
import {createMockGraph} from '../../../test-shared';

import {conv2d} from './test-conv-utils';

function createRandomArray(size: number): Float32Array {
  const randomTable = [0, 3, 6, 9, 2, 5, 8, 1, 4, 7];
  return new Float32Array(
      Array.from({length: size}, (_v, k) => randomTable[k % 10] * 0.1 + randomTable[Math.trunc(k / 10) % 10] * 0.01));
}
interface TestData {
  inputShape: number[];
  kernelShape: number[];
  biasShape: number[];
  autoPad?: string;
  pads?: number[];
  dilations: number[];
  strides: number[];
  group: number;
}
function getTestData(): TestData[] {
  return [
    {
      inputShape: [1, 3, 416, 416],
      kernelShape: [64, 3, 3, 3],
      biasShape: [],
      autoPad: 'SAME_UPPER',
      dilations: [1, 1],
      strides: [1, 1],
      group: 1
    },
    {
      inputShape: [1, 3, 224, 224],
      kernelShape: [64, 3, 3, 3],
      biasShape: [64],
      pads: [0, 0, 0, 0],
      dilations: [1, 1],
      strides: [2, 2],
      group: 1
    },
    {
      inputShape: [1, 64, 55, 55],
      kernelShape: [16, 64, 1, 1],
      biasShape: [16],
      pads: [0, 0, 0, 0],
      dilations: [1, 1],
      strides: [1, 1],
      group: 1
    },
    // {
    //   inputShape: [1, 16, 55, 55],
    //   kernelShape: [64, 16, 1, 1],
    //   biasShape: [64],
    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 16, 55, 55],
    //   kernelShape: [64, 16, 3, 3],
    //   biasShape: [64],
    //   pads: [1, 1, 1, 1],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 128, 55, 55],
    //   kernelShape: [16, 128, 1, 1],
    //   biasShape: [16],
    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 16, 55, 55],
    //   kernelShape: [64, 16, 1, 1],
    //   biasShape: [64],
    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 16, 55, 55],
    //   kernelShape: [64, 16, 3, 3],
    //   biasShape: [64],
    //   pads: [1, 1, 1, 1],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 128, 27, 27],
    //   kernelShape: [32, 128, 1, 1],
    //   biasShape: [32],
    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 32, 27, 27],
    //   kernelShape: [128, 32, 1, 1],
    //   biasShape: [128],
    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 32, 27, 27],
    //   kernelShape: [128, 32, 3, 3],
    //   biasShape: [128],
    //   pads: [1, 1, 1, 1],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 256, 27, 27],
    //   kernelShape: [32, 256, 1, 1],
    //   biasShape: [32],
    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 32, 27, 27],
    //   kernelShape: [128, 32, 1, 1],
    //   biasShape: [128],
    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 32, 27, 27],
    //   kernelShape: [128, 32, 3, 3],
    //   biasShape: [128],
    //   pads: [1, 1, 1, 1],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 256, 13, 13],
    //   kernelShape: [48, 256, 1, 1],
    //   biasShape: [48],
    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 48, 13, 13],
    //   kernelShape: [192, 48, 1, 1],
    //   biasShape: [192],
    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 48, 13, 13],
    //   kernelShape: [192, 48, 3, 3],
    //   biasShape: [192],
    //   pads: [1, 1, 1, 1],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 384, 13, 13],
    //   kernelShape: [48, 384, 1, 1],
    //   biasShape: [48],
    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 48, 13, 13],
    //   kernelShape: [192, 48, 1, 1],
    //   biasShape: [192],
    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 48, 13, 13],
    //   kernelShape: [192, 48, 3, 3],
    //   biasShape: [192],
    //   pads: [1, 1, 1, 1],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 384, 13, 13],
    //   kernelShape: [64, 384, 1, 1],
    //   biasShape: [64],
    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 64, 13, 13],
    //   kernelShape: [256, 64, 1, 1],
    //   biasShape: [256],
    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 64, 13, 13],
    //   kernelShape: [256, 64, 3, 3],
    //   biasShape: [256],
    //   pads: [1, 1, 1, 1],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 512, 13, 13],
    //   kernelShape: [64, 512, 1, 1],
    //   biasShape: [64],
    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 64, 13, 13],
    //   kernelShape: [256, 64, 1, 1],
    //   biasShape: [256],
    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 64, 13, 13],
    //   kernelShape: [256, 64, 3, 3],
    //   biasShape: [256],
    //   pads: [1, 1, 1, 1],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 512, 13, 13],
    //   kernelShape: [1000, 512, 1, 1],
    //   biasShape: [1000],
    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    {
      inputShape: [1, 3, 7, 7],
      kernelShape: [8, 3, 3, 3],
      biasShape: [],
      pads: [1, 1, 1, 1],
      dilations: [1, 1],
      strides: [1, 1],
      group: 1
    },
    {
      inputShape: [1, 2, 3, 3],
      kernelShape: [4, 2, 1, 1],
      biasShape: [],
      pads: [0, 0, 0, 0],
      dilations: [1, 1],
      strides: [1, 1],
      group: 1
    },
    {
      inputShape: [1, 3, 224, 224],
      kernelShape: [64, 3, 7, 7],
      biasShape: [],
      pads: [3, 3, 3, 3],
      dilations: [1, 1],
      strides: [2, 2],
      group: 1
    },
    // {
    //   inputShape: [1, 64, 56, 56],
    //   kernelShape: [256, 64, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 64, 56, 56],
    //   kernelShape: [64, 64, 3, 3],
    //   biasShape:[],

    //   pads: [1, 1, 1, 1],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 64, 56, 56],
    //   kernelShape: [256, 64, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 256, 56, 56],
    //   kernelShape: [64, 256, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 64, 56, 56],
    //   kernelShape: [64, 64, 3, 3],
    //   biasShape:[],

    //   pads: [1, 1, 1, 1],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 64, 56, 56],
    //   kernelShape: [256, 64, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 256, 56, 56],
    //   kernelShape: [64, 256, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 64, 56, 56],
    //   kernelShape: [64, 64, 3, 3],
    //   biasShape:[],

    //   pads: [1, 1, 1, 1],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 64, 56, 56],
    //   kernelShape: [256, 64, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 256, 56, 56],
    //   kernelShape: [128, 256, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 256, 56, 56],
    //   kernelShape: [512, 256, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [2, 2],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 128, 56, 56],
    //   kernelShape: [128, 128, 3, 3],
    //   biasShape:[],

    //   pads: [1, 1, 1, 1],
    //   dilations: [1, 1],
    //   strides: [2, 2],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 128, 28, 28],
    //   kernelShape: [512, 128, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 512, 28, 28],
    //   kernelShape: [128, 512, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 128, 28, 28],
    //   kernelShape: [128, 128, 3, 3],
    //   biasShape:[],

    //   pads: [1, 1, 1, 1],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 128, 28, 28],
    //   kernelShape: [512, 128, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 512, 28, 28],
    //   kernelShape: [128, 512, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 128, 28, 28],
    //   kernelShape: [128, 128, 3, 3],
    //   biasShape:[],

    //   pads: [1, 1, 1, 1],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 128, 28, 28],
    //   kernelShape: [512, 128, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 512, 28, 28],
    //   kernelShape: [128, 512, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 128, 28, 28],
    //   kernelShape: [128, 128, 3, 3],
    //   biasShape:[],

    //   pads: [1, 1, 1, 1],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 128, 28, 28],
    //   kernelShape: [512, 128, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 512, 28, 28],
    //   kernelShape: [256, 512, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 512, 28, 28],
    //   kernelShape: [1024, 512, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [2, 2],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 256, 28, 28],
    //   kernelShape: [256, 256, 3, 3],
    //   biasShape:[],

    //   pads: [1, 1, 1, 1],
    //   dilations: [1, 1],
    //   strides: [2, 2],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 256, 14, 14],
    //   kernelShape: [1024, 256, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 1024, 14, 14],
    //   kernelShape: [256, 1024, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 256, 14, 14],
    //   kernelShape: [256, 256, 3, 3],
    //   biasShape:[],

    //   pads: [1, 1, 1, 1],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 256, 14, 14],
    //   kernelShape: [1024, 256, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 1024, 14, 14],
    //   kernelShape: [256, 1024, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 256, 14, 14],
    //   kernelShape: [256, 256, 3, 3],
    //   biasShape:[],

    //   pads: [1, 1, 1, 1],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 256, 14, 14],
    //   kernelShape: [1024, 256, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 1024, 14, 14],
    //   kernelShape: [256, 1024, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 256, 14, 14],
    //   kernelShape: [256, 256, 3, 3],
    //   biasShape:[],

    //   pads: [1, 1, 1, 1],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 256, 14, 14],
    //   kernelShape: [1024, 256, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 1024, 14, 14],
    //   kernelShape: [256, 1024, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 256, 14, 14],
    //   kernelShape: [256, 256, 3, 3],
    //   biasShape:[],

    //   pads: [1, 1, 1, 1],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 256, 14, 14],
    //   kernelShape: [1024, 256, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 1024, 14, 14],
    //   kernelShape: [256, 1024, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 256, 14, 14],
    //   kernelShape: [256, 256, 3, 3],
    //   biasShape:[],

    //   pads: [1, 1, 1, 1],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 256, 14, 14],
    //   kernelShape: [1024, 256, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 1024, 14, 14],
    //   kernelShape: [512, 1024, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 1024, 14, 14],
    //   kernelShape: [2048, 1024, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [2, 2],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 512, 14, 14],
    //   kernelShape: [512, 512, 3, 3],
    //   biasShape:[],

    //   pads: [1, 1, 1, 1],
    //   dilations: [1, 1],
    //   strides: [2, 2],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 512, 7, 7],
    //   kernelShape: [2048, 512, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 2048, 7, 7],
    //   kernelShape: [512, 2048, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    {
      inputShape: [1, 512, 7, 7],
      kernelShape: [512, 512, 3, 3],
      biasShape: [],
      pads: [1, 1, 1, 1],
      dilations: [1, 1],
      strides: [1, 1],
      group: 1
    },
    {
      inputShape: [1, 512, 7, 7],
      kernelShape: [2048, 512, 1, 1],
      biasShape: [],

      pads: [0, 0, 0, 0],
      dilations: [1, 1],
      strides: [1, 1],
      group: 1
    },
    // {
    //   inputShape: [1, 2048, 7, 7],
    //   kernelShape: [512, 2048, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 512, 7, 7],
    //   kernelShape: [512, 512, 3, 3],
    //   biasShape:[],

    //   pads: [1, 1, 1, 1],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
    // {
    //   inputShape: [1, 512, 7, 7],
    //   kernelShape: [2048, 512, 1, 1],
    //   biasShape:[],

    //   pads: [0, 0, 0, 0],
    //   dilations: [1, 1],
    //   strides: [1, 1],
    //   group: 1
    // },
  ];
}

const validator = new TensorResultValidator('webgl');
let webglBackend: Backend|undefined;
let webglSessionhandler: SessionHandler|undefined;
let webglInferenceHandler: InferenceHandler|undefined;

function webglConv(
    inputTensor: Tensor, kernelTensor: Tensor, biasTensor: Tensor|null, autoPad: string|undefined, dilations: number[],
    pads: number[]|undefined, strides: number[]): Tensor {
  const attributes = new Attribute(undefined);
  attributes.set('dilations', 'ints', dilations);
  attributes.set('auto_pad', 'string', autoPad ? autoPad : '');
  attributes.set('kernel_shape', 'ints', kernelTensor.dims.slice(2));
  if (pads) {
    attributes.set('pads', 'ints', pads);
  }
  attributes.set('strides', 'ints', strides);
  const graph = createMockGraph('Conv', attributes);
  const op = webglSessionhandler!.resolve(graph.getNodes()[0], [{domain: '', version: 7}], graph);
  const inputs = [inputTensor, kernelTensor];
  if (biasTensor) {
    inputs.push(biasTensor);
  }
  return (op.impl(webglInferenceHandler!, inputs, op.context))[0];
}
function cpuConv(
    inputTensor: Tensor, kernelTensor: Tensor, biasTensor: Tensor|null, autoPad: string|undefined, dilations: number[],
    pads: number[]|undefined, strides: number[]): Tensor {
  const attributes = new Attribute(undefined);
  attributes.set('dilations', 'ints', dilations);
  attributes.set('auto_pad', 'string', autoPad ? autoPad : '');
  attributes.set('kernel_shape', 'ints', kernelTensor.dims.slice(2));
  if (pads) {
    attributes.set('pads', 'ints', pads);
  }
  attributes.set('strides', 'ints', strides);

  const x = inputTensor;
  const w = kernelTensor;
  const b = biasTensor || undefined;

  const adjustedPads = pads ? pads.slice(0) : [0, 0, 0, 0];
  const outputDims = PoolConvUtil.computeConvOutputShape(
      x.dims, w.dims, strides, dilations, kernelTensor.dims.slice(2), adjustedPads, autoPad);
  const y = new Tensor(outputDims, x.type);
  conv2d(y, x, w, b, dilations, 1, adjustedPads, strides);
  return y;
}
describe('New Conv tests', () => {
  before(async () => {
    const profiler = Profiler.create();
    webglBackend = await resolveBackend('webgl');
    webglSessionhandler = webglBackend.createSessionHandler({profiler});
    webglInferenceHandler = webglSessionhandler.createInferenceHandler();
  });
  const testDataSet = getTestData();
  for (let k = 0; k < testDataSet.length; ++k) {
    const testData = testDataSet[k];
    describe(`Testing ${JSON.stringify(testData)}`, () => {
      const inputData = createRandomArray(testData.inputShape.reduce((a, b) => a * b));
      const kernelData = createRandomArray(testData.kernelShape.reduce((a, b) => a * b));
      const biasData = testData.biasShape.length === 1 ? createRandomArray(testData.biasShape[0]) : null;
      const rgbas = [false];
      rgbas.forEach(rgba => {
        describe(`RGBA: ${rgba}`, () => {
          before(function() {
            const patchSize = testData.kernelShape.slice(1).reduce((a, b) => a * b);
            if (rgba && patchSize % 4 !== 0) {
              // eslint-disable-next-line no-invalid-this
              this.skip();
            }
          });
          it('', () => {
            // create new Tensors otherwise the session/inference level caching would cause issues
            const inputTensor = new Tensor(testData.inputShape, 'float32', undefined, undefined, inputData);
            const kernelTensor = new Tensor(testData.kernelShape, 'float32', undefined, undefined, kernelData);
            const biasTensor =
                biasData ? new Tensor(testData.biasShape, 'float32', undefined, undefined, biasData) : null;
            const actual = webglConv(
                inputTensor, kernelTensor, biasTensor, testData.autoPad, testData.dilations, testData.pads,
                testData.strides);
            const expected = cpuConv(
                inputTensor, kernelTensor, biasTensor, testData.autoPad, testData.dilations, testData.pads,
                testData.strides);
            try {
              validator.checkTensorResult([actual], [expected]);
            } catch {
              console.log(actual.dims, `[${actual.numberData.slice(0, 20).join(',')},...]`);
              console.log(expected.dims, `[${expected.numberData.slice(0, 20).join(',')},...]`);
              throw new Error('Expected and Actual did not match');
            }
          });
        });
      });
    });
  }
});
