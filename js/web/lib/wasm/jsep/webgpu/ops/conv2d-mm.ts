// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor';
import {GpuDataType, ProgramInfoLoader, ProgramMetadata} from '../types';

import {createConv2DMatMulProgramInfo} from './3rd-party/conv2d_mm_webgpu';
import {ConvAttributes} from './conv';


const createConv2DMatMulProgramMetadata = (hasBias: boolean, cacheHint: string): ProgramMetadata => ({
  name: 'Conv2DMatMul',
  inputTypes: hasBias ? [GpuDataType.default, GpuDataType.default, GpuDataType.default] :
                        [GpuDataType.default, GpuDataType.default],
  cacheHint
});

export const createConv2DMatMulProgramInfoLoader =
    (inputs: readonly TensorView[], attributes: ConvAttributes, outputShape: readonly number[], dimAOuter: number,
     dimBOuter: number, dimInner: number, hasBias: boolean, sequentialAccessByThreads: boolean): ProgramInfoLoader => {
      const metadata = createConv2DMatMulProgramMetadata(hasBias, attributes.cacheKey);
      return {
        ...metadata,
        get: () => createConv2DMatMulProgramInfo(
            inputs, metadata, attributes, outputShape, dimAOuter, dimBOuter, dimInner, hasBias,
            sequentialAccessByThreads)
      };
    };
