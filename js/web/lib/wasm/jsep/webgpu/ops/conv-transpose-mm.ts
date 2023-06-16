// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor';
import {GpuDataType, ProgramInfoLoader, ProgramMetadata} from '../types';

import {createConvTranspose2DMatMulProgramInfo} from './3rd-party/conv_backprop_mm_webgpu';
import {ConvAttributes} from './conv';


const createConvTranspose2DMatMulProgramMetadata = (hasBias: boolean, cacheHint: string): ProgramMetadata => ({
  name: 'ConvTranspose2DMatMul',
  inputTypes: hasBias ? [GpuDataType.default, GpuDataType.default, GpuDataType.default] :
                        [GpuDataType.default, GpuDataType.default],
  cacheHint
});

export const createConvTranspose2DMatMulProgramInfoLoader =
    (inputs: readonly TensorView[], attributes: ConvAttributes, outputShape: readonly number[], dimAOuter: number,
     dimBOuter: number, dimInner: number, hasBias: boolean, sequentialAccessByThreads: boolean): ProgramInfoLoader => {
      const metadata = createConvTranspose2DMatMulProgramMetadata(hasBias, attributes.cacheKey);
      return {
        ...metadata,
        get: () => createConvTranspose2DMatMulProgramInfo(
            inputs, metadata, attributes, outputShape, dimAOuter, dimBOuter, dimInner, hasBias,
            sequentialAccessByThreads)
      };
    };
