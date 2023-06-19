// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor';
import {GpuDataType, ProgramInfoLoader, ProgramMetadata} from '../types';

import {createConv2dTransposeMatMulProgramInfo} from './3rd-party/conv_backprop_mm_webgpu';
import {ConvAttributes} from './conv';


const createConv2dTransposeMatMulProgramMetadata = (hasBias: boolean, cacheHint: string): ProgramMetadata => ({
  name: 'ConvTranspose2DMatMul',
  inputTypes: hasBias ? [GpuDataType.default, GpuDataType.default, GpuDataType.default] :
                        [GpuDataType.default, GpuDataType.default],
  cacheHint
});

export const createConv2dTransposeMatMulProgramInfoLoader =
    (inputs: readonly TensorView[], attributes: ConvAttributes, outputShape: readonly number[], dimAOuter: number,
     dimBOuter: number, dimInner: number, hasBias: boolean, sequentialAccessByThreads: boolean): ProgramInfoLoader => {
      const metadata = createConv2dTransposeMatMulProgramMetadata(hasBias, attributes.cacheKey);
      return {
        ...metadata,
        get: () => createConv2dTransposeMatMulProgramInfo(
            inputs, metadata, attributes, outputShape, dimAOuter, dimBOuter, dimInner, hasBias,
            sequentialAccessByThreads)
      };
    };
