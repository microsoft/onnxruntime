// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor-view';
import {GpuDataType, ProgramInfoLoader, ProgramMetadata} from '../types';

import {createConv2DTransposeMatMulProgramInfo} from './3rd-party/conv_backprop_mm_webgpu';
import {ConvTransposeAttributes} from './conv-transpose';


const createConv2DTransposeMatMulProgramMetadata = (hasBias: boolean, cacheHint: string): ProgramMetadata => ({
  name: 'Conv2DTransposeMatMul',
  inputTypes: hasBias ? [GpuDataType.default, GpuDataType.default, GpuDataType.default] :
                        [GpuDataType.default, GpuDataType.default],
  cacheHint
});

export const createConv2DTransposeMatMulProgramInfoLoader =
    (inputs: readonly TensorView[], attributes: ConvTransposeAttributes, outputShape: readonly number[],
     dimAOuter: number, dimBOuter: number, dimInner: number, hasBias: boolean,
     sequentialAccessByThreads: boolean): ProgramInfoLoader => {
      const metadata = createConv2DTransposeMatMulProgramMetadata(hasBias, attributes.cacheKey);
      return {
        ...metadata,
        get: () => createConv2DTransposeMatMulProgramInfo(
            inputs, metadata, attributes, outputShape, dimAOuter, dimBOuter, dimInner, hasBias,
            sequentialAccessByThreads)
      };
    };
