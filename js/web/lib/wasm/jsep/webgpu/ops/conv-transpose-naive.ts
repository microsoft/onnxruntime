// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
import {TensorView} from '../../tensor';
import {GpuDataType, ProgramInfoLoader, ProgramMetadata} from '../types';

import {createConvTranspose2DProgramInfo} from './3rd-party/conv_backprop_webgpu'
import {ConvTransposeAttributes} from './conv-transpose';


const createConvTranspose2DProgramMetadata = (hasBias: boolean, cacheHint: string): ProgramMetadata => ({
  name: 'ConvTranspose2D',
  inputTypes: hasBias ? [GpuDataType.default, GpuDataType.default, GpuDataType.default] :
                        [GpuDataType.default, GpuDataType.default],
  cacheHint
});

export const createConvTranspose2DProgramInfoLoader =
    (inputs: readonly TensorView[], attributes: ConvTransposeAttributes, outputShape: readonly number[],
     hasBias: boolean): ProgramInfoLoader => {
      const metadata = createConvTranspose2DProgramMetadata(hasBias, attributes.cacheKey);
      return {
        ...metadata,
        get: () => createConvTranspose2DProgramInfo(inputs, metadata, attributes, outputShape, hasBias)
      };
    };
