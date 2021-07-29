// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tensor} from '../../../tensor';
import {WebGLInferenceHandler} from '../inference-handler';
import {calculateOutputShape, ConvAttributes} from './conv';
import {createPackedIm2ColProgramInfo} from './im2col-pack';
import {createPackedMatmulProgramInfo} from './matmul-pack';

export const conv2DPacked =
    (inferenceHandler: WebGLInferenceHandler, inputs: readonly Tensor[], attributes: ConvAttributes): Tensor => {
      const xshape = inputs[0].dims;
      const kshape = inputs[1].dims;
      const outputShape =
          calculateOutputShape(xshape, kshape, attributes.dilations, attributes.pads, attributes.strides);

      // run im2col
      const im2colOutput = inferenceHandler.run(
          createPackedIm2ColProgramInfo(inferenceHandler, inputs[0], inputs[1], outputShape, attributes), [inputs[0]]);

      // reshape kernel
      const kernelReshaped = inferenceHandler.reshapePacked(inputs[1], [kshape[0], kshape[1] * kshape[2] * kshape[3]]);

      // run matmul
      const matmulInputs =
          (inputs.length === 3) ? [kernelReshaped, im2colOutput, inputs[2]] : [kernelReshaped, im2colOutput];
      const matmulOutput =
          inferenceHandler.run(createPackedMatmulProgramInfo(inferenceHandler, matmulInputs, attributes), matmulInputs);

      // reshape output
      const outputReshaped = inferenceHandler.reshapePacked(matmulOutput, outputShape);
      return outputReshaped;
    };
