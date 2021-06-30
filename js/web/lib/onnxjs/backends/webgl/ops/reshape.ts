// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Reshape} from '../../../ops/reshape';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
import {TextureLayout} from '../types';
import {getPackedShape} from '../utils';
import {createPackedReshapeProgramInfo} from './reshape-packed';
import {ProgramInfo, TextureType} from '../types';
import {getGlsl} from '../glsl-source';

const createReshapeProgramInfo =
    (handler: WebGLInferenceHandler, input0: Tensor, input1: Tensor):
        ProgramInfo => {
        if (handler.session.pack) {
            return createPackedReshapeProgramInfo(handler, input0, input1);
        } else {
            // TODO: how do we handle no-op? For reshape in unpacked mode, we don't need to run
            // any shaders. Instead, we can just use the same tensor data with different dims value.
            // This would require a change in how we represent tensor and whether we allow de-couple
            // tensor data (memory) with its meta-data (like dims, types ect).
            // Before we implement the feature above, temporarily return a dummy programInfo.
            const glsl = getGlsl(handler.session.backend.glContext.version);
            return {
                inputTypes: [TextureType.unpacked],
                inputNames: ['A'],
                output: { dims: input1.integerData, type: input0.type, TextureType.unpacked },
                shaderSource: `
             void main() {
               vec4 v = ${glsl.texture2D}(A, TexCoords);
               ${glsl.output} = v;
             }
             `,
                hasMain: true
            };
        };
    }

export const reshape = (handler: WebGLInferenceHandler, inputs: Tensor[]):
    Tensor[] => [handler.run(createReshapeProgramInfo(handler, inputs[0], inputs[1]), inputs)];

// export class WebGLReshape extends Reshape {
//   packedImpl: WebGLReshapePacked;
//   constructor() {
//     super();
//     this.packedImpl = new WebGLReshapePacked();
//   }
//   run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
//     if (inferenceHandler.session.pack) {
//       return inferenceHandler.run(this.packedImpl, inputs);
//     } else {
//       const reshapedDims = ShapeUtil.calculateReshapedDims(inputs[0].dims, inputs[1].integerData);
//       const reshapedTensor = reshape(inferenceHandler, inputs[0], reshapedDims);
//       return [reshapedTensor];
//     }
//   }
// }

// export function reshape(
//     inferenceHandler: WebGLInferenceHandler, input: Tensor, reshapedDims: readonly number[]): Tensor {
//   const inputTD = inferenceHandler.getOrCreateTextureData(input);
//   let packedShape = reshapedDims;
//   if (inputTD.channels === 4) {
//     packedShape = getPackedShape(reshapedDims);
//   }
//   const newTextureLayout: TextureLayout = {
//     channels: inputTD.channels,
//     height: inputTD.height,
//     width: inputTD.width,
//     // handle reshaping into scalar Tensors
//     shape: packedShape.length !== 0 ? packedShape : [1],
//     strides: ShapeUtil.computeStrides(packedShape),
//     unpackedShape: reshapedDims,
//   };

//   const newTextureData = inferenceHandler.createSharedTextureData(newTextureLayout, input.type, inputTD.texture);
//   return newTextureData.tensor;
// }
