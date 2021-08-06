// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {MatMul} from '../../../ops/matmul';
import {Tensor} from '../../../tensor';
import {BroadcastUtil, ShapeUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, WebGLOperator} from '../types';
import {getCoordsDataType} from '../utils';
import {getActicationSnippet} from './fuse-utils';
import {WebGLMatMulPacked} from './matmul-pack';

export class WebGLMatMul extends MatMul implements WebGLOperator {
  private usePackedTexture?: boolean;

  packedImpl: WebGLMatMulPacked;
  unpackedImpl: WebGLUnpackedMatMul;
  constructor() {
    super();
    this.packedImpl = new WebGLMatMulPacked();
    this.unpackedImpl = new WebGLUnpackedMatMul();
  }

  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    if (this.usePackedTexture === undefined) {
      this.usePackedTexture = inferenceHandler.session.pack;
    }

    if (this.usePackedTexture) {
      return inferenceHandler.run(this.packedImpl, inputs);
    } else {
      return inferenceHandler.run(this.unpackedImpl, inputs);
    }
  }

  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    if (this.usePackedTexture === undefined) {
      this.usePackedTexture = handler.session.pack;
    }

    if (this.usePackedTexture && inputs[0].dims.length > 1) {
      return this.packedImpl.createProgramInfo(handler, inputs);
    } else {
      return this.unpackedImpl.createProgramInfo(handler, inputs);
    }
  }

  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    if (this.usePackedTexture && inputs[0].dims.length > 1) {
      return this.packedImpl.createRunData(handler, programInfo, inputs);
    } else {
      return this.unpackedImpl.createRunData(handler, programInfo, inputs);
    }
  }
}

export class WebGLUnpackedMatMul extends MatMul implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const hasBias = inputs.length > 2;
    const processBias = hasBias ? 'value += getBiasForMatmul();' : '';
    const aShape = inputs[0].dims;
    const bShape = inputs[1].dims;
    const outputShape = BroadcastUtil.calcShape(aShape, bShape, true);
    if (!outputShape) {
      throw new Error('Can\'t use matmul on the given tensors');
    }
    const coordsDataType = getCoordsDataType(outputShape.length);
    const allGlChannels = ['x', 'y', 'z', 'w', 'u', 'v'];

    const {activationFunction, applyActivation} = getActicationSnippet(this.activation);
    const additionalVars = this.activation === 'Clip' ? `
    float min = float(${this.clipMin});
    float max = float(${this.clipMax});` :
                                                        '';
    const getBiasForMatmulSnippet =
        hasBias ? `${getBiasForMatmul(coordsDataType, allGlChannels, inputs[2].dims, outputShape, false)}` : '';

    const rank = outputShape.length;
    const arank = aShape.length;
    const brank = bShape.length;
    const sharedDim = aShape[aShape.length - 1];
    const shaderSource = `
    ${additionalVars}
    ${activationFunction}
    ${getBiasForMatmulSnippet}

      float process(int indices[${rank}]) {
          int a[${arank}];
          int b[${brank}];
          bcastMatmulIndices_A(indices, a);
          bcastMatmulIndices_B(indices, b);

          float value;
          for (int k=0; k<${sharedDim}; ++k) {
              a[${arank - 1}] = k;
              b[${brank - 2}] = k;
              value += _A(a) * _B(b);
          }
          ${processBias}
          ${applyActivation}

          return value;
      }`;
    return {
      inputLayouts: inputs.map(t => handler.getOrCreateTextureLayout(t)),
      outputLayout: handler.createTextureLayoutFromShape(outputShape),
      samplers: hasBias ? ['A', 'B', 'Bias'] : ['A', 'B'],
      shaderSource,
    };
  }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = inputs.map((t, i) => handler.getOrCreateTextureData(t, programInfo.inputLayouts[i]));
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].tensor.type),
      uniformData: {}
    };
  }
}

export function getBiasForMatmul(
    coordsDataType: string, allGlChannels: readonly string[], inShape: readonly number[], outShape: readonly number[],
    isPacked = true): string {
  let unpackedCoordsSnippet = '';
  const inRank = inShape.length;
  const outRank = outShape.length;
  const rankDiff = outRank - inRank;
  if (outRank < 2 && inRank > 0) {
    unpackedCoordsSnippet = 'coords';
  } else {
    unpackedCoordsSnippet = inShape.map((s, i) => `coords.${allGlChannels[i + rankDiff]}`).join(', ');
  }
  const broadcastDims = BroadcastUtil.getBroadcastDims(inShape, outShape);
  const coordsSnippet = broadcastDims.map(d => `coords.${allGlChannels[d + rankDiff]} = 0;`).join('\n');
  const inSize = ShapeUtil.size(inShape);
  const isInputScalar = inSize === 1;
  let output = 'vec4(outputValue.xx, outputValue.yy)';
  if (isInputScalar) {
    output = 'vec4(outputValue.x)';
  }
  const getBiasForMatmulSource = isPacked ? `
vec4 getBiasForMatmul() {
  ${coordsDataType} coords = getOutputCoords();
  ${coordsSnippet}
  vec4 outputValue = getBias(${unpackedCoordsSnippet});
  return ${output};
}` :
                                            `
float getBiasForMatmul() {
  ${coordsDataType} coords = getOutputCoords();
  ${coordsSnippet}
  return getBias(coords.x);
}
`;

  return getBiasForMatmulSource;
}
