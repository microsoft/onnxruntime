// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Transpose} from '../../../ops/transpose';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {FunctionType, GlslPositionalFunction} from '../glsl-definitions';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, WebGLOperator} from '../types';

export class WebGLTranspose extends Transpose implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  getOutputShape(inputShapes: Array<readonly number[]>): readonly number[] {
    const perm = this.getAdjustedPerm(inputShapes[0]);
    return ShapeUtil.sortBasedOnPerm(inputShapes[0], perm);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const inputShapes = inputs.map(t => t.dims.slice());
    const perm = this.getAdjustedPerm(inputShapes[0]);
    const unpackedOutputShape = this.getOutputShape(inputShapes);
    const rank = inputs[0].dims.length;
    // A dims=[${inputs[0].dims.toString()}]
    // out Dims=[${unpackedOutputShape.toString()}]
    // based on perm=[${perm.toString()}]
    const shaderSource = `
      ${this.getPermFunctionBody('perm', perm, rank)}
      float process(int indices[${rank}]) {
        int a[${rank}];
        perm(a, indices);
        return _A(a);
      }`;
    const outputLayout = handler.createTextureLayoutFromShape(unpackedOutputShape, 1, unpackedOutputShape);
    return {inputLayouts: [handler.getOrCreateTextureLayout(inputs[0])], outputLayout, samplers: ['A'], shaderSource};
  }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = [handler.getOrCreateTextureData(inputs[0], programInfo.inputLayouts[0])];
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].tensor.type),
      uniformData: {}
    };
  }
  getPositionalFunction(handler: WebGLInferenceHandler, inputShape: number[], name?: string): GlslPositionalFunction {
    const outputShape = this.getOutputShape([inputShape]);
    if (!name) {
      name = 'perm';
    }
    return {
      name,
      body: this.getPermFunctionBody(name, this.getAdjustedPerm(inputShape), outputShape.length),
      type: FunctionType.Positional,
      inputShape,
      outputShape
    };
  }
  protected getAdjustedPerm(inputShape: readonly number[]): number[] {
    let perm = this.perm;
    if (perm && perm.length !== inputShape.length) {
      perm = [...(inputShape.keys())].reverse();
    }
    return perm;
  }
  protected getPermFunctionBody(name: string, perm: number[], rank: number): string {
    const reverseFunc = [];
    reverseFunc.push(`void ${name}(out int a[${rank}], int src[${rank}]) {`);
    for (let i = 0; i < rank; ++i) {
      reverseFunc.push(`\ta[${perm[i]}]=src[${i}];`);
    }
    reverseFunc.push('\t}');
    return reverseFunc.join('\n');
  }
}
