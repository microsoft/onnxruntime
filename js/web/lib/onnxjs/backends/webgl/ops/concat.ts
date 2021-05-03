// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Attribute} from '../../../attribute';
import {Concat} from '../../../ops/concat';
import {Tensor} from '../../../tensor';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, WebGLOperator} from '../types';

import {WebGLPackedConcat} from './concat-packed';

// We provide a wrapper class so that the kernel can switch between packed and unpacked depending on the inputs on the
// fly.
export class WebGLConcat extends Concat implements WebGLOperator {
  unpackedImpl: WebGLUnpackedConcat;
  packedImpl: WebGLPackedConcat;
  constructor() {
    super();
    this.unpackedImpl = new WebGLUnpackedConcat();
    this.packedImpl = new WebGLPackedConcat();
  }

  // No need to call super since this class only serves as a wrapper.
  initialize(attributes: Attribute): void {
    this.unpackedImpl.initialize(attributes);
    this.packedImpl.initialize(attributes);
  }

  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    if (inferenceHandler.session.pack && inputs[0].dims.length > 1) {
      return inferenceHandler.run(this.packedImpl, inputs);
    } else {
      return inferenceHandler.run(this.unpackedImpl, inputs);
    }
  }

  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    if (handler.session.pack && inputs[0].dims.length > 1) {
      return this.packedImpl.createProgramInfo(handler, inputs);
    } else {
      return this.unpackedImpl.createProgramInfo(handler, inputs);
    }
  }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    if (handler.session.pack && inputs[0].dims.length > 1) {
      return this.packedImpl.createRunData(handler, programInfo, inputs);
    } else {
      return this.unpackedImpl.createRunData(handler, programInfo, inputs);
    }
  }
}
export class WebGLUnpackedConcat extends Concat implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const inputShape = inputs[0].dims.slice();
    if (this.axis >= inputShape.length || this.axis < (-1 * inputShape.length)) {
      throw new Error('axis specified for concat doesn\'t match input dimensionality');
    }
    if (this.axis < 0) {
      this.axis = inputShape.length + this.axis;
    }
    // ensure all of the non-concatenated axes match each other
    // calculate the shape of the output tensor while we do that
    const outputShape = inputShape.slice(0);
    for (let i = 1; i < inputs.length; i++) {
      const dataNShape = inputs[i].dims.slice();
      for (let axisIndex = 0; axisIndex < inputShape.length; axisIndex++) {
        // add to the placeholder for computing output shape
        if (axisIndex === this.axis) {
          outputShape[this.axis] += dataNShape[axisIndex];
        }
        // ensure all non-cancatenated axes match each other
        else if (inputShape[axisIndex] !== dataNShape[axisIndex]) {
          throw new Error('non concat dimensions must match');
        }
      }
    }

    const rank = outputShape.length;

    let getTextureIndexWhereDataResidesMethod = '';
    // in most cases linear search is sufficient, as in most scenarios, only 2 tensors are concatenated
    if (inputs.length < 5) {
      getTextureIndexWhereDataResidesMethod = this.getTextureIndexWhereDataResidesLinearSearch(inputs.length);
    } else {
      getTextureIndexWhereDataResidesMethod = this.getTextureIndexWhereDataResidesBinarySearch(inputs.length);
    }

    const fetchDataFromCorrectTextureMethod = this.fetchDataFromCorrectTextureMethod(inputs.length, rank);
    const getValueFromArrayIndexMethod = this.getValueFromArrayIndexMethod(inputs.length);
    const samplers = inputs.map((v, i) => `X${i}`);
    const shaderSource = `
      ${fetchDataFromCorrectTextureMethod}
      ${getValueFromArrayIndexMethod}
      ${getTextureIndexWhereDataResidesMethod}
      float process(int indices[${rank}]) {
        int textureIndex = getTextureWhereDataResides (indices[${this.axis}]);

        if(textureIndex != 0) {
          indices[${this.axis}] = indices[${
        this.axis}] - int(getValueFromArrayIndex(sizeInConcatAxis, textureIndex-int(1)));
        }

        return fetchDataFromCorrectTexture(textureIndex, indices);
      }`;
    return {
      inputLayouts: inputs.map(t => handler.getOrCreateTextureLayout(t)),
      outputLayout: handler.createTextureLayoutFromShape(outputShape),
      samplers,
      variables: [{name: 'sizeInConcatAxis', type: 'int', arrayLength: inputs.length}],
      shaderSource,
    };
  }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = inputs.map((t, i) => handler.getOrCreateTextureData(t, programInfo.inputLayouts[i]));
    const sizeInConcatAxis = new Array<number>(programInfo.inputLayouts.length);
    let previousSum = 0;
    for (let i = 0; i < programInfo.inputLayouts.length; ++i) {
      previousSum += programInfo.inputLayouts[i].shape[this.axis];
      sizeInConcatAxis[i] = previousSum;
    }
    const uniformData = {sizeInConcatAxis};
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].tensor.type),
      uniformData
    };
  }
  private getTextureIndexWhereDataResidesLinearSearch(numberOfTensors: number): string {
    return `int getTextureWhereDataResides(int index) {
      for(int i=0; i<${numberOfTensors}; i++) {
          if(index < int(sizeInConcatAxis[i])){
              return i;
          }
        }
      }`;
  }

  // TODO: Implement BinarySearch in GLSL
  private getTextureIndexWhereDataResidesBinarySearch(numberOfTensors: number): string {
    return this.getTextureIndexWhereDataResidesLinearSearch(numberOfTensors);
  }

  private fetchDataFromCorrectTextureMethod(numberOfTensors: number, tensorRank: number) {
    const codeLines: string[] = [`float fetchDataFromCorrectTexture(int textureIndex, int indices[${tensorRank}]) {`];
    for (let i = 0; i < numberOfTensors; ++i) {
      if (i === 0) {
        codeLines.push(
            '\t' +
            `if (textureIndex == ${i}) { return _X${i}(indices); }`);
      } else if (i === numberOfTensors - 1) {
        codeLines.push(
            '\t' +
            `else { return _X${i}(indices); }`);
      } else {
        codeLines.push(
            '\t' +
            `else if (textureIndex == ${i}) { return _X${i}(indices); }`);
      }
    }
    codeLines.push(
        '\t' +
        '}');
    return codeLines.join('\n');
  }

  private getValueFromArrayIndexMethod(arrayRank: number): string {
    const codeLines: string[] = [`int getValueFromArrayIndex(int arr[${arrayRank}], int index) {`];
    for (let i = 0; i < arrayRank; ++i) {
      if (i === 0) {
        codeLines.push(
            '\t' +
            `if (index == ${i}) { return arr[${i}]; }`);
      } else if (i === arrayRank - 1) {
        codeLines.push(
            '\t' +
            `else { return arr[${i}]; }`);
      } else {
        codeLines.push(
            '\t' +
            `else if (index == ${i}) { return arr[${i}]; }`);
      }
    }
    codeLines.push(
        '\t' +
        '}');

    return codeLines.join('\n');
  }
}
