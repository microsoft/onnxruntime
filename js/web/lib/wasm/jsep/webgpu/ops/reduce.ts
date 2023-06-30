// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';

import {createIndicesHelper, ShaderHelper} from './common';

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length === 0 || inputs.length > 2) {
    throw new Error('Reduce op requires 1 or 2 inputs.');
  }

  if (inputs.length === 2 && inputs[1].dims.length !== 1) {
    throw new Error('Invalid axes input dims.');
  }

  if (inputs[0].dataType !== DataType.float) {
    throw new Error('Invalid input type.');
  }
};

export interface ReduceAttributes extends AttributeWithCacheKey {
  keepDims: boolean;
  noopWithEmptyAxes: boolean;
  axes: number[];
}

type ReduceOp = (inputs: readonly TensorView[], axes: number[]) => string[];
const noOp: ReduceOp = (): string[] => ['', '', 'value = _A[inputIdx];', ''];
const createReduceProgramInfo =
    (metadata: ProgramMetadata, inputs: readonly TensorView[], attributes: ReduceAttributes,
     reduceOp: ReduceOp): ProgramInfo => {
      const outputShape: number[] = [];
      const inputShape = inputs[0].dims;

      const idxCopy: string[] = [];  // copy output indexes to input indexes

      const axes = ShapeUtil.normalizeAxes(attributes.axes, inputs[0].dims.length);
      const outputDimsLength = inputs[0].dims.length - (attributes.keepDims ? 0 : axes.length);
      const ops = reduceOp(inputs, axes);
      const inputIndicesHelper = createIndicesHelper('input', inputShape);
      const initInputIdx = (ops[1] === '') ? '' : `let inputIdx = ${inputIndicesHelper.i2oExpression('inputIndices')};`;
      let reduceOps = `
          let inputIdx = ${inputIndicesHelper.i2oExpression('inputIndices')};
          ${ops[2]};`;
      const reduceOnAllAxes = !attributes.noopWithEmptyAxes && attributes.axes.length === 0;
      for (let k = 0; k < inputs[0].dims.length; k++) {
        // if this axis is reduced
        if (reduceOnAllAxes || axes.indexOf(k) >= 0) {
          if (attributes.keepDims) {
            outputShape.push(1);
          }  // else { remove the axis from outputShape; }

          // loop over the d-th axis
          reduceOps = `for(var j${k}: u32 = 0; j${k} < ${inputs[0].dims[k]}; j${k}++) {
                            inputIndices[${k}] = j${k};
                            ${reduceOps}
                          }`;
        } else {
          if (outputDimsLength > 1) {
            idxCopy.push(`inputIndices[${k}] = outputIndices[${outputShape.length}];`);
          } else {
            idxCopy.push(`inputIndices[${k}] = outputIndices;`);
          }
          outputShape.push(inputs[0].dims[k]);
        }
      }

      const outputIndicesHelper = createIndicesHelper('output', outputShape);
      const outputSize = ShapeUtil.size(outputShape);
      const dataType = 'f32';

      const getShaderSource = (shaderHelper: ShaderHelper) => `
          @group(0) @binding(0) var<storage, read> _A : array<${dataType}>;
          @group(0) @binding(1) var<storage, read_write> output : array<${dataType}>;

          ${outputIndicesHelper.o2iImpl}
          ${inputIndicesHelper.i2oImpl}

          ${shaderHelper.mainStart()}
          ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
          ${inputIndicesHelper.indicesVariableDeclaration('inputIndices')}
          ${outputIndicesHelper.indicesVariableDeclaration('outputIndices')}
          ${outputIndicesHelper.o2iCall('global_idx', 'outputIndices')}

          var value = ${dataType}(0);

          ${idxCopy.join('\n')}
          ${ops[0]}       // init ops for reduce max/min
          ${initInputIdx}
          ${ops[1]}
          ${reduceOps}
          ${ops[3]}       // final computation for reduce mean
          output[global_idx] = value;
        }`;

      return {
        ...metadata,
        getShaderSource,
        outputs: [{dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default}],
        dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
      };
    };

const createReduceAttributesFromInputs =
    (inputs: readonly TensorView[], attributes: ReduceAttributes): ReduceAttributes => {
      const axes: number[] = [];
      if (inputs[1].dims[0] > 0) {
        inputs[1].getBigInt64Array().forEach(v => axes.push(Number(v)));
      }
      return createAttributeWithCacheKey(
          {axes, keepDims: attributes.keepDims, noopWithEmptyAxes: attributes.noopWithEmptyAxes});
    };

const createReduceProgramInfoLoader =
    (inputs: readonly TensorView[], name: string, attributes: ReduceAttributes, reduceOp: ReduceOp):
        ProgramInfoLoader => {
          const updatedAttributes: ReduceAttributes =
              inputs.length === 1 ? attributes : createReduceAttributesFromInputs(inputs, attributes);
          const metadata:
              ProgramMetadata = {name, inputTypes: [GpuDataType.default], cacheHint: updatedAttributes.cacheKey};
          return {
            ...metadata,
            get: () => createReduceProgramInfo(
                metadata, [inputs[0]], updatedAttributes,
                updatedAttributes.noopWithEmptyAxes && updatedAttributes.axes.length === 0 ? noOp : reduceOp)
          };
        };

export const reduceLogSum = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (): string[] => ['value = 0.0;', '', 'value += _A[inputIdx];', 'value = log(value);'];
  context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceLogSum', attributes, reduceOp), {inputs: [0]});
};

export const reduceL1 = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (): string[] => ['value = 0.0;', '', 'value += abs(_A[inputIdx]);', ''];
  context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceL1', attributes, reduceOp), {inputs: [0]});
};

export const reduceL2 = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = ():
      string[] => ['var t = f32(0); value = 0.0;', '', 't = _A[inputIdx]; value += (t * t);', 'value = sqrt(value);'];
  context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceL2', attributes, reduceOp), {inputs: [0]});
};

export const reduceLogSumExp = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (): string[] => ['value = 0.0;', '', 'value += exp(_A[inputIdx]);', 'value = log(value);'];
  context.compute(
      createReduceProgramInfoLoader(context.inputs, 'ReduceLogSumExp', attributes, reduceOp), {inputs: [0]});
};

export const reduceMax = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (inputs: TensorView[], axes: number[]): string[] => {
    const idxZero = [];
    for (let k = 0; k < inputs[0].dims.length; k++) {
      if (axes.indexOf(k) >= 0 || axes.length === 0) {
        idxZero.push(`inputIndices[${k}] = 0;`);  // first element
      }
    }

    return [`${idxZero.join('\n')}`, 'value = _A[inputIdx];', 'value = max(value, _A[inputIdx]);', ''];
  };
  context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceMax', attributes, reduceOp), {inputs: [0]});
};

export const reduceMean = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (inputs: TensorView[], axes: number[]): string[] => {
    let size = 1.0;
    for (let k = 0; k < inputs[0].dims.length; k++) {
      if (axes.indexOf(k) >= 0 || axes.length === 0) {
        size *= inputs[0].dims[k];
      }
    }

    return ['value = 0.0;', '', 'value += _A[inputIdx];', `value = value / ${size}.;`];  // ensure real number with `.`
  };
  context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceMean', attributes, reduceOp), {inputs: [0]});
};

export const reduceMin = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (inputs: TensorView[], axes: number[]): string[] => {
    const idxZero = [];
    for (let k = 0; k < inputs[0].dims.length; k++) {
      if (axes.indexOf(k) >= 0 || axes.length === 0) {
        idxZero.push(`inputIndices[${k}] = 0;`);  // first element
      }
    }

    return [`${idxZero.join('\n')}`, 'value = _A[inputIdx];', 'value = min(value, _A[inputIdx]);', ''];
  };
  context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceMin', attributes, reduceOp), {inputs: [0]});
};

export const reduceProd = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (): string[] => ['value = 1.0;', '', 'value *= _A[inputIdx];', ''];
  context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceProd', attributes, reduceOp), {inputs: [0]});
};

export const reduceSum = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (): string[] => ['value = 0.0;', '', 'value += _A[inputIdx];', ''];
  context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceSum', attributes, reduceOp), {inputs: [0]});
};

export const reduceSumSquare = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp =
      (): string[] => ['var t = f32(0); value = 0.0;', '', 't = _A[inputIdx]; value += t * t;', ''];
  context.compute(
      createReduceProgramInfoLoader(context.inputs, 'ReduceSumSquare', attributes, reduceOp), {inputs: [0]});
};

export const parseReduceAttributes = (attributes: Record<string, unknown>): ReduceAttributes =>
    createAttributeWithCacheKey(attributes as Omit<ReduceAttributes, keyof AttributeWithCacheKey>);
