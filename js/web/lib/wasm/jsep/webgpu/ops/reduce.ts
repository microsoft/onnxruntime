// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';

import {ShaderHelper} from './common';


const validateInputs = (inputs: readonly TensorView[]): void => {
  // TODO: support Reduce* operators with 2 inputs.
  if (!inputs || inputs.length !== 1) {
    throw new Error('Reduce op requires 1 input.');
  }

  if (inputs[0].dataType != DataType.float) {
    throw new Error('Invalid input type.');
  }
};

export interface ReduceAttributes extends AttributeWithCacheKey {
  axes: number[];
  keepDims: number;
}

type ReduceOp = (inputs: readonly TensorView[], axes: number[]) => string[];

const createReduceProgramInfo =
    (metadata: ProgramMetadata, inputs: readonly TensorView[], attributes: ReduceAttributes, reduceOp: ReduceOp):
        ProgramInfo => {
          const outputShape: number[] = [];
          const iRank = inputs[0].dims.length || 1;

          const idxCopy: string[] = [];  // copy output indexes to input indexes

          const axes = ShapeUtil.normalizeAxes(attributes.axes, inputs[0].dims.length);
          const ops = reduceOp(inputs, axes);
          let reduceOps = ops[1];

          for (let k = 0; k < inputs[0].dims.length; k++) {
            // if this axis is reduced
            if (axes.indexOf(k) >= 0 || axes.length === 0) {
              if (attributes.keepDims) {
                outputShape.push(1);
              }  // else { remove the axis from outputShape; }

              // loop over the d-th axis
              reduceOps = `
            for(int j${k} = 0; j${k} < ${inputs[0].dims[k]}; j${k}++) {
            inputIdx[${k}] = j${k};
            ${reduceOps}
            }`;
            } else {
              idxCopy.push(`inputIdx[${k}] = outputIdx[${outputShape.length}];`);
              outputShape.push(inputs[0].dims[k]);
            }
          }

          const oRank = outputShape.length || 1;
          const outputSize = ShapeUtil.size(outputShape);

          const getShaderSource = (shaderHelper: ShaderHelper) => `
    float process(int outputIdx[${oRank}]) {
      float value;                 // final result
      int inputIdx[${iRank}];      // addressing input data
      ${idxCopy.join('\n')}
      ${ops[0]}       // init ops for reduce max/min
      ${reduceOps}
      ${ops[2]}       // final computation for reduce mean
      return value;
    }`;

          return {
            ...metadata,
            getShaderSource,
            outputs: [{dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default}],
            dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
          };
        };

const createReduceProgramInfoLoader =
    (inputs: readonly TensorView[], name: string, attributes: ReduceAttributes, reduceOp: ReduceOp):
        ProgramInfoLoader => {
          const metadata: ProgramMetadata = {name, inputTypes: [GpuDataType.default]};
          return {...metadata, get: () => createReduceProgramInfo(metadata, inputs, attributes, reduceOp)};
        };

export const reduceSum = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (inputs: TensorView[], axes: number[]): string[] => {
    return ['value = 0.0;', 'value += _A(inputIdx);', ''];
  };
  context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceSum', attributes, reduceOp))
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

    return ['value = 0.0;', 'value += _A(inputIdx);', `value /= ${size}.;`];  // ensure real number with `.`
  };
  context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceMean', attributes, reduceOp))
};

export const reduceMax = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (inputs: TensorView[], axes: number[]): string[] => {
    const idxZero = [];
    for (let k = 0; k < inputs[0].dims.length; k++) {
      if (axes.indexOf(k) >= 0 || axes.length === 0) {
        idxZero.push(`inputIdx[${k}] = 0;`);  // first element
      }
    }

    return [`${idxZero.join('\n')}\nvalue = _A(inputIdx);`, 'value = max(value, _A(inputIdx));', ''];
  };
  context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceMax', attributes, reduceOp))
};

export const reduceMin = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (inputs: TensorView[], axes: number[]): string[] => {
    const idxZero = [];
    for (let k = 0; k < inputs[0].dims.length; k++) {
      if (axes.indexOf(k) >= 0 || axes.length === 0) {
        idxZero.push(`inputIdx[${k}] = 0;`);  // first element
      }
    }

    return [`${idxZero.join('\n')}\nvalue = _A(inputIdx);`, 'value = min(value, _A(inputIdx));', ''];
  };
  context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceMin', attributes, reduceOp))
};

export const reduceProd = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (): string[] => {
    return ['value = 1.0;', 'value *= _A(inputIdx);', ''];
  };
  context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceProd', attributes, reduceOp))
};

export const reduceLogSum = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (): string[] => {
    return ['value = 0.0;', 'value += _A(inputIdx);', 'value = log(value);'];
  };
  context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceLogSum', attributes, reduceOp))
};

export const reduceLogSumSquare = (context: ComputeContext, attributes: ReduceAttributes): void => {
  validateInputs(context.inputs);
  const reduceOp: ReduceOp = (): string[] => {
    return ['float t; value = 0.0;', 't = _A(inputIdx); value += t * t;', ''];
  };
  context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceLogSumSquare', attributes, reduceOp))
};

export const parseReduceAttributes = (attributes: Record<string, unknown>): ReduceAttributes =>
    createAttributeWithCacheKey(attributes as Omit<ReduceAttributes, keyof AttributeWithCacheKey>);
