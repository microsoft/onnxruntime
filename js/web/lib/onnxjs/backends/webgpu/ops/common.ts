// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {ShapeUtil} from '../../../util';

/**
 * constant value for a workgroup size.
 *
 * We definitely can do further optimization in future, but for now we use 64.
 *
 * rule of thumb: Use [a workgroup size of] 64 unless you know what GPU you are targeting or that your workload
 *                needs something different.
 *
 * from: https://surma.dev/things/webgpu/
 **/
export const WORKGROUP_SIZE = 64;

export interface IndicesHelper {
  /**
   * WGSL code of function implementation for offset-to-indices
   */
  o2iImpl: string;
  /**
   * WGSL code of function call for offset-to-indices
   */
  o2iCall: (varOffset: string, varIndices: string) => string;
  /**
   * WGSL code of function implementation for indices-to-offset
   */
  i2oImpl: string;
  /**
   * WGSL code of function implementation for indices-to-offset
   */
  i2oExpression: (varIndices: string) => string;
  /**
   * WGSL code of indices variable declaration
   */
  indicesVariableDeclaration: (v: string) => string;
}

export const createIndicesHelper = (name: string, shape: readonly number[]) => {
  const iType = shape.length < 2 ? 'u32' : `array<u32, ${shape.length}>`;

  const strides = ShapeUtil.computeStrides(shape);
  let o2iSnippet = '';
  for (let i = 0; i < shape.length - 1; i++) {
    o2iSnippet += `
    let dim${i} = current / ${strides[i]}u;
    let rest${i} = current % ${strides[i]}u;
    (*indices)[${i}] = dim${i};
    current = rest${i};
    `;
  }
  o2iSnippet += `(*indices)[${shape.length - 1}] = current;`;

  const o2iImpl = shape.length < 2 ? '' : `
  fn ih_o2i_${name}(offset: u32, indices: ptr<function, ${iType}>) {
    var current = offset;
    ${o2iSnippet}
  }`;

  const o2iCall = (varOffset: string, varIndices: string) =>
      shape.length < 2 ? `${varIndices}=${varOffset};` : `ih_o2i_${name}(${varOffset}, &${varIndices});`;

  const offsets: string[] = [];
  for (let i = shape.length - 1; i >= 0; i--) {
    offsets.push(`${strides[i]}u * ((*indices)[${i}])`);
  }

  const i2oImpl = shape.length < 2 ? '' : `
  fn ih_i2o_${name}(indices: ptr<function, ${iType}>) -> u32 {
    return ${offsets.length > 0 ? offsets.join('+') : '0u'}
  }`;

  const i2oExpression = (varIndices: string) => shape.length < 2 ? varIndices : `ih_i2o_${name}(&${varIndices})`;

  const indicesVariableDeclaration = (v: string) => `var ${v}:${iType};`;

  return {o2iImpl, o2iCall, i2oImpl, i2oExpression, indicesVariableDeclaration};
};
