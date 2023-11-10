// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {ShapeUtil} from '../../util';

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
   *
   * @param isPtr - whether the variable is a pointer. default is false.
   */
  i2oExpression: (varIndices: string, isPtr?: boolean) => string;
  /**
   * WGSL code of indices variable declaration
   *
   * @param v - variable name.
   * @param init - initial value.
   */
  indicesVariableDeclaration: (v: string, init?: string[]) => string;
  /**
   * data type of indices
   */
  iType: string;
}

export const createIndicesHelper = (name: string, shape: readonly number[]): IndicesHelper => {
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
  if (shape.length === 0) {
    offsets.push('0u');
  } else if (shape.length < 2) {
    offsets.push('(*indices)');
  } else {
    for (let i = shape.length - 1; i >= 0; i--) {
      offsets.push(`${strides[i]}u * ((*indices)[${i}])`);
    }
  }

  const i2oImpl = shape.length < 2 ? '' : `
  fn ih_i2o_${name}(indices: ptr<function, ${iType}>) -> u32 {
    return ${offsets.join('+')};
  }`;

  const i2oExpression = (varIndices: string, isPtr?: boolean) =>
      shape.length < 2 ? `(${isPtr ? '*' : ''}${varIndices})` : `ih_i2o_${name}(${isPtr ? '' : '&'}${varIndices})`;

  const indicesVariableDeclaration = (v: string, init?: string[]) =>
      `var ${v}:${iType}${init ? `=${iType}(${init.join(',')})` : ''};`;

  return {o2iImpl, o2iCall, i2oImpl, i2oExpression, indicesVariableDeclaration, iType};
};

/**
 * A ShaderHelper is a helper class for generating WGSL code.
 */
export interface ShaderHelper {
  mainStart(workgroupSize?: number|[number, number, number]): string;
  guardAgainstOutOfBoundsWorkgroupSizes(size: unknown): string;
}

class ShaderHelperImpl implements ShaderHelper {
  constructor(private normalizedDispatchGroup: [number, number, number]) {}
  guardAgainstOutOfBoundsWorkgroupSizes(size: number|string): string {
    // Guard against out-of-bounds work group sizes
    const sizeInCode = typeof size === 'number' ? `${size}u` : size;
    return `if (global_idx >= ${sizeInCode}) { return; }`;
  }
  mainStart(workgroupSize: number|[number, number, number] = WORKGROUP_SIZE) {
    const workgroupSizeX = typeof workgroupSize === 'number' ? workgroupSize : workgroupSize[0];
    const workgroupSizeY = typeof workgroupSize === 'number' ? 1 : workgroupSize[1];
    const workgroupSizeZ = typeof workgroupSize === 'number' ? 1 : workgroupSize[2];

    const is1DimensionDispatch = this.normalizedDispatchGroup[1] === 1 && this.normalizedDispatchGroup[2] === 1;
    const paramList = is1DimensionDispatch ? '@builtin(global_invocation_id) global_id : vec3<u32>' :
                                             `@builtin(local_invocation_index) local_index : u32,
    @builtin(workgroup_id) workgroup_id : vec3<u32>`;
    const globalIdxDefinition = is1DimensionDispatch ?
        'let global_idx = global_id.x;' :
        `let global_idx = (workgroup_id.z * ${this.normalizedDispatchGroup[0] * this.normalizedDispatchGroup[1]}u +
          workgroup_id.y * ${this.normalizedDispatchGroup[0]}u + workgroup_id.x) * ${
            workgroupSizeX * workgroupSizeY * workgroupSizeZ}u + local_index;`;

    return `@compute @workgroup_size(${workgroupSizeX}, ${workgroupSizeY}, ${workgroupSizeZ})
  fn main(${paramList}) {
    ${globalIdxDefinition}
  `;
  }
}

export const createShaderHelper = (dispatchGroup: [number, number, number]): ShaderHelper =>
    new ShaderHelperImpl(dispatchGroup);
