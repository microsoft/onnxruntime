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

/**
 * A helper class for generating WGSL code for manipulating indices and data for a shader's input or output.
 *
 * This class is designed to offer a unified way to generate WGSL code for manipulating indices and data for a shader's
 * input or output.
 *
 * The following is a list of terminologies used in this class:
 * - `offset`: a uint32 value representing the offset of an element in the data buffer.
 * - `indices`: an abstraction of a multi-dimensional array's indices representing the data's shape.
 * - `indicesType`: a type representing the indices.
 * - `value`: a value of a data element.
 * - `valueType`: a type representing the data.
 *
 * Users are expected to create an instance of this class for each shader's input or output, and use the instance to
 * generate WGSL code for manipulating indices and data. The following 2 exported functions are for users to call to
 * create an instance of an indices helper:
 * - `inputVariable()`: create an indices helper instance for an input.
 * - `outputVariable()`: create an indices helper instance for an output.
 *
 * An indices helper instance contains helper functions for the following operations:
 * - access readonly basic information, including: `name`(the name of the input or output), `usage`(whether it's an
 * input or an output), `isVec4`(whether the data is a vec4), `dataType` and `indicesType`.
 * - generate WGSL code for getting indices from offset. Use `offsetToIndices()` for WGSL code snippet to calculate
 * incides from offset, and use `indicesToOffset()` for WGSL code snippet to calculate offset from indices.
 * - to manipulate an instance of indices, use `setIndices()` and `getIndices()` to set and get the indices on an
 * indices variable.
 * - to manipulate data, use `set()` and `get()` to set and get data at the given indices, or use `setByOffset()` and
 * `getByOffset()` to set and get data at the given offset.
 */
export interface IndicesHelper {
  /**
   * WGSL code of function implementation for offset-to-indices.
   *
   * This string should be put in global scope of a shader, if `offsetToIndices()` is used.
   */
  readonly offsetToIndicesImplementation: string;

  /**
   * WGSL code of a statement for getting indices from offset.
   *
   * This string should be used as a statement in a shader.
   *
   * @param varOffset - a u32 expression representing the offset.
   * @param varIndices - a variable name for the indices. The variable should be declared as `indicesType` before this
   *     statement.
   */
  readonly offsetToIndices: (varOffset: string, varIndices: string) => string;

  /**
   * WGSL code of function implementation for indices-to-offset
   *
   * This string should be put in global scope of a shader, if `indicesToOffset()` is used.
   */
  readonly indicesToOffsetImplementation: string;

  /**
   * WGSL code of an `u32` expression for getting offset from indices.
   *
   * @param varIndices - a `indicesType` expression representing the indices.
   * @param isPtr - whether the variable is a pointer. default is false.
   */
  readonly indicesToOffset: (varIndices: string, isPtr?: boolean) => string;

  /**
   * WGSL code of indices variable declaration
   *
   * @param v - variable name.
   * @param init - initial value.
   */
  readonly indicesVariableDeclaration: (v: string, init?: string[]) => string;

  /**
   * WGSL code of a statement for setting indices.
   *
   * @param varIndices - a variable name for the indices.
   * @param idx - the index of the indices to set. can be a number or a string (WGSL `u32` expression).
   * @param value - the value to set. can be a number or a string (WGSL `u32` expression).
   */
  readonly indicesSet: (varIndices: string, idx: number|string, value: number|string) => void;

  /**
   * WGSL code of an `u32` expression for getting indices.
   *
   * @param varIndices - a variable name for the indices.
   * @param idx - the index of the indices to get. can be a number or a string (WGSL `u32` expression).
   */
  readonly indicesGet: (varIndices: string, idx: number|string) => string;

  /**
   * WGSL code of function implementation for set and set by offset.
   */
  readonly setImplementation: string;

  /**
   * WGSL code for a statement for setting data at the given indices.
   *
   * @param indices - an array of numbers or strings (WGSL `u32` expression) representing the indices, or a string of a
   *     indices variable name.
   * @param value - the value to set. should be a WGSL expression.
   */
  readonly set: (indices: ReadonlyArray<number|string>|string, value: string) => string;

  /**
   * WGSL code for a statement for setting data at the given offset.
   *
   * @param offset - a number or a string (WGSL `u32` expression) representing the offset.
   * @param value - the value to set. should be a WGSL expression.
   */
  readonly setByOffset: (offset: number|string, value: string) => string;

  /**
   * WGSL code of function implementation for get and get by offset.
   */
  readonly getImplementation: string;

  /**
   * WGSL code for an expression for getting data at the given indices.
   *
   * @param indices - an array of numbers or strings (WGSL `u32` expression) representing the indices, or a string of a
   *    indices variable name.
   */
  readonly get: (indices: ReadonlyArray<number|string>|string) => string;

  /**
   * WGSL code for an expression for getting data at the given offset.
   *
   * @param offset - a number or a string (WGSL `u32` expression) representing the offset.
   */
  readonly getByOffset: (offset: number|string) => string;

  /**
   * data type of indices
   */
  readonly indicesType: string;

  /**
   * data type of data
   */
  readonly dataType: string;

  /**
   * name of the data variable
   */
  readonly name: string;

  /**
   * whether the input or output is a vec4
   */
  readonly isVec4: boolean;

  /**
   * whether the helper is for an input or an output.
   */
  readonly usage: 'input'|'output';
}

/**
 * A helper function to get a IndicesHelper for a given input or output.
 *
 * @example
 * const getShaderSource = (shaderHelper: ShaderHelper) => {
 *   const X = shaderHelper.createIndicesHelper('X', XShape);
 *   const Y = shaderHelper.createIndicesHelper('Y', YShape);
 *
 *   return `
 *   ...
 *
 *   ${shaderHelper.mainStart()}
 *     ...
 *   }
 *   `;
 *
 * @param name - the name of the input or output.
 * @param dataType - the tensor type of the input or output.
 * @param shape - the tensor shape of the input or output.
 * @param isInput - whether the helper is for an input or an output.
 * @param isVec4 - whether the helper is for a vec4 input or output.
 */
const createIndicesHelper =
    (name: string, dataType: string, shape: readonly number[], isInput: boolean, isVec4: boolean): IndicesHelper => {
      const rank = shape.length;
      const indicesType = rank < 2 ? 'u32' : `array<u32, ${rank}>`;
      const mappedDataType = isVec4 ? `vec4<${dataType}>` : dataType;

      const normalizeDim = (dim: number|string): string => typeof dim === 'string' ? dim : `${dim}u`;

      const strides = ShapeUtil.computeStrides(shape);
      let o2iSnippet = '';
      for (let i = 0; i < rank - 1; i++) {
        o2iSnippet += `
    let dim${i} = current / ${strides[i]}u;
    let rest${i} = current % ${strides[i]}u;
    (*indices)[${i}] = dim${i};
    current = rest${i};
    `;
      }
      o2iSnippet += `(*indices)[${rank - 1}] = current;`;

      const offsetToIndicesImplementation = rank < 2 ? '' : `
  fn ih_o2i_${name}(offset: u32, indices: ptr<function, ${indicesType}>) {
    var current = offset;
    ${o2iSnippet}
  }`;

      const offsetToIndices = (varOffset: string, varIndices: string) =>
          rank < 2 ? `${varIndices}=${varOffset};` : `ih_o2i_${name}(${varOffset}, &${varIndices});`;

      const offsets: string[] = [];
      if (rank === 0) {
        offsets.push('0u');
      } else if (rank < 2) {
        offsets.push('(*indices)');
      } else {
        for (let i = rank - 1; i >= 0; i--) {
          offsets.push(`${strides[i]}u * ((*indices)[${i}])`);
        }
      }

      const indicesToOffsetImplementation = rank < 2 ? '' : `
  fn ih_i2o_${name}(indices: ptr<function, ${indicesType}>) -> u32 {
    return ${offsets.join('+')};
  }`;

      const indicesToOffset = (varIndices: string, isPtr?: boolean) =>
          rank < 2 ? `(${isPtr ? '*' : ''}${varIndices})` : `ih_i2o_${name}(${isPtr ? '' : '&'}${varIndices})`;

      const indicesVariableDeclaration = (v: string, init?: string[]) =>
          `var ${v}:${indicesType}${init ? `=${indicesType}(${init.join(',')})` : ''};`;

      const indicesGet = (varIndices: string, idx: number|string) => {
        if (rank < 2) {
          return `${varIndices}`;
        } else {
          return `${varIndices}[${idx}]`;
        }
      };

      const indicesSet = (varIndices: string, idx: number|string, value: string) => {
        if (rank < 2) {
          return `${varIndices}=${value};`;
        } else {
          return `${varIndices}[${idx}]=${value};`;
        }
      };

      const setByOffset = (offset: number|string, value: string) => `${name}[${offset}]=${value};`;

      const getByOffset = (offset: number|string) => `${name}[${offset}]`;


      const getImplementation = rank < 2 ? '' : (() => {
        const parameters = new Array(rank).map((_, i) => `d${i}: u32`).join(', ');
        const parametersAssignment = new Array(rank).map((_, i) => indicesSet('idx', i, `d${i}`)).join('\n');
        return `
  fn get_${name}(${parameters}) -> ${mappedDataType} {
    ${indicesVariableDeclaration('idx')}
    ${parametersAssignment}
    return get_${name}ByIndices(&idx);
  }
  fn get_${name}ByIndices(indices: ptr<function, ${indicesType}>) -> ${mappedDataType} {
    return ${name}[ih_i2o_${name}(&indices)];
  }`;
      })();

      const get = (indices: ReadonlyArray<number|string>|string) => {
        let normalizedIndices: string;
        let funcName: string;
        if (typeof indices === 'string') {
          normalizedIndices = indices;
          funcName = `get_${name}ByIndices`;
        } else {
          normalizedIndices = indices.map(normalizeDim).join(',');
          funcName = `get_${name}`;
        }

        if (rank === 0) {
          return getByOffset('0u');
        } else if (rank === 1) {
          return getByOffset(normalizedIndices[0]);
        } else {
          return `${funcName}(${normalizedIndices})`;
        }
      };

      const setImplementation = rank < 2 ? '' : (() => {
        const parameters = new Array(rank + 1).map((_, i) => `d${i}: u32`).join(', ');
        const parametersAssignment = new Array(rank).map((_, i) => indicesSet('idx', i, `d${i}`)).join('\n');
        return `
  fn set_${name}(${parameters}, value: ${mappedDataType}) {
    ${indicesVariableDeclaration('idx')}
    ${parametersAssignment}
    set_${name}ByIndices(&idx, value);
  }
  fn set_${name}ByIndices(indices: ptr<function, ${indicesType}>, value: ${mappedDataType}) {
    ${setByOffset(`ih_i2o_${name}(&indices)`, 'value')}
  }`;
      })();

      const set = (indices: ReadonlyArray<number|string>|string, value: string) => {
        let normalizedIndices: string;
        let funcName: string;
        if (typeof indices === 'string') {
          normalizedIndices = indices;
          funcName = `set_${name}ByIndices`;
        } else {
          normalizedIndices = indices.map(normalizeDim).join(',');
          funcName = `set_${name}`;
        }

        if (rank === 0) {
          return setByOffset('0u', value);
        } else if (rank === 1) {
          return setByOffset(normalizedIndices[0], value);
        } else {
          return `${funcName}(${normalizedIndices}, ${value})`;
        }
      };

      return {
        offsetToIndicesImplementation,
        offsetToIndices,
        indicesToOffsetImplementation,
        indicesToOffset,
        indicesVariableDeclaration,
        indicesType,
        dataType,
        isVec4,
        usage: isInput ? 'input' : 'output',
        name,
        indicesGet,
        indicesSet,
        set,
        setImplementation,
        setByOffset,
        get,
        getImplementation,
        getByOffset
      };
    };

/**
 * Create a IndicesHelper for an input.
 *
 * @param name - the name of the input.
 * @param type - the tensor type of the input.
 * @param shape - the tensor shape of the input.
 * @returns an IndicesHelper for the input.
 */
export const inputVariable = (name: string, type: string, shape: readonly number[], isVec4 = false): IndicesHelper =>
    createIndicesHelper(name, type, shape, true, isVec4);

/**
 * Create a IndicesHelper for an output.
 *
 * @param name - the name of the output.
 * @param type - the tensor type of the output.
 * @param shape - the tensor shape of the output.
 * @returns an IndicesHelper for the output.
 */
export const outputVariable = (name: string, type: string, shape: readonly number[], isVec4 = false): IndicesHelper =>
    createIndicesHelper(name, type, shape, false, isVec4);

/**
 * A ShaderHelper is a helper class for generating WGSL code.
 */
export interface ShaderHelper {
  /**
   * A helper function to generate the start of main function in WGSL source code.
   *
   * @example
   * const getShaderSource = (shaderHelper: ShaderHelper) => `
   *  ...
   *
   *  ${shaderHelper.mainStart()}
   *    // your code here inside main() function
   *    ...
   *  }
   * `;
   *
   * @param workgroupSize - an optional workgroup size. default is WORKGROUP_SIZE.
   */
  mainStart(workgroupSize?: number|[number, number, number]): string;

  /**
   * A helper function to generate the code snippet for guarding against out-of-bounds size.
   *
   * @example
   * const getShaderSource = (shaderHelper: ShaderHelper) => `
   *  ...
   *
   *  ${shaderHelper.mainStart()}
   *    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
   *
   *    // your code here inside main() function
   *    ...
   *  }
   * `;
   *
   * @param size - the size of the data to guard against. can be a number or a string (WGSL `u32` expression).
   */
  guardAgainstOutOfBoundsWorkgroupSizes(size: unknown): string;

  /**
   * A helper function to generate the code snippet for declaring an input or output.
   *
   * @param variable - the IndicesHelper for the variable.
   * @param bindingIndex - the index of the variable binding.
   */
  declareVariable(variable: IndicesHelper, bindingIndex: number): string;
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

  declareVariable(variable: IndicesHelper, bindingIndex: number): string {
    const type = variable.isVec4 ? `vec4<${variable.dataType}>` : variable.dataType;
    const access = variable.usage === 'input' ? 'read' : 'read_write';
    return `@group(0) @binding(${bindingIndex}) var<storage, ${access}> ${variable.name}: array<${type}>;`;
  }
}

export const createShaderHelper = (dispatchGroup: [number, number, number]): ShaderHelper =>
    new ShaderHelperImpl(dispatchGroup);
