// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
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

interface IndicesHelperTypes {
  /**
   * WGSL type of indices expression
   */
  readonly indices: string;

  /**
   * WGSL type of a value
   */
  readonly value: string;

  /**
   * WGSL type of storage type representing a value
   *
   * This is usually the same to `value`, but for some type (eg. bool), we need to use `u32` as storage type for
   * value type `vec4<bool>`
   */
  readonly storage: string;

  /**
   * tensor type as represented in TensorView
   */
  readonly tensor: number;
}

/**
 * A helper class for generating WGSL code for manipulating indices and data for a shader's input or output.
 *
 * This class is designed to offer a unified way to generate WGSL code for manipulating indices and data for a shader's
 * input or output.
 *
 * The following is a list of terminologies used in this class:
 * - `offset`: a uint32 value representing the offset of an element in the data buffer.
 * - `indices`: an abstraction of a multi-dimensional array's indices representing the data's index on each dimension.
 * - `value`: a value of a data element.
 *
 * Users are expected to create an instance of this class for each shader's input or output, and use the instance to
 * generate WGSL code for manipulating indices and data. The following 2 exported functions are for users to call to
 * create an instance of an indices helper:
 * - `inputVariable()`: create an indices helper instance for an input.
 * - `outputVariable()`: create an indices helper instance for an output.
 *
 * An indices helper instance contains helper functions for the following operations:
 * - access readonly basic information, including: `name`(the name of the input or output), `usage`(whether it's an
 * input or an output) and `shape`(the passed in shape).
 * - `type`: access readonly type information, including: `indices`(the type of indices), `value`(the type of value at
 * runtime), `storage`(the type of value at storage) and `tensor`(the tensor type as represented in TensorView).
 * - generate WGSL code for getting indices from offset. Use `offsetToIndices()` for WGSL code snippet to calculate
 * indices from offset, and use `indicesToOffset()` for WGSL code snippet to calculate offset from indices.
 * - to manipulate an instance of indices, use `setIndices()` and `getIndices()` to set and get the indices on an
 * indices variable.
 * - to manipulate data, use `set()`/`get()` to access data at the given indices from parameter list, use
 * `setByIndices()`/`getByIndices()` to access data at the given indices from an indices variable, and use
 * `setByOffset()`/`getByOffset()` to access data at the given offset.
 * - `impl`: get WGSL code of function implementation for the util functions mentioned above.
 */
export interface IndicesHelper {
  /**
   * get WGSL code of function implementation for the util functions.
   *
   */
  readonly impl: () => string;

  /**
   * get type info
   */
  readonly type: IndicesHelperTypes;

  /**
   * WGSL code of a expression for getting indices from offset.
   *
   * @param varOffset - a u32 expression representing the offset.
   *
   * @returns an `type.indices` expression
   */
  readonly offsetToIndices: (varOffset: string) => string;

  /**
   * WGSL code of an `u32` expression for getting offset from indices.
   *
   * @param varIndices - a `type.indices` expression representing the indices.
   *
   * @returns an `u32` expression
   */
  readonly indicesToOffset: (varIndices: string) => string;

  /**
   * WGSL code of generating an indices literal
   *
   * @param init - initial value.
   */
  readonly indices: (...init: ReadonlyArray<number|string>) => string;

  /**
   * WGSL code of a statement for setting indices.
   *
   * @param varIndices - a variable name for the indices.
   * @param idx - the index of the indices to set. can be a number or a string (WGSL `u32` expression).
   * @param value - the value to set. can be a number or a string (WGSL `u32` expression).
   *
   * @returns a WGSL statement
   */
  readonly indicesSet: (varIndices: string, idx: number|string, value: number|string) => void;

  /**
   * WGSL code of an `u32` expression for getting indices.
   *
   * @param varIndices - a variable name for the indices.
   * @param idx - the index of the indices to get. can be a number or a string (WGSL `u32` expression).
   *
   * @returns an `u32` expression
   */
  readonly indicesGet: (varIndices: string, idx: number|string) => string;

  /**
   * WGSL code for a statement for setting data at the given indices.
   *
   * @param indicesAndValue - an array of numbers or strings (WGSL `u32` expression) representing the indices, followed
   *     by the value to set. This array should have exactly `shape.length + 1` elements.
   */
  readonly set: (...indicesAndValue: ReadonlyArray<number|string>) => string;

  /**
   * WGSL code for a statement for setting data at the given indices variable.
   *
   * @param varIndices - a variable name for the indices.
   * @param value - the value to set. should be a WGSL expression.
   */
  readonly setByIndices: (varIndices: string, value: string) => string;

  /**
   * WGSL code for a statement for setting data at the given offset.
   *
   * @param offset - a number or a string (WGSL `u32` expression) representing the offset.
   * @param value - the value to set. should be a WGSL expression.
   */
  readonly setByOffset: (offset: number|string, value: string) => string;

  /**
   * WGSL code for an expression for getting data at the given indices.
   *
   * @param indices - an array of numbers or strings (WGSL `u32` expression) representing the indices.
   */
  readonly get: (...indices: ReadonlyArray<number|string>) => string;

  /**
   * WGSL code for an expression for getting data at the given indices variable.
   *
   * @param varIndices - a variable name for the indices.
   */
  readonly getByIndices: (varIndices: string) => string;

  /**
   * WGSL code for an expression for getting data at the given offset.
   *
   * @param offset - a number or a string (WGSL `u32` expression) representing the offset.
   */
  readonly getByOffset: (offset: number|string) => string;

  /**
   * name of the data variable
   */
  readonly name: string;

  /**
   * whether the helper is for an input or an output.
   */
  readonly usage: 'input'|'output';

  /**
   * the shape of the input or output.
   */
  readonly shape: readonly number[];
}

const getWgslMappedType = (type: number, components: 1|2|3|4): string|[string, string] => {
  // return type is [ storage type, runtime type ] or a single string for both
  switch (type) {
    // TODO: enable after "shader-f16" WSGL extension release
    // case DataType.float16:
    //   return components > 1 ? `vec${components}<f16>` : 'f16';
    case DataType.float:
      return components > 1 ? `vec${components}<f32>` : 'f32';
    case DataType.int32:
      return components > 1 ? `vec${components}<i32>` : 'i32';
    case DataType.uint32:
      return components > 1 ? `vec${components}<u32>` : 'u32';
    case DataType.int64:
      if (components > 1) {
        throw new Error('currently not supported vecX of uint64 yet');
      }
      return ['vec2<u32>', 'i32'];
    case DataType.uint64:
      if (components > 1) {
        throw new Error('currently not supported vecX of uint64 yet');
      }
      return ['vec2<u32>', 'u32'];
    case DataType.bool:
      if (components !== 4) {
        throw new Error('bool must be vec4');
      }
      return ['u32', 'vec4<bool>'];

    default:
      throw new Error(`Unknown data type: ${type}`);
  }
};

export const tensorTypeToWsglStorageType = (type: DataType, components: 1|2|3|4 = 1) => {
  const mappedType = getWgslMappedType(type, components);
  return typeof mappedType === 'string' ? mappedType : mappedType[0];
};

/**
 * A helper function to get a IndicesHelper for a given input or output.
 *
 * @param name - the name of the input or output.
 * @param tensorType - the tensor type of the input or output.
 * @param shape - the tensor shape of the input or output.
 * @param isInput - whether the helper is for an input or an output.
 * @param components - indicates the number of components of each element. 1 for scalar, 2 for vec2, 3 for vec3, 4 for
 *    vec4.
 */
const createIndicesHelper =
    (name: string, tensorType: number, shape: readonly number[], isInput: boolean,
     components: 1|2|3|4): IndicesHelper => {
      const rank = shape.length;
      const indicesType = rank < 2 ? 'u32' : rank <= 4 ? `vec${rank}<u32>` : `array<u32, ${rank}>`;
      const mappedType = getWgslMappedType(tensorType, components);
      const valueType = typeof mappedType === 'string' ? mappedType : mappedType[1];
      const storageType = typeof mappedType === 'string' ? mappedType : mappedType[0];
      const type = {indices: indicesType, value: valueType, storage: storageType, tensor: tensorType};

      const normalizeDim = (dim: number|string): string => typeof dim === 'string' ? dim : `${dim}u`;

      const implementationUsed = {
        offsetToIndices: false,
        indicesToOffset: false,
        set: false,
        setByIndices: false,
        get: false,
        getByIndices: false,
      };

      const strides = ShapeUtil.computeStrides(shape);
      let o2iSnippet = '';
      for (let i = 0; i < rank - 1; i++) {
        o2iSnippet += `
    let dim${i} = current / ${strides[i]}u;
    let rest${i} = current % ${strides[i]}u;
    indices[${i}] = dim${i};
    current = rest${i};
    `;
      }
      o2iSnippet += `indices[${rank - 1}] = current;`;

      const offsetToIndicesImplementation = rank < 2 ? '' : `
  fn o2i_${name}(offset: u32) -> ${type.indices} {
    var indices: ${type.indices};
    var current = offset;
    ${o2iSnippet}
    return indices;
  }`;

      const offsetToIndices = (varOffset: string) => {
        implementationUsed.offsetToIndices = true;
        return rank < 2 ? varOffset : `o2i_${name}(${varOffset})`;
      };

      const offsets: string[] = [];
      if (rank >= 2) {
        for (let i = rank - 1; i >= 0; i--) {
          offsets.push(`${strides[i]}u * (indices[${i}])`);
        }
      }

      const indicesToOffsetImplementation = rank < 2 ? '' : `
  fn i2o_${name}(indices: ${type.indices}) -> u32 {
    return ${offsets.join('+')};
  }`;

      const indicesToOffset = (varIndices: string) => {
        implementationUsed.indicesToOffset = true;
        return rank < 2 ? varIndices : `i2o_${name}(${varIndices})`;
      };

      const indices = (...init: ReadonlyArray<number|string>) =>
          rank === 0 ? '0u' : `${type.indices}(${init.map(normalizeDim).join(',')})`;

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

      const setByOffset = (offset: number|string, value: string) => (() => {
        if (type.storage === type.value) {
          return `${name}[${offset}]=${value};`;
        } else if (type.storage === 'vec2<u32>' && type.value === 'i32') {
          // int64, components === 1
          return `${name}[${offset}]=vec2<u32>(u32(${value}), select(0u, 0xFFFFFFFFu, ${value} < 0));`;
        } else if (type.storage === 'vec2<u32>' && type.value === 'u32') {
          // uint64, components === 1
          return `${name}[${offset}]=vec2<u32>(u32(${value}), 0u);`;
        } else if (type.storage === 'u32' && type.value === 'vec4<bool>') {
          // bool, components === 4
          return `${name}[${offset}]=dot(vec4<u32>(0x1, 0x100, 0x10000, 0x1000000), vec4<u32>(${value}));`;
        } else {
          throw new Error(`not supported combination of storage type ${type.storage} and value type ${type.value} yet`);
        }
      })();

      const getByOffset = (offset: number|string) => (() => {
        if (type.storage === type.value) {
          return `${name}[${offset}]`;
        } else if (type.storage === 'vec2<u32>' && type.value === 'i32') {
          // int64, components === 1
          return `i32(${name}[${offset}].x)`;
        } else if (type.storage === 'vec2<u32>' && type.value === 'u32') {
          // uint64, components === 1
          return `u32(${name}[${offset}].x)`;
        } else if (type.storage === 'u32' && type.value === 'vec4<bool>') {
          // bool, components === 4
          return `vec4<bool>(bool(${name}[${offset}] & 0xFFu), bool(${name}[${offset}] & 0xFF00u), bool(${name}[${
              offset}] & 0xFF0000u), bool(${name}[${offset}] & 0xFF000000u))`;
        } else {
          throw new Error(`not supported combination of storage type ${type.storage} and value type ${type.value} yet`);
        }
      })();

      const getByIndicesImplementation = rank < 2 ? '' : `
  fn get_${name}ByIndices(indices: ${type.indices}) -> ${valueType} {
    return ${name}[i2o_${name}(indices)];
  }`;

      const getImplementation = rank < 2 ? '' : (() => {
        const params = shape.map((_, i) => `d${i}: u32`).join(', ');
        const dims = shape.map((_, i) => `d${i}`).join(', ');
        return `
  fn get_${name}(${params}) -> ${valueType} {
    return get_${name}ByIndices(${indices(dims)});
  }`;
      })();

      const get = (...indices: ReadonlyArray<number|string>) => {
        if (indices.length !== rank) {
          throw new Error(`indices length must be ${rank}`);
        }

        const normalizedIndices = indices.map(normalizeDim).join(',');

        if (rank === 0) {
          return getByOffset('0u');
        } else if (rank === 1) {
          return getByOffset(normalizedIndices[0]);
        } else {
          implementationUsed.get = true;
          implementationUsed.getByIndices = true;
          implementationUsed.indicesToOffset = true;
          return `get_${name}(${normalizedIndices})`;
        }
      };

      const getByIndices = (varIndices: string) => {
        if (rank < 2) {
          return getByOffset(varIndices);
        } else {
          implementationUsed.getByIndices = true;
          implementationUsed.indicesToOffset = true;
          return `get_${name}ByIndices(${varIndices})`;
        }
      };

      const setByIndicesImplementation = rank < 2 ? '' : `
  fn set_${name}ByIndices(indices: ${type.indices}, value: ${valueType}) {
    ${setByOffset(`i2o_${name}(indices)`, 'value')}
  }`;

      const setImplementation = rank < 2 ? '' : (() => {
        const params = shape.map((_, i) => `d${i}: u32`).join(', ');
        const dims = shape.map((_, i) => `d${i}`).join(', ');
        return `
  fn set_${name}(${params}, value: ${valueType}) {
    set_${name}ByIndices(${indices(dims)}, value);
  }`;
      })();

      const set = (...indicesAndValue: ReadonlyArray<number|string>) => {
        if (indicesAndValue.length !== rank + 1) {
          throw new Error(`indices length must be ${rank}`);
        }
        const value = indicesAndValue[rank];
        if (typeof value !== 'string') {
          throw new Error('value must be string');
        }

        const normalizedIndices = indicesAndValue.slice(0, rank).map(normalizeDim).join(',');

        if (rank === 0) {
          return setByOffset('0u', value);
        } else if (rank === 1) {
          return setByOffset(normalizedIndices[0], value);
        } else {
          implementationUsed.set = true;
          implementationUsed.setByIndices = true;
          implementationUsed.indicesToOffset = true;
          return `set_${name}(${normalizedIndices}, ${value})`;
        }
      };

      const setByIndices = (varIndices: string, value: string) => {
        if (rank < 2) {
          return setByOffset(varIndices, value);
        } else {
          implementationUsed.setByIndices = true;
          implementationUsed.indicesToOffset = true;
          return `set_${name}ByIndices(${varIndices}, ${value});`;
        }
      };

      const impl = () => {
        const impls = [];
        if (implementationUsed.offsetToIndices) {
          impls.push(offsetToIndicesImplementation);
        }
        if (implementationUsed.indicesToOffset) {
          impls.push(indicesToOffsetImplementation);
        }
        if (implementationUsed.set) {
          impls.push(setImplementation);
        }
        if (implementationUsed.setByIndices) {
          impls.push(setByIndicesImplementation);
        }
        if (implementationUsed.get) {
          impls.push(getImplementation);
        }
        if (implementationUsed.getByIndices) {
          impls.push(getByIndicesImplementation);
        }
        return impls.join('\n');
      };

      return {
        impl,
        type,
        offsetToIndices,
        indicesToOffset,
        indices,
        indicesGet,
        indicesSet,
        set,
        setByOffset,
        setByIndices,
        get,
        getByOffset,
        getByIndices,
        // isVec4,
        usage: isInput ? 'input' : 'output',
        name,
        shape
      };
    };

/**
 * Create a IndicesHelper for an input.
 *
 * @param name - the name of the input.
 * @param type - the tensor type of the input.
 * @param shape - the tensor shape of the input.
 * @param components - the number of components of the input. available values are 1, 2, 3, 4. default is 1.
 * @returns an IndicesHelper for the input.
 */
export const inputVariable =
    (name: string, type: number, shape: readonly number[], components: 1|2|3|4 = 1): IndicesHelper =>
        createIndicesHelper(name, type, shape, true, components);

/**
 * Create a IndicesHelper for an output.
 *
 * @param name - the name of the output.
 * @param type - the tensor type of the output.
 * @param shape - the tensor shape of the output.
 * @param components - the number of components of the input. available values are 1, 2, 3, 4. default is 1.
 * @returns an IndicesHelper for the output.
 */
export const outputVariable =
    (name: string, type: number, shape: readonly number[], components: 1|2|3|4 = 1): IndicesHelper =>
        createIndicesHelper(name, type, shape, false, components);

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
   * A helper function to generate the code snippet for declaring multiple inputs or outputs.
   *
   * @param variables - an array of IndicesHelper for the variables.
   */
  declareVariables(...variables: IndicesHelper[]): string;

  /**
   * Get additional implementation that needs to be added to the shader source.
   */
  readonly additionalImplementations: string;
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
    this.indicesHelpers.push(variable);
    const access = variable.usage === 'input' ? 'read' : 'read_write';
    const storageType = variable.type.storage;
    return `@group(0) @binding(${bindingIndex}) var<storage, ${access}> ${variable.name}: array<${storageType}>;`;
  }

  declareVariables(...variables: IndicesHelper[]): string {
    let i = 0;
    return variables.filter(v => ShapeUtil.size(v.shape) > 0).map(v => this.declareVariable(v, i++)).join('\n');
  }

  private indicesHelpers: IndicesHelper[] = [];

  get additionalImplementations(): string {
    return this.indicesHelpers.map(i => i.impl()).join('\n');
  }
}

export const createShaderHelper = (dispatchGroup: [number, number, number]): ShaderHelper =>
    new ShaderHelperImpl(dispatchGroup);
