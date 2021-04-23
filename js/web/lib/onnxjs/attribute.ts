// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import Long from 'long';
import {onnx} from 'onnx-proto';

import {Tensor} from './tensor';
import {LongUtil} from './util';

export declare namespace Attribute {
  export interface DataTypeMap {
    float: number;
    int: number;
    string: string;
    tensor: Tensor;
    floats: number[];
    ints: number[];
    strings: string[];
    tensors: Tensor[];
  }

  export type DataType = keyof DataTypeMap;
}

type ValueTypes = Attribute.DataTypeMap[Attribute.DataType];

type Value = [ValueTypes, Attribute.DataType];

export class Attribute {
  constructor(attributes: onnx.IAttributeProto[]|null|undefined) {
    this._attributes = new Map();
    if (attributes !== null && attributes !== undefined) {
      for (const attr of attributes) {
        this._attributes.set(attr.name!, [Attribute.getValue(attr), Attribute.getType(attr)]);
      }

      if (this._attributes.size < attributes.length) {
        throw new Error('duplicated attribute names');
      }
    }
  }

  set(key: string, type: Attribute.DataType, value: ValueTypes): void {
    this._attributes.set(key, [value, type]);
  }
  delete(key: string): void {
    this._attributes.delete(key);
  }

  getFloat(key: string, defaultValue?: Attribute.DataTypeMap['float']) {
    return this.get(key, 'float', defaultValue);
  }

  getInt(key: string, defaultValue?: Attribute.DataTypeMap['int']) {
    return this.get(key, 'int', defaultValue);
  }

  getString(key: string, defaultValue?: Attribute.DataTypeMap['string']) {
    return this.get(key, 'string', defaultValue);
  }

  getTensor(key: string, defaultValue?: Attribute.DataTypeMap['tensor']) {
    return this.get(key, 'tensor', defaultValue);
  }

  getFloats(key: string, defaultValue?: Attribute.DataTypeMap['floats']) {
    return this.get(key, 'floats', defaultValue);
  }

  getInts(key: string, defaultValue?: Attribute.DataTypeMap['ints']) {
    return this.get(key, 'ints', defaultValue);
  }

  getStrings(key: string, defaultValue?: Attribute.DataTypeMap['strings']) {
    return this.get(key, 'strings', defaultValue);
  }

  getTensors(key: string, defaultValue?: Attribute.DataTypeMap['tensors']) {
    return this.get(key, 'tensors', defaultValue);
  }

  private get<V extends Attribute.DataTypeMap[Attribute.DataType]>(
      key: string, type: Attribute.DataType, defaultValue?: V): V {
    const valueAndType = this._attributes.get(key);
    if (valueAndType === undefined) {
      if (defaultValue !== undefined) {
        return defaultValue;
      }
      throw new Error(`required attribute not found: ${key}`);
    }
    if (valueAndType[1] !== type) {
      throw new Error(`type mismatch: expected ${type} but got ${valueAndType[1]}`);
    }
    return valueAndType[0] as V;
  }

  private static getType(attr: onnx.IAttributeProto): Attribute.DataType {
    switch (attr.type!) {
      case onnx.AttributeProto.AttributeType.FLOAT:
        return 'float';
      case onnx.AttributeProto.AttributeType.INT:
        return 'int';
      case onnx.AttributeProto.AttributeType.STRING:
        return 'string';
      case onnx.AttributeProto.AttributeType.TENSOR:
        return 'tensor';
      case onnx.AttributeProto.AttributeType.FLOATS:
        return 'floats';
      case onnx.AttributeProto.AttributeType.INTS:
        return 'ints';
      case onnx.AttributeProto.AttributeType.STRINGS:
        return 'strings';
      case onnx.AttributeProto.AttributeType.TENSORS:
        return 'tensors';
      default:
        throw new Error(`attribute type is not supported yet: ${onnx.AttributeProto.AttributeType[attr.type!]}`);
    }
  }

  private static getValue(attr: onnx.IAttributeProto) {
    if (attr.type === onnx.AttributeProto.AttributeType.GRAPH ||
        attr.type === onnx.AttributeProto.AttributeType.GRAPHS) {
      throw new Error('graph attribute is not supported yet');
    }

    const value = this.getValueNoCheck(attr);

    // cast LONG to number
    if (attr.type === onnx.AttributeProto.AttributeType.INT && Long.isLong(value)) {
      return value.toNumber();
    }

    // cast LONG[] to number[]
    if (attr.type === onnx.AttributeProto.AttributeType.INTS) {
      const arr = (value as Array<number|Long>);
      const numberValue: number[] = new Array<number>(arr.length);

      for (let i = 0; i < arr.length; i++) {
        const maybeLong = arr[i];
        numberValue[i] = LongUtil.longToNumber(maybeLong);
      }

      return numberValue;
    }

    // cast onnx.TensorProto to onnxjs.Tensor
    if (attr.type === onnx.AttributeProto.AttributeType.TENSOR) {
      return Tensor.fromProto(value as onnx.ITensorProto);
    }

    // cast onnx.TensorProto[] to onnxjs.Tensor[]
    if (attr.type === onnx.AttributeProto.AttributeType.TENSORS) {
      const tensorProtos = value as onnx.ITensorProto[];
      return tensorProtos.map(value => Tensor.fromProto(value));
    }

    // cast Uint8Array to string
    if (attr.type === onnx.AttributeProto.AttributeType.STRING) {
      const utf8String = value as Uint8Array;
      return Buffer.from(utf8String.buffer, utf8String.byteOffset, utf8String.byteLength).toString();
    }

    // cast Uint8Array[] to string[]
    if (attr.type === onnx.AttributeProto.AttributeType.STRINGS) {
      const utf8Strings = value as Uint8Array[];
      return utf8Strings.map(
          utf8String => Buffer.from(utf8String.buffer, utf8String.byteOffset, utf8String.byteLength).toString());
    }

    return value as ValueTypes;
  }

  private static getValueNoCheck(attr: onnx.IAttributeProto) {
    switch (attr.type!) {
      case onnx.AttributeProto.AttributeType.FLOAT:
        return attr.f;
      case onnx.AttributeProto.AttributeType.INT:
        return attr.i;
      case onnx.AttributeProto.AttributeType.STRING:
        return attr.s;
      case onnx.AttributeProto.AttributeType.TENSOR:
        return attr.t;
      case onnx.AttributeProto.AttributeType.GRAPH:
        return attr.g;
      case onnx.AttributeProto.AttributeType.FLOATS:
        return attr.floats;
      case onnx.AttributeProto.AttributeType.INTS:
        return attr.ints;
      case onnx.AttributeProto.AttributeType.STRINGS:
        return attr.strings;
      case onnx.AttributeProto.AttributeType.TENSORS:
        return attr.tensors;
      case onnx.AttributeProto.AttributeType.GRAPHS:
        return attr.graphs;
      default:
        throw new Error(`unsupported attribute type: ${onnx.AttributeProto.AttributeType[attr.type!]}`);
    }
  }

  protected _attributes: Map<string, Value>;
}
