// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Attribute} from '../attribute';
import {InferenceHandler} from '../backend';
import {Operator} from '../operators';
import {Tensor} from '../tensor';

export abstract class Conv implements Operator {
  abstract run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]>;

  initialize(attributes: Attribute): void {
    // TODO : Make this generic enough to compute default attributes for multi-dimensional conv
    this.autoPad = attributes.getString('auto_pad', 'NOTSET');
    this.dilations = attributes.getInts('dilations', [1, 1]);
    this.group = attributes.getInt('group', 1);
    this.kernelShape = attributes.getInts('kernel_shape', []);
    this.pads = attributes.getInts('pads', [0, 0, 0, 0]);
    this.strides = attributes.getInts('strides', [1, 1]);
    this.activation = attributes.getString('__internal_activation', '');
  }

  checkInputs(inputs: Tensor[]): boolean {
    // Refer to the below link for all input checks
    // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
    if (!inputs || (inputs.length !== 2 && inputs.length !== 3)) {
      return false;
    }

    // TODO : Need to add support for multi-dimensional conv
    // currently only support 2-dimensional conv
    if (inputs[0].dims.length !== 4 || inputs[1].dims.length !== 4) {
      return false;
    }

    // FILTER_IN_CHANNEL should be equal to DATA_CHANNEL
    const dataChannel = inputs[0].dims[1];
    const filterInChannel = inputs[1].dims[1] * this.group;
    if (dataChannel !== filterInChannel) {
      return false;
    }

    // if bias is provided it should be 1D and the number of elements should be equal to the number of feature maps
    if (inputs.length === 3 && (inputs[2].dims.length !== 1 || inputs[1].dims[0] !== inputs[2].dims[0])) {
      return false;
    }

    const spatialRank = inputs[0].dims.length - 2;
    // wrong dilations dimension
    if (this.dilations.length !== spatialRank) {
      return false;
    }

    // Wrong strides dimension
    if (this.strides.length !== spatialRank) {
      return false;
    }

    // Wrong pads dimension
    if (this.pads.length !== spatialRank * 2) {
      return false;
    }

    // if kernelShape is specified, it's data length must be 2 less than dims length of the weights tensor
    // (the first 2 dims are batch_size and channels)
    if (this.kernelShape.length !== 0 && this.kernelShape.length !== inputs[1].dims.length - 2) {
      return false;
    }

    return this.checkInputTypes(inputs);
  }

  protected checkInputTypes(inputs: Tensor[]): boolean {
    // TODO : Need to add support for float64
    if (inputs[0].type !== 'float32' || inputs[1].type !== 'float32') {
      return false;
    }

    if (inputs.length === 3 && inputs[2].type !== 'float32') {
      return false;
    }

    return true;
  }

  protected autoPad: string;
  protected dilations: number[];
  protected group: number;
  protected kernelShape: number[];
  protected pads: number[];
  protected strides: number[];
  protected activation: string;
}
