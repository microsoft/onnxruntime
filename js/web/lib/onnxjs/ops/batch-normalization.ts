// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Attribute} from '../attribute';
import {InferenceHandler} from '../backend';
import {Operator} from '../operators';
import {Tensor} from '../tensor';

// currently this operator supports ONLY 'test' mode
// inputs/outputs and parameters will reflect that
// the operator implementation only supports test mode
export abstract class BatchNormalization implements Operator {
  abstract run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]>;

  initialize(attributes: Attribute): void {
    this.epsilon = attributes.getFloat('epsilon', 1e-5);
    this.momentum = attributes.getFloat('momentum', 0.9);
    this.spatial = attributes.getInt('spatial', 1);
  }

  checkInputs(inputs: Tensor[]): boolean {
    if (!inputs || inputs.length !== 5) {
      return false;
    }

    return this.checkInputTypes(inputs);
  }

  protected checkInputTypes(inputs: Tensor[]): boolean {
    const X = inputs[0];
    const scale = inputs[1];
    const B = inputs[2];
    const mean = inputs[3];
    const var_ = inputs[4];

    // input should atleast have three dimensions - N,C,dim1,...,dimn
    // other inputs can have only one dimensions
    if (X.dims.length < 3 || scale.dims.length !== 1 || B.dims.length !== 1 || mean.dims.length !== 1 ||
        var_.dims.length !== 1) {
      return false;
    }
    if (scale.dims[0] !== X.dims[1] || B.dims[0] !== X.dims[1] || mean.dims[0] !== X.dims[1] ||
        var_.dims[0] !== X.dims[1]) {
      return false;
    }
    if ((X.type !== 'float32' && X.type !== 'float64') || (scale.type !== 'float32' && scale.type !== 'float64') ||
        (B.type !== 'float32' && B.type !== 'float64') || (mean.type !== 'float32' && mean.type !== 'float64') ||
        (var_.type !== 'float32' && var_.type !== 'float64')) {
      return false;
    }
    return true;
  }

  protected epsilon: number;
  protected momentum: number;
  protected spatial: number;
}
