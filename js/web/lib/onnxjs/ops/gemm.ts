// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Attribute} from '../attribute';
import {InferenceHandler} from '../backend';
import {Operator} from '../operators';
import {Tensor} from '../tensor';

export abstract class Gemm implements Operator {
  constructor(isOptionalC: boolean) {
    this.isOptionalC = isOptionalC;
  }

  abstract run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]>;

  initialize(attributes: Attribute): void {
    this.transA = attributes.getInt('transA', 0) !== 0;
    this.transB = attributes.getInt('transB', 0) !== 0;
    this.alpha = attributes.getFloat('alpha', 1);
    this.beta = attributes.getFloat('beta', 1);
  }

  checkInputs(inputs: Tensor[]): boolean {
    if (!inputs) {
      return false;
    }
    if (this.isOptionalC && (inputs.length < 2 || inputs.length > 3)) {
      return false;
    }
    if (!this.isOptionalC && inputs.length !== 3) {
      return false;
    }

    // 'C' can be of dimensionality 1 or 2 only
    if (inputs.length === 3 && inputs[2].dims.length !== 1 && inputs[2].dims.length !== 2) {
      return false;
    }

    return this.checkInputTypes(inputs);
  }

  protected checkInputTypes(inputs: Tensor[]): boolean {
    if ((inputs[0].type !== 'float32' && inputs[0].type !== 'float64') ||
        (inputs[1].type !== 'float32' && inputs[1].type !== 'float64') ||
        (inputs.length === 3 && inputs[2].type !== 'float32' && inputs[2].type !== 'float64')) {
      return false;
    }

    if ((inputs[0].type !== inputs[1].type) || (inputs.length === 3 && inputs[0].type !== inputs[2].type)) {
      return false;
    }

    return true;
  }

  protected transA: boolean;
  protected transB: boolean;
  protected alpha: number;
  protected beta: number;

  protected isOptionalC: boolean;  // in opset 11, C becomes optional
}
