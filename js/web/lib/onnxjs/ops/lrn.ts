// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Attribute} from '../attribute';
import {InferenceHandler} from '../backend';
import {Operator} from '../operators';
import {Tensor} from '../tensor';

export abstract class Lrn implements Operator {
  abstract run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]>;

  initialize(attributes: Attribute): void {
    this.alpha = attributes.getFloat('alpha', 1E-4);
    this.beta = attributes.getFloat('beta', 0.75);
    this.bias = attributes.getFloat('bias', 1.0);
    this.size = attributes.getInt('size');
  }

  checkInputs(inputs: Tensor[]): boolean {
    if (!inputs || inputs.length !== 1) {
      return false;
    }

    // input tensor must have atleast 3 dimensions
    if (inputs[0].dims.length < 3) {
      return false;
    }

    return this.checkInputTypes(inputs);
  }

  protected checkInputTypes(inputs: Tensor[]): boolean {
    if (inputs[0].type !== 'float32' && inputs[0].type !== 'float64') {
      return false;
    }

    return true;
  }

  protected alpha: number;
  protected beta: number;
  protected bias: number;
  protected size: number;
}
