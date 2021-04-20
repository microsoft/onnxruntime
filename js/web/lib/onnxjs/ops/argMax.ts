// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Attribute} from '../attribute';
import {InferenceHandler} from '../backend';
import {NUMBER_TYPES, Operator} from '../operators';
import {Tensor} from '../tensor';

export abstract class ArgMax implements Operator {
  abstract run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]>;

  initialize(attributes: Attribute): void {
    this.axis = attributes.getInt('axis', 0);
    this.keepDims = attributes.getInt('keepdims', 1) === 1;
  }

  checkInputs(inputs: Tensor[]): boolean {
    if (!inputs || inputs.length !== 1) {
      return false;
    }

    return this.checkInputTypes(inputs);
  }

  protected checkInputTypes(inputs: Tensor[]): boolean {
    if (NUMBER_TYPES.indexOf(inputs[0].type) === -1) {
      return false;
    }

    return true;
  }

  protected axis: number;
  protected keepDims: boolean;
}
