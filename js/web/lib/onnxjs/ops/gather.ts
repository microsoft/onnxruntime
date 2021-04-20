// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Attribute} from '../attribute';
import {InferenceHandler} from '../backend';
import {NUMBER_TYPES, Operator} from '../operators';
import {Tensor} from '../tensor';

export abstract class Gather implements Operator {
  abstract run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]>;

  initialize(attributes: Attribute): void {
    this.axis = attributes.getInt('axis', 0);
  }

  checkInputs(inputs: Tensor[]): boolean {
    if (!inputs || inputs.length !== 2) {
      return false;
    }
    const tensorRank = inputs[0].dims.length;
    if (tensorRank < 1) {
      return false;
    }
    if (this.axis < -tensorRank || this.axis > tensorRank - 1) {
      return false;
    }

    return this.checkInputTypes(inputs);
  }

  protected checkInputTypes(inputs: Tensor[]): boolean {
    if (NUMBER_TYPES.indexOf(inputs[0].type) === -1) {
      return false;
    }
    if (inputs[1].type !== 'int32' && inputs[1].type !== 'int16') {
      return false;
    }
    return true;
  }

  protected axis: number;
}
