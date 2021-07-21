// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Attribute} from '../attribute';
import {InferenceHandler} from '../backend';
import {Operator} from '../operators';
import {Tensor} from '../tensor';

export abstract class MatMul implements Operator {
  abstract run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]>;

  initialize(attributes: Attribute): void {
    this.activation = attributes.getString('__internal_activation', '');
    this.clipMax = attributes.getFloat('__clip_max', 3.402823e+38);
    this.clipMin = attributes.getFloat('__clip_min', -3.402823e+38);
  }

  checkInputs(inputs: Tensor[]): boolean {
    if (!inputs || inputs.length !== 2) {
      return false;
    }

    if (inputs[0].dims[inputs[0].dims.length - 1] !== inputs[1].dims[inputs[1].dims.length - 2]) {
      return false;
    }

    return this.checkInputTypes(inputs);
  }

  protected checkInputTypes(inputs: Tensor[]): boolean {
    if (inputs[0].type !== 'float32' && inputs[0].type !== 'float64') {
      return false;
    }

    if (inputs[1].type !== 'float32' && inputs[1].type !== 'float64') {
      return false;
    }

    if (inputs[0].type !== inputs[1].type) {
      return false;
    }

    return true;
  }
  protected activation: string;
  protected clipMin: number;
  protected clipMax: number;
}
