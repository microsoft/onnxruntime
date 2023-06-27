// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

export * from 'onnxruntime-common';
import * from 'onnxruntime-common';
import {WebAssemblyTrainingBackend} from './backend';
import { registerBackend } from 'onnxruntime-web';

registerBackend('cpu', WebAssemblyTrainingBackend, 100);
