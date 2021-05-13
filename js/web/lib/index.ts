// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

export * from 'onnxruntime-common';
import {registerBackend} from 'onnxruntime-common';
import {onnxjsBackend} from './backend-onnxjs';
import {wasmBackend} from './backend-wasm';

registerBackend('webgl', onnxjsBackend, 1);
registerBackend('wasm', wasmBackend, 2);
