// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

export * from 'onnxruntime-common';
import {registerBackend} from 'onnxruntime-common';
import {onnxruntimeBackend} from './backend';

registerBackend('cpu', onnxruntimeBackend);
