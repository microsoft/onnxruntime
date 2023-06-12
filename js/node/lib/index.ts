// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

export * from 'onnxruntime-common';
import {registerBackend, env} from 'onnxruntime-common';
import {onnxruntimeBackend} from './backend';
import {version} from './version';

registerBackend('cpu', onnxruntimeBackend, 100);

env.versions.node = version;
