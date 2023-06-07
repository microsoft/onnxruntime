// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

export * from 'onnxruntime-common';
import {registerBackend} from 'onnxruntime-common';
export { listSupportedBackends } from './backend';
import {onnxruntimeBackend, listSupportedBackends} from './backend';

const backends = listSupportedBackends();
for (const backend of backends) {
    registerBackend(backend.name, onnxruntimeBackend, 100);
}
