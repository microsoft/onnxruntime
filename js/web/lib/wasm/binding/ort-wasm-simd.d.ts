// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {OrtWasmModule} from './ort-wasm';

export interface OrtWasmSimdModule extends OrtWasmModule {
}

declare const moduleFactory: EmscriptenModuleFactory<OrtWasmSimdModule>;
export default moduleFactory;
