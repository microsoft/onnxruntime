// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {OrtWasmThreadedModule} from './ort-wasm-threaded';

export interface OrtWasmSimdThreadedModule extends OrtWasmThreadedModule {
}

declare const moduleFactory: EmscriptenModuleFactory<OrtWasmSimdThreadedModule>;
export default moduleFactory;
