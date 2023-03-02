// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {OrtWasmThreadedModule} from './ort-wasm-threaded';

declare const moduleFactory: EmscriptenModuleFactory<OrtWasmThreadedModule>;
export default moduleFactory;
