// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import Foundation
import onnxruntimec

public class ORTCoreMLExecutionProviderOptions {
    public var useCPUOnly = false
    public var enableOnSubgraphs = false
    public var onlyEnableForDevicesWithANE = false
}



extension ORTSessionOptions {
    public func appendCoreMLExecutionProvider(with options: ORTCoreMLExecutionProviderOptions?) -> Bool {
//        let flags: UInt32 = (options.useCPUOnly ? COREML_FLAG_USE_CPU_ONLY : 0) | (options.enableOnSubgraphs ? COREML_FLAG_ENABLE_ON_SUBGRAPH : 0) | (options.onlyEnableForDevicesWithANE ? COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE : 0)
        return true
    }
}

