// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace Microsoft.ML.OnnxRuntime
{
#if __ENABLE_TRAINING_APIS__
    /// <summary>
    /// Property types
    /// </summary>
    public enum OrtPropertyType
    {
        OrtIntProperty = 0,
        OrtFloatProperty = 1,
        OrtStringProperty = 2,
    }
#endif
}
