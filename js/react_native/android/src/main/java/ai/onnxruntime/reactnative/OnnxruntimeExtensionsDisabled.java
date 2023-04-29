// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package ai.onnxruntime.reactnative;

import ai.onnxruntime.OrtSession.SessionOptions;
import android.util.Log;

class OnnxruntimeExtensions {
  public void registerOrtExtensionsIfEnabled(SessionOptions sessionOptions) {
    Log.d(
        "Info",
        "ORT Extensions is not enabled in the current configuration. If you want to enable this support,"
            +
            "please add \"onnxruntimeEnableExtensions\": \"true\" as a top-level entry in your project's root directory package.json.");
    return;
  }
}
