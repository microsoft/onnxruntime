// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package ai.onnxruntime.reactnative;

import ai.onnxruntime.OrtSession.SessionOptions;
import android.util.Log;

class OnnxruntimeExtensions {
  public void registerOrtExtensionsIfEnabled(SessionOptions sessionOptions) {
    Log.i("OnnxruntimeExtensions",
          "ORT Extensions is not enabled in the current configuration. If you want to enable this support, "
              + "please add \"onnxruntimeEnableExtensions\": \"true\" in your project root directory package.json.");
    return;
  }
}
