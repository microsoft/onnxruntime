// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package ai.onnxruntime.reactnative;

import android.util.Log;

class OnnxruntimeExtensions {
  static public String getLibraryPath() {
    Log.i("OnnxruntimeExtensions",
          "ORT Extensions is not enabled in the current configuration. If you want to enable this support, "
              + "please add \"onnxruntimeEnableExtensions\": \"true\" in your project root directory package.json.");
    return null;
  }
}
