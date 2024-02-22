// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package ai.onnxruntime.reactnative;

import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.extensions.OrtxPackage;

class OnnxruntimeExtensions {
  public void registerOrtExtensionsIfEnabled(SessionOptions sessionOptions) throws OrtException {
    sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath());
  }
}
