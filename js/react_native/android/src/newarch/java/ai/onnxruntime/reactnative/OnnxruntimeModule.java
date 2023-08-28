// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package ai.onnxruntime.reactnative;

import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import android.net.Uri;
import android.util.Log;
import androidx.annotation.NonNull;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.ReadableType;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.module.annotations.ReactModule;
import com.facebook.react.modules.blob.BlobModule;
import java.util.Map;

@ReactModule(name = OnnxruntimeModule.NAME)
public class OnnxruntimeModule extends NativeOnnxruntimeSpec {
  public static final String NAME = "Onnxruntime";
  private static ReactApplicationContext reactContext;
  private static Onnxruntime onnxruntime;

  public OnnxruntimeModule(ReactApplicationContext context) {
    super(context);
    reactContext = context;
    onnxruntime = new Onnxruntime(context);
  }

  @NonNull
  @Override
  public String getName() {
    return NAME;
  }

  public Onnxruntime getOnnxruntime() { return onnxruntime; }

  @ReactMethod
  public void loadModel(String uri, ReadableMap options, Promise promise) {
    onnxruntime.loadModel(uri, options, promise);
  }

  @ReactMethod
  public void loadModelFromBlob(ReadableMap data, ReadableMap options, Promise promise) {
    onnxruntime.loadModelFromBlob(data, options, promise);
  }

  @ReactMethod
  public void dispose(String key, Promise promise) {
    onnxruntime.dispose(key, promise);
  }

  @ReactMethod
  public void run(String key, ReadableMap input, ReadableArray output, ReadableMap options, Promise promise) {
    onnxruntime.run(key, input, output, options, promise);
  }
}
