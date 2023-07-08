// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package ai.onnxruntime.reactnative;

import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import android.net.Uri;
import android.util.Log;
import androidx.annotation.NonNull;
import com.facebook.react.bridge.LifecycleEventListener;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.ReadableType;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.modules.blob.BlobModule;
import java.util.Map;

public class OnnxruntimeModule extends ReactContextBaseJavaModule implements LifecycleEventListener {
  private static ReactApplicationContext reactContext;
  private static OnnxruntimeModuleImpl onnxruntimeModuleImpl;

  protected BlobModule blobModule;

  public OnnxruntimeModule(ReactApplicationContext context) {
    super(context);
    reactContext = context;
    onnxruntimeModuleImpl = new OnnxruntimeModuleImpl(context);
  }

  @NonNull
  @Override
  public String getName() {
    return "Onnxruntime";
  }

  public void checkBlobModule() {
    if (blobModule == null) {
      blobModule = getReactApplicationContext().getNativeModule(BlobModule.class);
      if (blobModule == null) {
        throw new RuntimeException("BlobModule is not initialized");
      }
    }
  }

  /**
   * React native binding API to load a model using given uri.
   *
   * @param uri a model file location
   * @param options onnxruntime session options
   * @param promise output returning back to react native js
   * @note the value provided to `promise` includes a key representing the session.
   *       when run() is called, the key must be passed into the first parameter.
   */
  @ReactMethod
  public void loadModel(String uri, ReadableMap options, Promise promise) {
    try {
      WritableMap resultMap = onnxruntimeModuleImpl.loadModel(uri, options);
      promise.resolve(resultMap);
    } catch (Exception e) {
      promise.reject("Failed to load model \"" + uri + "\": " + e.getMessage(), e);
    }
  }

  /**
   * React native binding API to load a model using blob object that data stored in BlobModule.
   *
   * @param data the blob object
   * @param options onnxruntime session options
   * @param promise output returning back to react native js
   * @note the value provided to `promise` includes a key representing the session.
   *       when run() is called, the key must be passed into the first parameter.
   */
  @ReactMethod
  public void loadModelFromBlob(ReadableMap data, ReadableMap options, Promise promise) {
    try {
      checkBlobModule();
      String blobId = data.getString("blobId");
      byte[] bytes = blobModule.resolve(blobId, data.getInt("offset"), data.getInt("size"));
      blobModule.remove(blobId);
      WritableMap resultMap = onnxruntimeModuleImpl.loadModel(bytes, options);
      promise.resolve(resultMap);
    } catch (Exception e) {
      promise.reject("Failed to load model from buffer: " + e.getMessage(), e);
    }
  }

  /**
   * React native binding API to dispose a session.
   *
   * @param key session key representing a session given at loadModel()
   * @param promise output returning back to react native js
   */
  @ReactMethod
  public void dispose(String key, Promise promise) {
    try {
      onnxruntimeModuleImpl.dispose(key);
      promise.resolve(null);
    } catch (OrtException e) {
      promise.reject("Failed to dispose session: " + e.getMessage(), e);
    }
  }

  /**
   * React native binding API to run a model using given uri.
   *
   * @param key session key representing a session given at loadModel()
   * @param input an input tensor
   * @param output an output names to be returned
   * @param options onnxruntime run options
   * @param promise output returning back to react native js
   */
  @ReactMethod
  public void run(String key, ReadableMap input, ReadableArray output, ReadableMap options, Promise promise) {
    try {
      checkBlobModule();
      WritableMap resultMap = onnxruntimeModuleImpl.run(key, input, output, options);
      promise.resolve(resultMap);
    } catch (Exception e) {
      promise.reject("Fail to inference: " + e.getMessage(), e);
    }
  }


  @Override
  public void onHostResume() {}

  @Override
  public void onHostPause() {}

  @Override
  public void onHostDestroy() {
    Map<String, OrtSession> sessionMap = onnxruntimeModuleImpl.getSessionMap();
    for (String key : sessionMap.keySet()) {
      try {
        dispose(key);
      } catch (Exception e) {
        Log.e("onHostDestroy", "Failed to dispose session: " + key, e);
      }
    }
    sessionMap.clear();
  }
}
