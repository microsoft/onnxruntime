// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package ai.onnxruntime.reactnative;

import java.util.Map;
import java.util.HashMap;
import android.os.Build;
import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import com.facebook.react.bridge.JavaScriptContextHolder;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.turbomodule.core.CallInvokerHolderImpl;

@RequiresApi(api = Build.VERSION_CODES.N)
public class OnnxruntimeModule extends ReactContextBaseJavaModule {
  private static ReactApplicationContext reactContext;
  private volatile boolean nativeLibLoaded = false;

  public OnnxruntimeModule(ReactApplicationContext context) {
    super(context);
    reactContext = context;
  }

  @NonNull
  @Override
  public String getName() {
    return "Onnxruntime";
  }

  native void nativeInstall(long jsiPointer, CallInvokerHolderImpl jsCallInvokerHolder);

  native void nativeCleanup();

  @Override
  public void invalidate() {
    super.invalidate();
    // Guard: invalidate() can be called before install() loads the native library,
    // e.g. during bridge reload, causing UnsatisfiedLinkError and crashing the app.
    if (nativeLibLoaded) {
      nativeCleanup();
    }
  }

  /**
   * Install onnxruntime JSI API
   */
  @ReactMethod(isBlockingSynchronousMethod = true)
  public boolean install() {
    try {
      System.loadLibrary("onnxruntimejsi");
      nativeLibLoaded = true;
      JavaScriptContextHolder jsContext = getReactApplicationContext().getJavaScriptContextHolder();
      CallInvokerHolderImpl jsCallInvokerHolder =
        (CallInvokerHolderImpl) getReactApplicationContext().getCatalystInstance().getJSCallInvokerHolder();
      nativeInstall(jsContext.get(), jsCallInvokerHolder);
      return true;
    } catch (Exception e) {
      return false;
    }
  }

  @Override
  public Map<String, Object> getConstants() {
    final Map<String, Object> constants = new HashMap();
    constants.put("ORT_EXTENSIONS_PATH", OnnxruntimeExtensions.getLibraryPath());
    return constants;
  }
}
