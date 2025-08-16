// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package ai.onnxruntime.reactnative;

import android.os.Build;
import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import com.facebook.react.bridge.JavaScriptContextHolder;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.turbomodule.core.CallInvokerHolderImpl;
import com.facebook.react.common.annotations.FrameworkAPI;

@RequiresApi(api = Build.VERSION_CODES.N)
public class OnnxruntimeModule extends ReactContextBaseJavaModule {
  private static ReactApplicationContext reactContext;

  public OnnxruntimeModule(ReactApplicationContext context) {
    super(context);
    reactContext = context;
  }

  @NonNull
  @Override
  public String getName() {
    return "Onnxruntime";
  }

  @OptIn(FrameworkAPI::class)
  native void nativeInstall(long jsiPointer, CallInvokerHolderImpl jsCallInvokerHolder);

  native void nativeCleanup();

  @Override
  public void invalidate() {
    super.invalidate();
    nativeCleanup();
  }

  /**
   * Install onnxruntime JSI API
   */
  @ReactMethod(isBlockingSynchronousMethod = true)
  @OptIn(FrameworkAPI::class)
  public boolean install() {
    try {
      System.loadLibrary("onnxruntimejsi");
      JavaScriptContextHolder jsContext = getReactApplicationContext().getJavaScriptContextHolder();
      CallInvokerHolderImpl jsCallInvokerHolder = getReactApplicationContext().getCatalystInstance().getJSCallInvokerHolder();
      nativeInstall(jsContext.get(), jsCallInvokerHolder);
      return true;
    } catch (Exception e) {
      return false;
    }
  }

  @Override
  public Map<String, Object> getConstants() {
    final Map<String, Object> constants = new HashMap<>();
    constants.put("ORT_EXTENSIONS_PATH", OnnxruntimeExtensions.getLibraryPath());
    return constants;
  }
}
