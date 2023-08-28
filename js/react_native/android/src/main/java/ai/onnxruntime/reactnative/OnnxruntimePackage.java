// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package ai.onnxruntime.reactnative;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import com.facebook.react.TurboReactPackage;
import com.facebook.react.bridge.NativeModule;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.module.model.ReactModuleInfo;
import com.facebook.react.module.model.ReactModuleInfoProvider;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class OnnxruntimePackage extends TurboReactPackage {

  @Nullable
  @Override
  public NativeModule getModule(String name, ReactApplicationContext reactContext) {
    if (name.equals(OnnxruntimeModule.NAME)) {
      return new ai.onnxruntime.reactnative.OnnxruntimeModule(reactContext);
    } else if (name.equals(OnnxruntimeJSIHelper.NAME)) {
      return new ai.onnxruntime.reactnative.OnnxruntimeJSIHelper(reactContext);
    } else {
      return null;
    }
  }

  @Override
  public ReactModuleInfoProvider getReactModuleInfoProvider() {
    return () -> {
      final Map<String, ReactModuleInfo> moduleInfos = new HashMap<>();
      boolean isTurboModule = BuildConfig.IS_NEW_ARCHITECTURE_ENABLED;
      moduleInfos.put(OnnxruntimeModule.NAME, new ReactModuleInfo(OnnxruntimeModule.NAME, "OnnxruntimeModule",
                                                                  false,        // canOverrideExistingModule
                                                                  false,        // needsEagerInit
                                                                  true,         // hasConstants
                                                                  false,        // isCxxModule
                                                                  isTurboModule // isTurboModule
                                                                  ));
      moduleInfos.put(OnnxruntimeJSIHelper.NAME,
                      new ReactModuleInfo(OnnxruntimeJSIHelper.NAME, OnnxruntimeJSIHelper.NAME,
                                          false, // canOverrideExistingModule
                                          false, // needsEagerInit
                                          true,  // hasConstants
                                          false, // isCxxModule
                                          false  // isTurboModule
                                          ));
      return moduleInfos;
    };
  }
}
