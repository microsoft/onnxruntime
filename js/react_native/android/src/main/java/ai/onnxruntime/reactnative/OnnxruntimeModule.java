// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package ai.onnxruntime.reactnative;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.OrtSession.RunOptions;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.providers.NNAPIFlags;
import android.net.Uri;
import android.os.Build;
import android.util.Log;
import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.LifecycleEventListener;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.ReadableType;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.modules.blob.BlobModule;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.math.BigInteger;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@RequiresApi(api = Build.VERSION_CODES.N)
public class OnnxruntimeModule extends ReactContextBaseJavaModule implements LifecycleEventListener {
  private static ReactApplicationContext reactContext;

  private static OrtEnvironment ortEnvironment = OrtEnvironment.getEnvironment();
  private static Map<String, OrtSession> sessionMap = new HashMap<>();

  private static BigInteger nextSessionId = new BigInteger("0");
  private static String getNextSessionKey() {
    String key = nextSessionId.toString();
    nextSessionId = nextSessionId.add(BigInteger.valueOf(1));
    return key;
  }

  protected BlobModule blobModule;

  public OnnxruntimeModule(ReactApplicationContext context) {
    super(context);
    reactContext = context;
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
      WritableMap resultMap = loadModel(uri, options);
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
      WritableMap resultMap = loadModel(bytes, options);
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
      dispose(key);
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
      WritableMap resultMap = run(key, input, output, options);
      promise.resolve(resultMap);
    } catch (Exception e) {
      promise.reject("Fail to inference: " + e.getMessage(), e);
    }
  }

  /**
   * Load a model from raw resource directory.
   *
   * @param uri uri parameter from react native loadModel()
   * @param options onnxruntime session options
   * @return model loading information, such as key, input names, and output names
   */
  public WritableMap loadModel(String uri, ReadableMap options) throws Exception {
    return loadModelImpl(uri, null, options);
  }

  /**
   * Load a model from buffer.
   *
   * @param modelData the model data buffer
   * @param options onnxruntime session options
   * @return model loading information, such as key, input names, and output names
   */
  public WritableMap loadModel(byte[] modelData, ReadableMap options) throws Exception {
    return loadModelImpl("", modelData, options);
  }

  /**
   * Load model implementation method for either from model path or model data buffer.
   *
   * @param uri uri parameter from react native loadModel()
   * @param modelData model data buffer
   * @param options onnxruntime session options
   * @return model loading information map, such as key, input names, and output names
   */
  private WritableMap loadModelImpl(String uri, byte[] modelData, ReadableMap options) throws Exception {
    OrtSession ortSession;
    SessionOptions sessionOptions = parseSessionOptions(options);

    // optional call for registering custom ops when ort extensions enabled
    OnnxruntimeExtensions ortExt = new OnnxruntimeExtensions();
    ortExt.registerOrtExtensionsIfEnabled(sessionOptions);

    if (modelData != null && modelData.length > 0) {
      // load model via model data array
      ortSession = ortEnvironment.createSession(modelData, sessionOptions);
    } else {
      // load model via model path string uri
      InputStream modelStream =
          reactContext.getApplicationContext().getContentResolver().openInputStream(Uri.parse(uri));
      Reader reader = new BufferedReader(new InputStreamReader(modelStream));
      byte[] modelArray = new byte[modelStream.available()];
      modelStream.read(modelArray);
      modelStream.close();
      ortSession = ortEnvironment.createSession(modelArray, sessionOptions);
    }

    String key = getNextSessionKey();
    sessionMap.put(key, ortSession);

    WritableMap resultMap = Arguments.createMap();
    resultMap.putString("key", key);
    WritableArray inputNames = Arguments.createArray();
    for (String inputName : ortSession.getInputNames()) {
      inputNames.pushString(inputName);
    }
    resultMap.putArray("inputNames", inputNames);
    WritableArray outputNames = Arguments.createArray();
    for (String outputName : ortSession.getOutputNames()) {
      outputNames.pushString(outputName);
    }
    resultMap.putArray("outputNames", outputNames);

    return resultMap;
  }

  /**
   * Dispose a model using given key.
   *
   * @param key a session key representing the session given at loadModel()
   */
  public void dispose(String key) throws OrtException {
    OrtSession ortSession = sessionMap.get(key);
    if (ortSession != null) {
      ortSession.close();
      sessionMap.remove(key);
    }
  }

  /**
   * Run a model using given uri.
   *
   * @param key a session key representing the session given at loadModel()
   * @param input an input tensor
   * @param output an output names to be returned
   * @param options onnxruntime run options
   * @return inference result
   */
  public WritableMap run(String key, ReadableMap input, ReadableArray output, ReadableMap options) throws Exception {
    OrtSession ortSession = sessionMap.get(key);
    if (ortSession == null) {
      throw new Exception("Model is not loaded.");
    }

    RunOptions runOptions = parseRunOptions(options);

    checkBlobModule();

    long startTime = System.currentTimeMillis();
    Map<String, OnnxTensor> feed = new HashMap<>();
    Iterator<String> iterator = ortSession.getInputNames().iterator();
    Result result = null;
    try {
      while (iterator.hasNext()) {
        String inputName = iterator.next();

        ReadableMap inputMap = input.getMap(inputName);
        if (inputMap == null) {
          throw new Exception("Can't find input: " + inputName);
        }

        OnnxTensor onnxTensor = TensorHelper.createInputTensor(blobModule, inputMap, ortEnvironment);
        feed.put(inputName, onnxTensor);
      }

      Set<String> requestedOutputs = null;
      if (output.size() > 0) {
        requestedOutputs = new HashSet<>();
        for (int i = 0; i < output.size(); ++i) {
          requestedOutputs.add(output.getString(i));
        }
      }

      long duration = System.currentTimeMillis() - startTime;
      Log.d("Duration", "createInputTensor: " + duration);

      startTime = System.currentTimeMillis();
      if (requestedOutputs != null) {
        result = ortSession.run(feed, requestedOutputs, runOptions);
      } else {
        result = ortSession.run(feed, runOptions);
      }
      duration = System.currentTimeMillis() - startTime;
      Log.d("Duration", "inference: " + duration);

      startTime = System.currentTimeMillis();
      WritableMap resultMap = TensorHelper.createOutputTensor(blobModule, result);
      duration = System.currentTimeMillis() - startTime;
      Log.d("Duration", "createOutputTensor: " + duration);

      return resultMap;

    } finally {
      OnnxValue.close(feed);
      if (result != null) {
        result.close();
      }
    }
  }

  private static final Map<String, SessionOptions.OptLevel> graphOptimizationLevelTable =
      Stream
          .of(new Object[][] {
              {"disabled", SessionOptions.OptLevel.NO_OPT},
              {"basic", SessionOptions.OptLevel.BASIC_OPT},
              {"extended", SessionOptions.OptLevel.EXTENDED_OPT},
              {"all", SessionOptions.OptLevel.ALL_OPT},
          })
          .collect(Collectors.toMap(p -> (String)p[0], p -> (SessionOptions.OptLevel)p[1]));

  private static final Map<String, SessionOptions.ExecutionMode> executionModeTable =
      Stream
          .of(new Object[][] {{"sequential", SessionOptions.ExecutionMode.SEQUENTIAL},
                              {"parallel", SessionOptions.ExecutionMode.PARALLEL}})
          .collect(Collectors.toMap(p -> (String)p[0], p -> (SessionOptions.ExecutionMode)p[1]));

  private SessionOptions parseSessionOptions(ReadableMap options) throws OrtException {
    SessionOptions sessionOptions = new SessionOptions();

    if (options.hasKey("intraOpNumThreads")) {
      int intraOpNumThreads = options.getInt("intraOpNumThreads");
      if (intraOpNumThreads > 0 && intraOpNumThreads < Integer.MAX_VALUE) {
        sessionOptions.setIntraOpNumThreads(intraOpNumThreads);
      }
    }

    if (options.hasKey("interOpNumThreads")) {
      int interOpNumThreads = options.getInt("interOpNumThreads");
      if (interOpNumThreads > 0 && interOpNumThreads < Integer.MAX_VALUE) {
        sessionOptions.setIntraOpNumThreads(interOpNumThreads);
      }
    }

    if (options.hasKey("graphOptimizationLevel")) {
      String graphOptimizationLevel = options.getString("graphOptimizationLevel");
      if (graphOptimizationLevelTable.containsKey(graphOptimizationLevel)) {
        sessionOptions.setOptimizationLevel(graphOptimizationLevelTable.get(graphOptimizationLevel));
      }
    }

    if (options.hasKey("enableCpuMemArena")) {
      boolean enableCpuMemArena = options.getBoolean("enableCpuMemArena");
      sessionOptions.setCPUArenaAllocator(enableCpuMemArena);
    }

    if (options.hasKey("enableMemPattern")) {
      boolean enableMemPattern = options.getBoolean("enableMemPattern");
      sessionOptions.setMemoryPatternOptimization(enableMemPattern);
    }

    if (options.hasKey("executionMode")) {
      String executionMode = options.getString("executionMode");
      if (executionModeTable.containsKey(executionMode)) {
        sessionOptions.setExecutionMode(executionModeTable.get(executionMode));
      }
    }

    if (options.hasKey("executionProviders")) {
      ReadableArray executionProviders = options.getArray("executionProviders");
      for (int i = 0; i < executionProviders.size(); ++i) {
        String epName = null;
        ReadableMap epOptions = null;
        if (executionProviders.getType(i) == ReadableType.String) {
          epName = executionProviders.getString(i);
        } else {
          epOptions = executionProviders.getMap(i);
          epName = epOptions.getString("name");
        }
        if (epName.equals("nnapi")) {
          EnumSet<NNAPIFlags> flags = EnumSet.noneOf(NNAPIFlags.class);
          if (epOptions != null) {
            if (epOptions.hasKey("useFP16") && epOptions.getBoolean("useFP16")) {
              flags.add(NNAPIFlags.USE_FP16);
            }
            if (epOptions.hasKey("useNCHW") && epOptions.getBoolean("useNCHW")) {
              flags.add(NNAPIFlags.USE_NCHW);
            }
            if (epOptions.hasKey("cpuDisabled") && epOptions.getBoolean("cpuDisabled")) {
              flags.add(NNAPIFlags.CPU_DISABLED);
            }
            if (epOptions.hasKey("cpuOnly") && epOptions.getBoolean("cpuOnly")) {
              flags.add(NNAPIFlags.CPU_ONLY);
            }
          }
          sessionOptions.addNnapi(flags);
        } else if (epName.equals("xnnpack")) {
          sessionOptions.addXnnpack(Collections.emptyMap());
        } else if (epName.equals("cpu")) {
          continue;
        } else {
          throw new OrtException("Unsupported execution provider: " + epName);
        }
      }
    }

    if (options.hasKey("logId")) {
      String logId = options.getString("logId");
      sessionOptions.setLoggerId(logId);
    }

    if (options.hasKey("logSeverityLevel")) {
      int logSeverityLevel = options.getInt("logSeverityLevel");
      sessionOptions.setSessionLogVerbosityLevel(logSeverityLevel);
    }

    return sessionOptions;
  }

  private RunOptions parseRunOptions(ReadableMap options) throws OrtException {
    RunOptions runOptions = new RunOptions();

    if (options.hasKey("logSeverityLevel")) {
      int logSeverityLevel = options.getInt("logSeverityLevel");
      runOptions.setLogVerbosityLevel(logSeverityLevel);
    }

    if (options.hasKey("tag")) {
      String tag = options.getString("tag");
      runOptions.setRunTag(tag);
    }

    return runOptions;
  }

  @Override
  public void onHostResume() {}

  @Override
  public void onHostPause() {}

  @Override
  public void onHostDestroy() {
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
