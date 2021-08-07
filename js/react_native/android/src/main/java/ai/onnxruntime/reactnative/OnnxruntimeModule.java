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
import android.net.Uri;
import android.os.Build;
import android.util.Log;
import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.ReadableType;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@RequiresApi(api = Build.VERSION_CODES.N)
public class OnnxruntimeModule extends ReactContextBaseJavaModule {
  private static ReactApplicationContext reactContext;

  private static OrtEnvironment ortEnvironment = OrtEnvironment.getEnvironment();
  private static Map<String, OrtSession> sessionMap = new HashMap<>();

  public OnnxruntimeModule(ReactApplicationContext context) {
    super(context);
    reactContext = context;
  }

  @NonNull
  @Override
  public String getName() {
    return "Onnxruntime";
  }

  /**
   * React native binding API to load a model using given uri.
   *
   * @param uri a model file location. it's used as a key when multiple sessions are created, i.e. multiple models are
   *     loaded.
   * @param options onnxruntime session options
   * @param promise output returning back to react native js
   * @note when run() is called, the same uri must be passed into the first parameter.
   */
  @ReactMethod
  public void loadModel(String uri, ReadableMap options, Promise promise) {
    try {
      WritableMap resultMap = loadModel(uri, options);
      promise.resolve(resultMap);
    } catch (Exception e) {
      promise.reject("Can't read a model " + uri, e);
    }
  }

  /**
   * React native binding API to run a model using given uri.
   *
   * @param key a model file location given at loadModel()
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
      promise.reject("Fail to inference", e);
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
    InputStream modelStream = reactContext.getApplicationContext().getContentResolver().openInputStream(Uri.parse(uri));
    WritableMap resultMap = loadModel(uri, modelStream, options);
    modelStream.close();
    return resultMap;
  }

  /**
   * Load a model from raw resource directory.
   *
   * @param uri uri parameter from react native loadModel() or dummy string for unit testing purpose
   * @param modelStream a input stream to read a model
   * @param options onnxruntime session options
   * @return model loading information, such as key, input names, and output names
   */
  public WritableMap loadModel(String uri, InputStream modelStream, ReadableMap options) throws Exception {
    OrtSession ortSession = null;

    if (!sessionMap.containsKey(uri)) {
      byte[] modelArray = null;
      try {
        Reader reader = new BufferedReader(new InputStreamReader(modelStream));
        modelArray = new byte[modelStream.available()];
        modelStream.read(modelArray);
      } catch (IOException e) {
        throw new Exception("Can't read a model " + uri, e);
      }

      try {
        SessionOptions sessionOptions = parseSessionOptions(options);
        ortSession = ortEnvironment.createSession(modelArray, sessionOptions);
        sessionMap.put(uri, ortSession);
      } catch (OrtException e) {
        throw new Exception("Can't create InferenceSession", e);
      }
    } else {
      ortSession = sessionMap.get(uri);
    }

    WritableMap resultMap = Arguments.createMap();
    resultMap.putString("key", uri);
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
   * Run a model using given uri.
   *
   * @param key a model file location given at loadModel()
   * @param input an input tensor
   * @param output an output names to be returned
   * @param options onnxruntime run options
   * @return inference result
   */
  public WritableMap run(String key, ReadableMap input, ReadableArray output, ReadableMap options) throws Exception {
    OrtSession ortSession = sessionMap.get(key);
    if (ortSession == null) {
      throw new Exception("Model is not loaded " + key);
    }

    RunOptions runOptions = parseRunOptions(options);

    long startTime = System.currentTimeMillis();
    Map<String, OnnxTensor> feed = new HashMap<>();
    Iterator<String> iterator = ortSession.getInputNames().iterator();
    while (iterator.hasNext()) {
      String inputName = iterator.next();

      ReadableMap inputMap = input.getMap(inputName);
      if (inputMap == null) {
        throw new Exception("Can't find input: " + inputName);
      }

      if (inputMap.getType("data") != ReadableType.String) {
        throw new Exception("Non string type of a tensor data is not allowed");
      }

      OnnxTensor onnxTensor = TensorHelper.createInputTensor(inputMap, ortEnvironment);
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
    Result result = null;
    if (requestedOutputs != null) {
      result = ortSession.run(feed, requestedOutputs, runOptions);
    } else {
      result = ortSession.run(feed, runOptions);
    }
    duration = System.currentTimeMillis() - startTime;
    Log.d("Duration", "inference: " + duration);

    startTime = System.currentTimeMillis();
    WritableMap resultMap = TensorHelper.createOutputTensor(result);
    duration = System.currentTimeMillis() - startTime;
    Log.d("Duration", "createOutputTensor: " + duration);

    OnnxValue.close(feed);

    return resultMap;
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
}
