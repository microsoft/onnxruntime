// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public final class RuntimeSmokeTest {
  public static void main(String[] args) throws Exception {
    boolean cuda = args.length == 2 && "--cuda".equals(args[1]);
    if (args.length != 1 && args.length != 3 && !cuda) {
      throw new IllegalArgumentException(
          "Pass a model path, optionally --cuda or the runtime and JNI library paths.");
    }
    if (args.length == 3) {
      System.setProperty(
          "onnxruntime.native.onnxruntime.path", Paths.get(args[1]).toAbsolutePath().toString());
      System.setProperty(
          "onnxruntime.native.onnxruntime4j_jni.path", Paths.get(args[2]).toAbsolutePath().toString());
    }
    try (OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = sessionOptions(cuda);
        OrtSession session = env.createSession(args[0], options);
        OnnxTensor a = OnnxTensor.createTensor(env, new float[][] {{1, 2, 3}, {4, 5, 6}});
        OnnxTensor b = OnnxTensor.createTensor(env, new float[][] {{10, 20, 30}, {40, 50, 60}})) {
      Map<String, OnnxTensor> inputs = new HashMap<>();
      inputs.put("A", a);
      inputs.put("B", b);
      try (OrtSession.Result result = session.run(inputs)) {
        float[][] expected = {{11, 22, 33}, {44, 55, 66}};
        if (!Arrays.deepEquals(expected, (float[][]) result.get(0).getValue())) {
          throw new AssertionError("Java Add inference returned incorrect values or shape.");
        }
      }
      System.out.println("Java/JNI Add inference passed.");
    }
  }

  private static OrtSession.SessionOptions sessionOptions(boolean cuda)
      throws ai.onnxruntime.OrtException {
    OrtSession.SessionOptions options = new OrtSession.SessionOptions();
    try {
      if (cuda) {
        options.addCUDA();
        options.addConfigEntry("session.disable_cpu_ep_fallback", "1");
      }
      return options;
    } catch (ai.onnxruntime.OrtException exception) {
      options.close();
      throw exception;
    }
  }
}
