/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import static ai.onnxruntime.TestHelpers.getResourcePath;
import static ai.onnxruntime.TestHelpers.loadTensorFromFile;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/** This test is in a separate class to ensure it is run in a clean JVM. */
public class EnvironmentThreadPoolTest {

  @EnabledIfSystemProperty(named = "JAVA_FULL_TEST", matches = "1")
  @Test
  public void environmentThreadPoolTest() throws OrtException {
    Path squeezeNet = getResourcePath("/squeezenet.onnx");
    String modelPath = squeezeNet.toString();
    float[] inputData = loadTensorFromFile(getResourcePath("/bench.in"));
    float[] expectedOutput = loadTensorFromFile(getResourcePath("/bench.expected_out"));
    Map<String, OnnxTensor> container = new HashMap<>();

    OrtEnvironment.ThreadingOptions threadOpts = new OrtEnvironment.ThreadingOptions();
    threadOpts.setGlobalInterOpNumThreads(2);
    threadOpts.setGlobalIntraOpNumThreads(2);
    threadOpts.setGlobalDenormalAsZero();
    threadOpts.setGlobalSpinControl(true);
    OrtEnvironment env =
        OrtEnvironment.getEnvironment(
            OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL, "environmentThreadPoolTest", threadOpts);
    try (OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        OrtSession.SessionOptions disableThreadOptions = new OrtSession.SessionOptions()) {
      disableThreadOptions.disablePerSessionThreads();

      // Check that the regular session executes
      try (OrtSession session = env.createSession(modelPath, options)) {
        NodeInfo inputMeta = session.getInputInfo().values().iterator().next();
        long[] inputShape = ((TensorInfo) inputMeta.getInfo()).shape;
        Object tensorData = OrtUtil.reshape(inputData, inputShape);
        OnnxTensor tensor = OnnxTensor.createTensor(env, tensorData);
        container.put(inputMeta.getName(), tensor);
        try (OrtSession.Result result = session.run(container)) {
          OnnxValue resultTensor = result.get(0);
          float[] resultArray = TestHelpers.flattenFloat(resultTensor.getValue());
          assertEquals(expectedOutput.length, resultArray.length);
          assertArrayEquals(expectedOutput, resultArray, 1e-6f);
        }
        container.clear();
        tensor.close();
      }

      // Check that the session using the env thread pool executes
      try (OrtSession session = env.createSession(modelPath, disableThreadOptions)) {
        NodeInfo inputMeta = session.getInputInfo().values().iterator().next();
        long[] inputShape = ((TensorInfo) inputMeta.getInfo()).shape;
        Object tensorData = OrtUtil.reshape(inputData, inputShape);
        OnnxTensor tensor = OnnxTensor.createTensor(env, tensorData);
        container.put(inputMeta.getName(), tensor);
        try (OrtSession.Result result = session.run(container)) {
          OnnxValue resultTensor = result.get(0);
          float[] resultArray = TestHelpers.flattenFloat(resultTensor.getValue());
          assertEquals(expectedOutput.length, resultArray.length);
          assertArrayEquals(expectedOutput, resultArray, 1e-6f);
        }
        container.clear();
        tensor.close();
      }
    }

    try {
      OrtEnvironment newEnv =
          OrtEnvironment.getEnvironment(
              OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL, "fail", threadOpts);
      // fail as we can't recreate environments with different threading options
      fail("Should have thrown IllegalStateException");
    } catch (IllegalStateException e) {
      // pass
    }

    threadOpts.close();
  }
}
