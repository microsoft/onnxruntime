/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.providers;

import static ai.onnxruntime.TestHelpers.getResourcePath;
import static ai.onnxruntime.TestHelpers.loadTensorFromFile;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.onnxruntime.InferenceTest;
import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtProvider;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtUtil;
import ai.onnxruntime.TensorInfo;
import ai.onnxruntime.TestHelpers;
import java.nio.file.Path;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

public class ProviderOptionsTest {
  private static final OrtEnvironment env = OrtEnvironment.getEnvironment();

  @Test
  @EnabledIfSystemProperty(named = "USE_CUDA", matches = "1")
  public void testCUDAOptions() throws OrtException {
    // Test standard options
    OrtCUDAProviderOptions cudaOpts = new OrtCUDAProviderOptions(0);
    cudaOpts.add("gpu_mem_limit", "" + (512 * 1024 * 1024));
    OrtSession.SessionOptions sessionOpts = new OrtSession.SessionOptions();
    sessionOpts.addCUDA(cudaOpts);
    runProvider(OrtProvider.CUDA, sessionOpts);

    // Test invalid device num throws
    assertThrows(IllegalArgumentException.class, () -> new OrtCUDAProviderOptions(-1));

    // Test invalid key name throws
    OrtCUDAProviderOptions invalidKeyOpts = new OrtCUDAProviderOptions(0);
    assertThrows(
        OrtException.class, () -> invalidKeyOpts.add("not_a_real_provider_option", "not a number"));
    // Test invalid value throws
    OrtCUDAProviderOptions invalidValueOpts = new OrtCUDAProviderOptions(0);
    assertThrows(OrtException.class, () -> invalidValueOpts.add("gpu_mem_limit", "not a number"));
  }

  @Test
  @EnabledIfSystemProperty(named = "USE_TENSORRT", matches = "1")
  public void testTensorRT() throws OrtException {
    // Test standard options
    OrtTensorRTProviderOptions cudaOpts = new OrtTensorRTProviderOptions(0);
    cudaOpts.add("trt_max_workspace_size", "" + (512 * 1024 * 1024));
    OrtSession.SessionOptions sessionOpts = new OrtSession.SessionOptions();
    sessionOpts.addTensorrt(cudaOpts);
    runProvider(OrtProvider.TENSOR_RT, sessionOpts);

    // Test invalid device num throws
    assertThrows(IllegalArgumentException.class, () -> new OrtTensorRTProviderOptions(-1));

    // Test invalid key name throws
    OrtTensorRTProviderOptions invalidKeyOpts = new OrtTensorRTProviderOptions(0);
    assertThrows(
        OrtException.class, () -> invalidKeyOpts.add("not_a_real_provider_option", "not a number"));
    // Test invalid value throws
    OrtTensorRTProviderOptions invalidValueOpts = new OrtTensorRTProviderOptions(0);
    assertThrows(
        OrtException.class, () -> invalidValueOpts.add("trt_max_workspace_size", "not a number"));
  }

  private static void runProvider(OrtProvider provider, OrtSession.SessionOptions options)
      throws OrtException {
    EnumSet<OrtProvider> providers = OrtEnvironment.getAvailableProviders();
    assertTrue(providers.size() > 1);
    assertTrue(providers.contains(OrtProvider.CPU));
    assertTrue(providers.contains(provider));
    InferenceTest.SqueezeNetTuple tuple = openSessionSqueezeNet(options);
    try (OrtSession session = tuple.session) {
      float[] inputData = tuple.inputData;
      float[] expectedOutput = tuple.outputData;
      NodeInfo inputMeta = session.getInputInfo().values().iterator().next();
      Map<String, OnnxTensor> container = new HashMap<>();
      long[] inputShape = ((TensorInfo) inputMeta.getInfo()).getShape();
      Object tensor = OrtUtil.reshape(inputData, inputShape);
      container.put(inputMeta.getName(), OnnxTensor.createTensor(env, tensor));
      try (OrtSession.Result result = session.run(container)) {
        OnnxValue resultTensor = result.get(0);
        float[] resultArray = TestHelpers.flattenFloat(resultTensor.getValue());
        assertEquals(expectedOutput.length, resultArray.length);
        assertArrayEquals(expectedOutput, resultArray, 1e-5f);
      } catch (OrtException e) {
        throw new IllegalStateException("Failed to execute a scoring operation", e);
      }
      OnnxValue.close(container.values());
    }
  }

  /**
   * Loads the squeezenet model into a session using the supplied session options.
   *
   * @param options The session options.
   * @return The squeezenet session, input and output.
   * @throws OrtException If the native code failed.
   */
  private static InferenceTest.SqueezeNetTuple openSessionSqueezeNet(
      OrtSession.SessionOptions options) throws OrtException {
    Path squeezeNet = getResourcePath("/squeezenet.onnx");
    String modelPath = squeezeNet.toString();
    OrtSession session = env.createSession(modelPath, options);
    float[] inputData = loadTensorFromFile(getResourcePath("/bench.in"));
    float[] expectedOutput = loadTensorFromFile(getResourcePath("/bench.expected_out"));
    return new InferenceTest.SqueezeNetTuple(session, inputData, expectedOutput);
  }
}
