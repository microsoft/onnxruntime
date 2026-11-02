/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.providers.webgpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtEpDevice;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtUtil;
import ai.onnxruntime.TensorInfo;
import ai.onnxruntime.TestHelpers;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/** Public Java binding coverage for the separately loaded WebGPU plugin. */
@EnabledIfSystemProperty(named = "WEBGPU_PLUGIN_LIBRARY", matches = ".+")
public class WebGpuPluginTest {
  @Test
  public void inferenceUsesWebGpu() throws OrtException {
    OrtEnvironment env = OrtEnvironment.getEnvironment();
    String registrationName = "WebGPUJavaTest";
    env.registerExecutionProviderLibrary(
        registrationName, System.getProperty("WEBGPU_PLUGIN_LIBRARY"));
    try {
      List<OrtEpDevice> devices =
          env.getEpDevices().stream()
              .filter(device -> device.getEpName().equals("WebGpuExecutionProvider"))
              .limit(1)
              .collect(Collectors.toList());
      assertFalse(devices.isEmpty(), "The WebGPU plugin must expose a device");
      try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
        options.addExecutionProvider(devices, Collections.emptyMap());
        options.addConfigEntry("session.disable_cpu_ep_fallback", "1");
        String model = TestHelpers.getResourcePath("/squeezenet.onnx").toString();
        try (OrtSession session = env.createSession(model, options)) {
          NodeInfo input = session.getInputInfo().values().iterator().next();
          float[] data = TestHelpers.loadTensorFromFile(TestHelpers.getResourcePath("/bench.in"));
          float[] expected =
              TestHelpers.loadTensorFromFile(TestHelpers.getResourcePath("/bench.expected_out"));
          Object tensor = OrtUtil.reshape(data, ((TensorInfo) input.getInfo()).getShape());
          try (OnnxTensor value = OnnxTensor.createTensor(env, tensor);
              OrtSession.Result result =
                  session.run(Collections.singletonMap(input.getName(), value))) {
            assertArrayEquals(expected, TestHelpers.flattenFloat(result.get(0).getValue()), 1e-5f);
          }
        }
      }
    } finally {
      env.unregisterExecutionProviderLibrary(registrationName);
    }
  }
}
