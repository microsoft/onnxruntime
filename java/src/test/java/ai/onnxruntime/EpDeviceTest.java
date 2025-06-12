/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import ai.onnxruntime.OrtHardwareDevice.OrtHardwareDeviceType;
import ai.onnxruntime.OrtSession.SessionOptions;
import java.io.File;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

/** Tests for {@link OrtEpDevice} and {@link OrtHardwareDevice}. */
public class EpDeviceTest {
  private final OrtEnvironment ortEnv = OrtEnvironment.getEnvironment();

  private void readHardwareDeviceValues(OrtHardwareDevice device) {
    OrtHardwareDeviceType type = device.getType();

    Assertions.assertTrue(
        type == OrtHardwareDeviceType.CPU
            || type == OrtHardwareDeviceType.GPU
            || type == OrtHardwareDeviceType.NPU);

    if (type == OrtHardwareDeviceType.CPU) {
      Assertions.assertFalse(device.getVendor().isEmpty());
    } else {
      Assertions.assertTrue(device.getVendorId() != 0);
      Assertions.assertTrue(device.getDeviceId() != 0);
    }

    Map<String, String> metadata = device.getMetadata();
    Assertions.assertNotNull(metadata);
    for (Map.Entry<String, String> kvp : metadata.entrySet()) {
      Assertions.assertFalse(kvp.getKey().isEmpty());
    }
  }

  @Test
  public void getEpDevices() throws OrtException {
    List<OrtEpDevice> epDevices = ortEnv.getEpDevices();
    Assertions.assertNotNull(epDevices);
    Assertions.assertFalse(epDevices.isEmpty());
    for (OrtEpDevice epDevice : epDevices) {
      Assertions.assertFalse(epDevice.getName().isEmpty());
      Assertions.assertFalse(epDevice.getVendor().isEmpty());
      Map<String, String> metadata = epDevice.getMetadata();
      Assertions.assertNotNull(metadata);
      Map<String, String> options = epDevice.getOptions();
      Assertions.assertNotNull(options);
      readHardwareDeviceValues(epDevice.getDevice());
    }
  }

  @Test
  @EnabledOnOs(value = OS.WINDOWS)
  public void registerUnregisterLibrary() throws OrtException {
    String libFullPath = TestHelpers.getResourcePath("/example_plugin_ep.dll").toString();
    Assertions.assertTrue(
        new File(libFullPath).exists(), "Expected lib " + libFullPath + " does not exist.");

    // example plugin ep uses the registration name as the ep name
    String epName = "java_ep";

    // register. shouldn't throw
    ortEnv.registerExecutionProviderLibrary(epName, libFullPath);

    // check OrtEpDevice was found
    List<OrtEpDevice> epDevices = ortEnv.getEpDevices();
    boolean found = epDevices.stream().anyMatch(a -> a.getName().equals(epName));
    Assertions.assertTrue(found);

    // unregister
    ortEnv.unregisterExecutionProviderLibrary(epName);
  }

  @Test
  public void appendToSessionOptionsV2() {
    Consumer<Supplier<Map<String, String>>> runTest =
        (Supplier<Map<String, String>> options) -> {
          try (SessionOptions sessionOptions = new SessionOptions()) {
            sessionOptions.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE);

            List<OrtEpDevice> epDevices = ortEnv.getEpDevices();

            // cpu ep ignores the provider options so we can use any value in epOptions and it won't
            // break.
            List<OrtEpDevice> selectedEpDevices =
                epDevices.stream()
                    .filter(a -> a.getName().equals("CPUExecutionProvider"))
                    .collect(Collectors.toList());

            Map<String, String> epOptions = options.get();
            sessionOptions.addExecutionProvider(selectedEpDevices, epOptions);

            Path model = TestHelpers.getResourcePath("/squeezenet.onnx");
            String modelPath = model.toString();

            // session should load successfully
            try (OrtSession session = ortEnv.createSession(modelPath, sessionOptions)) {
              Assertions.assertNotNull(session);
            }
          } catch (OrtException e) {
            throw new RuntimeException(e);
          }
        };

    runTest.accept(() -> null);

    // empty options
    runTest.accept(Collections::emptyMap);

    // dummy options
    runTest.accept(() -> Collections.singletonMap("random_key", "value"));
  }
}
