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
@EnabledOnOs(value = OS.WINDOWS)
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
      Assertions.assertFalse(epDevice.getEpName().isEmpty());
      Assertions.assertFalse(epDevice.getEpVendor().isEmpty());
      Map<String, String> metadata = epDevice.getEpMetadata();
      Assertions.assertNotNull(metadata);
      Map<String, String> options = epDevice.getEpOptions();
      Assertions.assertNotNull(options);
      readHardwareDeviceValues(epDevice.getDevice());
    }
  }

  @Test
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
    boolean found = epDevices.stream().anyMatch(a -> a.getEpName().equals(epName));
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
                    .filter(a -> a.getEpName().equals("CPUExecutionProvider"))
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

    // empty options
    runTest.accept(Collections::emptyMap);

    // dummy options
    runTest.accept(() -> Collections.singletonMap("random_key", "value"));
  }

  @Test
  public void GetEpCompatibilityInvalidArgs() {
    Assertions.assertThrows(
        IllegalArgumentException.class,
        () -> ortEnv.getModelCompatibilityForEpDevices(null, "info"));
    Assertions.assertThrows(
        IllegalArgumentException.class,
        () -> ortEnv.getModelCompatibilityForEpDevices(Collections.emptyList(), "info"));
  }

  @Test
  public void GetEpCompatibilitySingleDeviceCpuProvider() throws OrtException {
    List<OrtEpDevice> epDevices = ortEnv.getEpDevices();
    String someInfo = "arbitrary-compat-string";

    // Use CPU device
    OrtEpDevice cpu =
        epDevices.stream()
            .filter(d -> d.getEpName().equals("CPUExecutionProvider"))
            .findFirst()
            .get();
    Assertions.assertNotNull(cpu);
    List<OrtEpDevice> selected = Collections.singletonList(cpu);
    OrtEnvironment.OrtCompiledModelCompatibility status =
        ortEnv.getModelCompatibilityForEpDevices(selected, someInfo);

    // CPU defaults to not applicable in this scenario
    Assertions.assertEquals(OrtEnvironment.OrtCompiledModelCompatibility.EP_NOT_APPLICABLE, status);
  }
}
