/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Collections;
import java.util.EnumSet;
import java.util.Map;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

public class EnvironmentTest {
  private static final OrtEnvironment env = TestHelpers.getOrtEnvironment();

  @Test
  public void environmentTest() {
    // Checks that the environment instance is the same.
    OrtEnvironment otherEnv = OrtEnvironment.getEnvironment();
    assertSame(env, otherEnv);
    TestHelpers.quietLogger(OrtEnvironment.class);
    otherEnv = OrtEnvironment.getEnvironment("test-name");
    TestHelpers.loudLogger(OrtEnvironment.class);
    assertSame(env, otherEnv);
  }

  @Test
  public void testVersion() {
    String version = env.getVersion();
    assertFalse(version.isEmpty());
  }

  @Test
  public void testProviders() {
    EnumSet<OrtProvider> providers = OrtEnvironment.getAvailableProviders();
    int providersSize = providers.size();
    assertTrue(providersSize > 0);
    assertTrue(providers.contains(OrtProvider.CPU));

    // Check that the providers are a copy of the original, note this does not enable the DNNL
    // provider
    providers.add(OrtProvider.DNNL);
    assertEquals(providersSize, OrtEnvironment.getAvailableProviders().size());
  }

  @Test
  public void testEmptyDefaultAllocatorStatistics() throws OrtException {
    Map<String, String> initialStats = env.defaultAllocator.getStats();
    assertTrue(initialStats.isEmpty());
  }

  // Disabled as the default allocator doesn't collect statistics.
  @Disabled
  @Test
  public void testDefaultAllocatorStatistics() throws OrtException {
    Map<String, String> initialStats = env.defaultAllocator.getStats();
    Map<String, String> midSessionStats;

    // Run a session to check that it uses some memory
    String modelPath = TestHelpers.getResourcePath("/test_types_FLOAT.pb").toString();
    try (OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        OrtSession session = env.createSession(modelPath, options)) {
      String inputName = session.getInputNames().iterator().next();
      float[][] input = new float[][] {{1.0f, 2.0f, -3.0f, Float.MIN_VALUE, Float.MAX_VALUE}};
      OnnxTensor ov = OnnxTensor.createTensor(env, input);
      Map<String, OnnxTensor> container = Collections.singletonMap(inputName, ov);
      try (OrtSession.Result res = session.run(container)) {
        // res should have allocated memory using the default allocator
        midSessionStats = env.defaultAllocator.getStats();
      }
      OnnxValue.close(container);
    }

    Map<String, String> sessionClosedStats = env.defaultAllocator.getStats();

    long initialMemory = Long.parseLong(initialStats.get("InUse"));
    long midSessionMemory = Long.parseLong(midSessionStats.get("InUse"));
    long sessionClosedMemory = Long.parseLong(sessionClosedStats.get("InUse"));
    Assertions.assertTrue(midSessionMemory > initialMemory);
    Assertions.assertTrue(midSessionMemory > sessionClosedMemory);
  }
}
