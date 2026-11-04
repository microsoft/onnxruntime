/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import ai.onnxruntime.OrtSession.SessionOptions;
import java.io.IOException;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Fault-injection tests for EPContext callback ownership and native thread cleanup. */
public class EpContextDataCallbackTest {
  @BeforeAll
  public static void loadTestLibrary() throws IOException {
    OnnxRuntime.init();
    boolean isLinuxAarch64 =
        "Linux".equals(System.getProperty("os.name"))
            && "aarch64".equals(System.getProperty("os.arch"));
    String libraryPath = isLinuxAarch64 ? "/linux-aarch64/" : "/";
    String libName = libraryPath + System.mapLibraryName("onnxruntime4j_jni_test");
    Assumptions.assumeTrue(
        EpContextDataCallbackTest.class.getResource(libName) != null,
        "The native test library '"
            + libName
            + "' is not on the classpath; skipping as it is not bundled in all build"
            + " configurations (e.g. packaged jar testing).");
    System.load(TestHelpers.getResourcePath(libName).toString());
  }

  @Test
  public void successfulReadDetachesNativeWorker() throws OrtException {
    exerciseReadCallbackAllocator(OnnxRuntime.ortApiHandle, name -> new byte[] {1, 2, 3}, 0);
  }

  @Test
  public void nullAllocationDetachesNativeWorker() throws OrtException {
    exerciseReadCallbackAllocator(OnnxRuntime.ortApiHandle, name -> new byte[] {1, 2, 3}, 1);
  }

  @Test
  public void throwingAllocationDetachesNativeWorker() throws OrtException {
    Assumptions.assumeTrue(
        supportsAllocatorExceptions(), "The native build disables C++ exceptions.");
    exerciseReadCallbackAllocator(OnnxRuntime.ortApiHandle, name -> new byte[] {1, 2, 3}, 2);
  }

  @Test
  public void failedHolderAllocationDoesNotInstallCallback() throws OrtException {
    try (SessionOptions options = new SessionOptions()) {
      assertHolderAllocationFailure(options);
      Assertions.assertNull(options.getEpContextDataReadCallbackRegistration());

      options.setEpContextDataReadCallback(name -> new byte[0], 1024);
      SessionOptions.EpContextDataReadCallbackRegistration registration =
          options.getEpContextDataReadCallbackRegistration();
      Assertions.assertEquals(1, registration.getReferenceCount());
      options.clearEpContextDataReadCallback();
      Assertions.assertEquals(0, registration.getReferenceCount());
    }
  }

  @Test
  public void failedHolderAllocationPreservesPreviousRegistration() throws OrtException {
    SessionOptions.EpContextDataReadCallbackRegistration registration;
    try (SessionOptions options = new SessionOptions()) {
      options.setEpContextDataReadCallback(name -> new byte[0], 1024);
      registration = options.getEpContextDataReadCallbackRegistration();

      assertHolderAllocationFailure(options);
      Assertions.assertSame(registration, options.getEpContextDataReadCallbackRegistration());
      Assertions.assertEquals(1, registration.getReferenceCount());
      try (SessionOptions.NativeSessionOptionsSnapshot snapshot = options.createSnapshot()) {
        Assertions.assertSame(registration, snapshot.getEpContextDataReadCallbackRegistration());
        Assertions.assertEquals(2, registration.getReferenceCount());
      }
      Assertions.assertEquals(1, registration.getReferenceCount());
    }
    Assertions.assertEquals(0, registration.getReferenceCount());
  }

  private static void assertHolderAllocationFailure(SessionOptions options) {
    int[] counts = new int[3];
    Assertions.assertThrows(
        OutOfMemoryError.class,
        () ->
            failReadCallbackHolderAllocation(
                OnnxRuntime.ortApiHandle, options.getNativeHandle(), name -> new byte[0], counts));
    // Global references created/released, followed by native installation attempts.
    Assertions.assertArrayEquals(new int[] {1, 1, 0}, counts);
  }

  private static native boolean supportsAllocatorExceptions();

  private static native void exerciseReadCallbackAllocator(
      long apiHandle, SessionOptions.EpContextDataReadCallback callback, int allocationMode)
      throws OrtException;

  private static native void failReadCallbackHolderAllocation(
      long apiHandle,
      long optionsHandle,
      SessionOptions.EpContextDataReadCallback callback,
      int[] counts)
      throws OrtException;
}
