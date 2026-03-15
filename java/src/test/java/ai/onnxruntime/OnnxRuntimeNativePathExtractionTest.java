/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assumptions;

public class OnnxRuntimeNativePathExtractionTest {
  private static final String TEST_PROVIDER_LIBRARY = "onnxruntime_providers_test_native_path";

  private String previousNativePath;
  private String previousLibraryDirPathProperty;
  private Path previousTempDirectory;

  private Path nativePathDir;
  private Path tempExtractionDir;

  @BeforeEach
  @SuppressWarnings("unchecked")
  void setUp() throws Exception {
    previousNativePath = System.getProperty(OnnxRuntime.ONNXRUNTIME_NATIVE_PATH);
    previousLibraryDirPathProperty = (String) getPrivateStaticField("libraryDirPathProperty");
    previousTempDirectory = (Path) getPrivateStaticField("tempDirectory");
    ((Set<String>) getPrivateStaticField("extractedSharedProviders")).remove(TEST_PROVIDER_LIBRARY);
    OnnxRuntime.setExtractFromResourcesHook(null);
  }

  @AfterEach
  @SuppressWarnings("unchecked")
  void tearDown() throws Exception {
    OnnxRuntime.setExtractFromResourcesHook(null);
    if (previousNativePath == null) {
      System.clearProperty(OnnxRuntime.ONNXRUNTIME_NATIVE_PATH);
    } else {
      System.setProperty(OnnxRuntime.ONNXRUNTIME_NATIVE_PATH, previousNativePath);
    }
    setPrivateStaticField("libraryDirPathProperty", previousLibraryDirPathProperty);
    setPrivateStaticField("tempDirectory", previousTempDirectory);
    ((Set<String>) getPrivateStaticField("extractedSharedProviders")).remove(TEST_PROVIDER_LIBRARY);
    if (nativePathDir != null && Files.exists(nativePathDir)) {
      TestHelpers.deleteDirectoryTree(nativePathDir);
    }
    if (tempExtractionDir != null && Files.exists(tempExtractionDir)) {
      TestHelpers.deleteDirectoryTree(tempExtractionDir);
    }
  }

  @Test
  void providerAlreadyInNativePathShouldNotAttemptResourceExtraction() throws Exception {
    Assumptions.assumeFalse(OnnxRuntime.isAndroid(), "Provider extraction is not used on Android.");

    nativePathDir = Files.createTempDirectory("ort-native-path");
    tempExtractionDir = Files.createTempDirectory("ort-extract-temp");

    String libraryFileName = System.mapLibraryName(TEST_PROVIDER_LIBRARY).replace("jnilib", "dylib");
    Files.write(nativePathDir.resolve(libraryFileName), new byte[] {1});

    System.setProperty(OnnxRuntime.ONNXRUNTIME_NATIVE_PATH, nativePathDir.toString());
    setPrivateStaticField("libraryDirPathProperty", nativePathDir.toString());
    setPrivateStaticField("tempDirectory", tempExtractionDir);

    AtomicInteger extractionAttempts = new AtomicInteger(0);
    OnnxRuntime.setExtractFromResourcesHook(
        library -> {
          if (TEST_PROVIDER_LIBRARY.equals(library)) {
            extractionAttempts.incrementAndGet();
          }
        });

    boolean libraryReady = OnnxRuntime.extractProviderLibrary(TEST_PROVIDER_LIBRARY);

    assertTrue(libraryReady, "Sanity check: provider library should be found in native path.");
    // Regression expectation for issue #27655:
    // when onnxruntime.native.path already contains the requested provider library,
    // extraction from JAR/resources should not be attempted.
    assertEquals(
        0,
        extractionAttempts.get(),
        "Bug #27655: extraction from resources was attempted even though the provider library "
            + "already exists in onnxruntime.native.path.");
  }

  private static Object getPrivateStaticField(String fieldName) throws Exception {
    Field field = OnnxRuntime.class.getDeclaredField(fieldName);
    field.setAccessible(true);
    return field.get(null);
  }

  private static void setPrivateStaticField(String fieldName, Object value) throws Exception {
    Field field = OnnxRuntime.class.getDeclaredField(fieldName);
    field.setAccessible(true);
    field.set(null, value);
  }
}
