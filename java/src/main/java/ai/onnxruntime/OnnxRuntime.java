/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Static loader for the JNI binding. No public API, but called from various classes in this package
 * to ensure shared libraries are properly loaded.
 */
final class OnnxRuntime {
  private static final Logger logger = Logger.getLogger(OnnxRuntime.class.getName());

  // The initial release of the ORT API.
  private static final int ORT_API_VERSION_1 = 1;
  // Post 1.0 builds of the ORT API.
  private static final int ORT_API_VERSION_2 = 2;

  /** The short name of the ONNX runtime shared library */
  static final String ONNXRUNTIME_LIBRARY_NAME = "onnxruntime";
  /** The short name of the ONNX runtime JNI shared library */
  static final String ONNXRUNTIME_JNI_LIBRARY_NAME = "onnxruntime4j_jni";

  private static boolean loaded = false;

  /** The API handle. */
  static long ortApiHandle;

  private OnnxRuntime() {}

  /**
   * Loads the native C library.
   *
   * @throws IOException If it can't write to disk to copy out the library from the jar file.
   */
  static synchronized void init() throws IOException {
    if (loaded) {
      return;
    }
    Path tempDirectory = isAndroid() ? null : Files.createTempDirectory("onnxruntime-java");
    try {
      load(tempDirectory, ONNXRUNTIME_LIBRARY_NAME);
      load(tempDirectory, ONNXRUNTIME_JNI_LIBRARY_NAME);
      ortApiHandle = initialiseAPIBase(ORT_API_VERSION_2);
      loaded = true;
    } finally {
      if (!isAndroid()) {
        cleanUp(tempDirectory.toFile());
      }
    }
  }

  /**
   * Attempt to remove a file and then mark for delete on exit if it cannot be deleted at this point
   * in time.
   *
   * @param file The file to remove.
   */
  private static void cleanUp(File file) {
    if (!file.exists()) {
      return;
    }
    logger.log(Level.FINE, "Deleting " + file);
    if (!file.delete()) {
      logger.log(Level.FINE, "Deleting " + file + " on exit");
      file.deleteOnExit();
    }
  }

  private static boolean isAndroid() {
    try {
      Class.forName("android.app.Activity");
      return true;
    } catch (ClassNotFoundException e) {
      return false;
    }
  }

  /**
   * Load a shared library by name.
   *
   * @param tempDirectory The temp directory to write the library resource to.
   * @param library The bare name of the library.
   * @throws IOException If the file failed to read or write.
   */
  private static void load(Path tempDirectory, String library) throws IOException {
    // On Android, we simply use System.loadLibrary
    if (isAndroid()) {
      System.loadLibrary("onnxruntime4j_jni");
      return;
    }

    // 1) The user may skip loading of this library:
    String skip = System.getProperty("onnxruntime.native." + library + ".skip");
    if (Boolean.TRUE.toString().equalsIgnoreCase(skip)) {
      logger.log(Level.FINE, "Skipping load of native library '" + library + "'");
      return;
    }

    // 2) The user may explicitly specify the path to their shared library:
    String libraryPathProperty = System.getProperty("onnxruntime.native." + library + ".path");
    if (libraryPathProperty != null) {
      logger.log(
          Level.FINE,
          "Attempting to load native library '"
              + library
              + "' from specified path: "
              + libraryPathProperty);
      File libraryFile = new File(libraryPathProperty);
      String libraryFilePath = libraryFile.getAbsolutePath();
      if (!libraryFile.exists()) {
        throw new IOException("Native library '" + library + "' not found at " + libraryFilePath);
      }
      System.load(libraryFilePath);
      logger.log(Level.FINE, "Loaded native library '" + library + "' from specified path");
      return;
    }

    // 3) try loading from resources or library path:
    // generate a platform specific library name
    // replace Mac's jnilib extension to dylib
    String libraryFileName = System.mapLibraryName(library).replace("jnilib", "dylib");
    String resourcePath = "/ai/onnxruntime/native/" + libraryFileName;
    File tempFile = tempDirectory.resolve(libraryFileName).toFile();
    try (InputStream is = OnnxRuntime.class.getResourceAsStream(resourcePath)) {
      if (is == null) {
        // 3a) Not found in resources, load from library path
        logger.log(
            Level.FINE, "Attempting to load native library '" + library + "' from library path");
        System.loadLibrary(library);
        logger.log(Level.FINE, "Loaded native library '" + library + "' from library path");
      } else {
        // 3b) Found in resources, load via temporary file
        logger.log(
            Level.FINE,
            "Attempting to load native library '"
                + library
                + "' from resource path "
                + resourcePath
                + " copying to "
                + tempFile);
        byte[] buffer = new byte[1024];
        int readBytes;
        try (FileOutputStream os = new FileOutputStream(tempFile)) {
          while ((readBytes = is.read(buffer)) != -1) {
            os.write(buffer, 0, readBytes);
          }
        }
        System.load(tempFile.getAbsolutePath());
        logger.log(Level.FINE, "Loaded native library '" + library + "' from resource path");
      }
    } finally {
      cleanUp(tempFile);
    }
  }

  /**
   * Get a reference to the API struct.
   *
   * @param apiVersionNumber The API version to use.
   * @return A pointer to the API struct.
   */
  private static native long initialiseAPIBase(int apiVersionNumber);
}
