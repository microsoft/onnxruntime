/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.logging.Level;
import java.util.logging.Logger;

/** Static loader for the JNI binding. */
final class OnnxRuntime {
  private static final Logger logger = Logger.getLogger(OnnxRuntime.class.getName());

  // The initial release of the ORT API.
  private static final int ORT_API_VERSION_1 = 1;

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
    load("onnxruntime");
    load("onnxruntime4j_jni");
    ortApiHandle = initialiseAPIBase(ORT_API_VERSION_1);
    loaded = true;
  }

  /**
   * Finds a named library in lib path. Falls back to classpath resources. In that case, it copies resource
   * to temporary file on the filesystem to be loaded using {@link System#load}.
   *
   * @param library The bare name of the library.
   * @throws IOException If the file failed to read or write.
   */
  private static void load(String library) throws IOException {
    try {
      logger.log(Level.FINE, "Attempting to load native library '" + library + "' from lib path");
      System.loadLibrary(library);
      logger.log(Level.FINE, "Loaded native library '" + library + "' from lib path");
      return;
    } catch (Exception | Error e) {
      logger.log(Level.FINE, "Native library '" + library + "' failed to load from lib path", e);
    }
    // generate a platform specific library name
    // replace Mac's jnilib extension to dylib
    String libraryFileName = System.mapLibraryName(library).replace("jnilib", "dylib");
    String resourcePath = "/ai/onnxruntime/native/" + libraryFileName;
    logger.log(
        Level.FINE,
        "Attempting to load native library '" + library + "' from resource path " + resourcePath);
    InputStream is = OnnxRuntime.class.getResourceAsStream(resourcePath);
    if (is == null) {
      logger.log(Level.FINE, "Native library '" + library + "' was not found in resources");
      return;
    }

    File temp = File.createTempFile("javaload_", libraryFileName);
    try {
      byte[] buffer = new byte[1024];
      int readBytes;
      try (FileOutputStream os = new FileOutputStream(temp)) {
        while ((readBytes = is.read(buffer)) != -1) {
          os.write(buffer, 0, readBytes);
        }
      }
      System.load(temp.getAbsolutePath());
      logger.log(Level.FINE, "Loaded native library '" + library + "' from resources");
    } finally {
      temp.delete();
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
