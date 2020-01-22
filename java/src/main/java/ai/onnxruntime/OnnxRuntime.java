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

/**
   * Static loader for the JNI binding. No public API, but called from various classes in this package to ensure shared libraries are properly loaded.
   * There are two shared libraries required: <code>onnxruntime</code> and <code>onnxruntime4j_jni</code>. This is the loading logic in order:
   * <ol>
   * <li>The user may signal to skip loading of a shared library using a property in the form <code>onnxruntime.native.LIB_NAME.skip</code> with a value of <code>true</code>. This means the user has decided to load the library by some other means.
   * <li>The user may specify an explicit location of the shared library file using a property in the form <code>onnxruntime.native.LIB_NAME.path</code>. This uses {@link java.lang.System#load}.
   * <li>The shared library is autodiscovered:<ol>
   * <li>If the shared library is present in the classpath resources, load using {@link java.lang.System#load} via a temporary file.
   * <li>If the shared library is not present in the classpath resources, then load using {@link java.lang.System#loadLibrary}, which usually looks elsewhere on the filesystem for the library.
   * The semantics and behavior of that method are system/JVM dependent.
   * Typically, the <code>java.library.path</code> property is used to specify the location of native libraries.
   * </ol></ol>
*/
public final class OnnxRuntime {
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
   * Load a shared library by name.
   *
   * @param library The bare name of the library.
   * @throws IOException If the file failed to read or write.
   */
  private static void load(String library) throws IOException {
    // 1) The user may skip loading of this library:
    String skip = System.getProperty("onnxruntime.native." + library + ".skip");
    if (Boolean.TRUE.toString().equalsIgnoreCase(skip)) {
      logger.log(Level.FINE, "Skipping load of native library '" + library + "'");
      return;
    }

    // 2) The user may explicitly specify the path to their shared library:
    String libraryPathProperty = System.getProperty("onnxruntime.native." + library + ".path");
    if (libraryPathProperty != null) {
      logger.log(Level.FINE, "Attempting to load native library '" + library + "' from specified path: " + libraryPathProperty);
      File libraryFile = new File(libraryPathProperty);
      if (!libraryFile.exists()) {
        throw new IOException("Native library '" + library + "' not found at "+ libraryPathProperty);
      }
      System.load(libraryFile.getAbsolutePath());
      logger.log(Level.FINE, "Loaded native library '" + library + "' from specified path");
      return;
    }

    // 3) try loading from resources or library path:
    // generate a platform specific library name
    // replace Mac's jnilib extension to dylib
    String libraryFileName = System.mapLibraryName(library).replace("jnilib", "dylib");
    String resourcePath = "/ai/onnxruntime/native/" + libraryFileName;
    InputStream is = OnnxRuntime.class.getResourceAsStream(resourcePath);
    // check resources for the library
    if (is == null) {
      // 3a) Not found in resources, load from library path
      logger.log(Level.FINE, "Attempting to load native library '" + library + "' from lib path");
      System.loadLibrary(library);
      logger.log(Level.FINE, "Loaded native library '" + library + "' from lib path");
    } else {
      // 3b) Found in resources, load via temporary file
      logger.log(
          Level.FINE,
          "Attempting to load native library '" + library + "' from resource path " + resourcePath);
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
        // attempt to delete, however the file may still be open on some platforms, so the delete may fail
        if(!temp.delete()) {
          // mark the file to be deleted at shutdown
          temp.deleteOnExit();
        }
      }
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
