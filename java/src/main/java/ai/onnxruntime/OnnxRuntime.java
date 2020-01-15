/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Static loader for the JNI binding.
 */
final class OnnxRuntime {
    private static final Logger logger = Logger.getLogger(OnnxRuntime.class.getName());

    // The initial release of the ORT API.
    private static final int ORT_API_VERSION_1 = 1;

    /**
     * Turns on debug logging during library loading.
     */
    public static final String LIBRARY_LOAD_LOGGING = "ORT_LOAD_LOGGING";

    /**
     * Specifies that the libraries should be loaded from java.library.path rather than unzipped from the jar file.
     */
    public static final String LOAD_LIBRARY_PATH = "ORT_LOAD_FROM_LIBRARY_PATH";

    private static boolean loaded = false;

    /**
     * The API handle.
     */
    static long ortApiHandle;

    /**
     * Library names stored in the jar.
     */
    private static final List<String> libraryNames = Arrays.asList("onnxruntime","onnxruntime4j_jni");

    private OnnxRuntime() {}

    /**
     * Loads the native C library.
     * @throws IOException If it can't write to disk to copy out the library from the jar file.
     */
    static synchronized void init() throws IOException {
        if (!loaded) {
            // Check system properties for load time configuration.
            Properties props = System.getProperties();
            boolean debug = props.containsKey(LIBRARY_LOAD_LOGGING);
            boolean loadLibraryPath = props.containsKey(LOAD_LIBRARY_PATH);
            if (loadLibraryPath) {
                if (debug) {
                    logger.info("Loading from java.library.path");
                }
                try {
                    for (String libraryName : libraryNames) {
                        if (debug) {
                            logger.info("Loading " + libraryName + " from java.library.path");
                        }
                        System.loadLibrary(libraryName);
                    }
                } catch (UnsatisfiedLinkError e) {
                    logger.log(Level.SEVERE, "Failed to load onnx-runtime library from library path.");
                    throw e;
                }
            } else {
                if (debug) {
                    logger.info("Loading from classpath resource");
                }
                try {
                    for (String libraryName : libraryNames) {
                        try {
                            // This code path is used during testing.
                            String libraryFromJar = "/" + System.mapLibraryName(libraryName);
                            if (debug) {
                                logger.info("Attempting to load library from classpath using " + libraryFromJar);
                            }
                            String tempLibraryPath = createTempFileFromResource(libraryFromJar, debug);
                            if (debug) {
                                logger.info("Copied resource " + libraryFromJar + " to location " + tempLibraryPath);
                            }
                            System.load(tempLibraryPath);
                        } catch (Exception e) {
                            if (debug) {
                                logger.info("Failed to load from testing location, looking for /lib/<library-name>");
                            }
                            String libraryFromJar = "/lib/" + System.mapLibraryName(libraryName);
                            if (debug) {
                                logger.info("Attempting to load library from classpath using " + libraryFromJar);
                            }
                            String tempLibraryPath = createTempFileFromResource(libraryFromJar, debug);
                            if (debug) {
                                logger.info("Copied resource " + libraryFromJar + " to location " + tempLibraryPath);
                            }
                            System.load(tempLibraryPath);
                        }
                    }
                } catch (IOException e) {
                    logger.log(Level.SEVERE, "Failed to load onnx-runtime library from jar");
                    throw e;
                }
            }
            ortApiHandle = initialiseAPIBase(ORT_API_VERSION_1);
            loaded = true;
        }
    }

    /**
     * Copies out the named file from the class path into a temporary directory so it can be loaded
     * by {@link System#load}.
     * <p>
     * The file is marked delete on exit. Throws {@link IllegalArgumentException} if the
     * supplied path is not absolute.
     * @param path The path to the file in the classpath.
     * @param debugLogging If true turn on debug logging.
     * @return The path to the extracted file on disk.
     * @throws IOException If the file failed to read or write.
     */
    private static String createTempFileFromResource(String path, boolean debugLogging) throws IOException {
        if (!path.startsWith("/")) {
            throw new IllegalArgumentException("The path has to be absolute (start with '/').");
        } else {
            String[] parts = path.split("/");
            String filename = parts.length > 1 ? parts[parts.length - 1] : null;
            String prefix = "";
            String suffix = null;
            if (filename != null) {
                parts = filename.split("\\.", 2);
                prefix = parts[0];
                suffix = parts.length > 1 ? "." + parts[parts.length - 1] : null;
            }

            if (filename != null && prefix.length() >= 3) {
                File temp = File.createTempFile(prefix, suffix);
                if (debugLogging) {
                    logger.info("Writing " + path + " out to " + temp.getAbsolutePath());
                }
                temp.deleteOnExit();
                if (!temp.exists()) {
                    throw new FileNotFoundException("File " + temp.getAbsolutePath() + " does not exist.");
                } else {
                    byte[] buffer = new byte[1024];
                    try (InputStream is = OnnxRuntime.class.getResourceAsStream(path)) {
                        if (is == null) {
                            throw new FileNotFoundException("File " + path + " was not found inside JAR.");
                        } else {
                            int readBytes;
                            try (FileOutputStream os = new FileOutputStream(temp)) {
                                while ((readBytes = is.read(buffer)) != -1) {
                                    os.write(buffer, 0, readBytes);
                                }
                            }
                            return temp.getAbsolutePath();
                        }
                    }
                }
            } else {
                throw new IllegalArgumentException("The filename has to be at least 3 characters long.");
            }
        }
    }

    /**
     * Get a reference to the API struct.
     * @param apiVersionNumber The API version to use.
     * @return A pointer to the API struct.
     */
    private static native long initialiseAPIBase(int apiVersionNumber);
}
