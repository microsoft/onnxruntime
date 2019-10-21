/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 */
package com.microsoft.onnxruntime;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Static loader for the JNI binding.
 */
final class ONNX {
    private static final Logger logger = Logger.getLogger(ONNX.class.getName());

    // The initial release of the ORT API.
    private static final int ORT_API_VERSION_1 = 1;

    private static boolean loaded = false;

    // The API handle.
    static long ortApiHandle;

    private static final List<String> libraryNames = Arrays.asList("onnxruntime","ONNX4j");

    private ONNX() {}

    /**
     * Loads the native C library.
     * @throws IOException If it can't write to disk to copy out the library from the jar file.
     */
    static synchronized void init() throws IOException {
        if (!loaded) {
            try {
                for (String libraryName : libraryNames) {
                    try {
                        // This code path is used during testing.
                        String libraryFromJar = "/" + System.mapLibraryName(libraryName);
                        String tempLibraryPath = createTempFileFromResource(libraryFromJar);
                        System.load(tempLibraryPath);
                    } catch (Exception e) {
                        String libraryFromJar = "/lib/" + System.mapLibraryName(libraryName);
                        String tempLibraryPath = createTempFileFromResource(libraryFromJar);
                        System.load(tempLibraryPath);
                    }
                }
                ortApiHandle = initialiseAPIBase(ORT_API_VERSION_1);
                loaded = true;
            } catch (IOException e) {
                logger.log(Level.SEVERE,"Failed to load ONNX library from jar");
                throw e;
            }
        }
    }

    private static String createTempFileFromResource(String path) throws IOException, IllegalArgumentException {
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
                temp.deleteOnExit();
                if (!temp.exists()) {
                    throw new FileNotFoundException("File " + temp.getAbsolutePath() + " does not exist.");
                } else {
                    byte[] buffer = new byte[1024];
                    try (InputStream is = ONNX.class.getResourceAsStream(path)) {
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
