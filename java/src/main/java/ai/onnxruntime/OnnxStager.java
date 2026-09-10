package ai.onnxruntime;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Stages native libraries required by the OnnxRuntime Java API into a temporary directory. Supports
 * several resolution strategies, see {@link ai.onnxruntime}.
 *
 * <p>Staged libraries are cached so that each logical library is only resolved once per {@code
 * OnnxStager} instance.
 */
final class OnnxStager {
  private static final Logger logger = Logger.getLogger(OnnxStager.class.getName());

  /**
   * Detects the current operating system and CPU architecture and returns a canonical {@code
   * "<os>-<arch>"} string (e.g. {@code "linux-x64"}, {@code "win-x64"}).
   *
   * @return the OS/architecture identifier string
   * @throws IllegalStateException if the OS or CPU architecture is not supported
   */
  private static String initOsArch() {
    String detectedOS = null;
    String os = System.getProperty("os.name", "generic").toLowerCase(Locale.ENGLISH);
    if (os.contains("mac") || os.contains("darwin")) {
      detectedOS = "osx";
    } else if (os.contains("win")) {
      detectedOS = "win";
    } else if (os.contains("nux")) {
      detectedOS = "linux";
    } else {
      throw new IllegalStateException("Unsupported os:" + os);
    }
    String detectedArch = null;
    String arch = System.getProperty("os.arch", "generic").toLowerCase(Locale.ENGLISH);
    if (arch.startsWith("amd64") || arch.startsWith("x86_64")) {
      detectedArch = "x64";
    } else if (arch.startsWith("x86")) {
      // 32-bit x86 is not supported by the Java API
      detectedArch = "x86";
    } else if (arch.startsWith("aarch64")) {
      detectedArch = "aarch64";
    } else if (arch.startsWith("ppc64")) {
      detectedArch = "ppc64";
    } else if (arch.startsWith("loongarch64")) {
      detectedArch = "loongarch64";
    } else {
      throw new IllegalStateException("Unsupported arch:" + arch);
    }
    return detectedOS + '-' + detectedArch;
  }

  /** The OS & CPU architecture string */
  private final String osArchStr;

  /**
   * Map from prefixed library name (e.g. {@code "onnxruntime.native.onnxruntime4j_jni"}) to its
   * staged path.
   */
  private final Map<String, Path> staged;

  /** Temporary directory into which native libraries are staged before loading. */
  private final Path stagingDirectory;

  /**
   * Creates a new {@code OnnxStager} and allocates a temporary staging directory. The directory is
   * registered for deletion on JVM exit.
   *
   * @throws IOException if the temporary directory cannot be created
   */
  OnnxStager() throws IOException {
    osArchStr = initOsArch();
    staged = new HashMap<>();
    stagingDirectory = Files.createTempDirectory("onnxruntime-java");
    stagingDirectory.toFile().deleteOnExit();
  }

  /**
   * Resolves and stages a single native library, returning its path in the staging directory. For
   * resolution order see {@link ai.onnxruntime}.
   *
   * <p>Once a library is staged its path is cached; subsequent calls with the same {@code prefix}
   * and {@code library} return the cached path immediately.
   *
   * @param prefix the property-key prefix (e.g. {@code "onnxruntime.native"})
   * @param library the platform-independent library name (e.g. {@code "onnxruntime4j_jni"})
   * @param classpathDirectory the classpath resource directory containing the bundled native
   *     libraries
   * @return the {@link Path} to the staged library file, or {@code null} if staging was skipped
   * @throws IOException if the library cannot be found or copied to the staging directory
   */
  public synchronized Path stage(String prefix, String library, String classpathDirectory)
      throws IOException {
    // 1) The user may skip staging of all libraries with this prefix:
    String skipAll = System.getProperty(prefix + ".skip");
    if (Boolean.TRUE.toString().equalsIgnoreCase(skipAll)) {
      logger.log(Level.FINE, "Skipping staging of native library '" + library + "'");
      return null;
    }

    String prefixedName = prefix + "." + library;
    if (staged.containsKey(prefixedName)) {
      return staged.get(prefixedName);
    }

    // 2) The user may skip staging of this library:
    String skip = System.getProperty(prefixedName + ".skip");
    if (Boolean.TRUE.toString().equalsIgnoreCase(skip)) {
      logger.log(Level.FINE, "Skipping staging of native library '" + library + "'");
      return null;
    }

    // Resolve the platform dependent library name.
    String libraryFileName = mapLibraryName(library);

    // 3) The user may explicitly specify the path to their shared library:
    String libraryPathProperty = System.getProperty(prefixedName + ".path");
    if (libraryPathProperty != null) {
      logger.log(
          Level.FINE,
          "Attempting to use native library '"
              + library
              + "' from specified path: "
              + libraryPathProperty);
      // TODO: Switch this to Path.of when the minimum Java version is 11.
      Path libraryFilePath = Paths.get(libraryPathProperty).toAbsolutePath();
      if (!libraryFilePath.toFile().exists()) {
        throw new IOException("Native library '" + library + "' not found at " + libraryFilePath);
      } else {
        // Keep the name the user provided
        Path target = stagingDirectory.resolve(libraryFilePath.getFileName());
        pull(libraryFilePath, target);
        staged.put(prefixedName, target);
        return target;
      }
    }

    Path target = stagingDirectory.resolve(libraryFileName);

    // 4) The user may explicitly specify the path to a directory containing all
    // shared libraries:
    String libraryDirPathProperty = System.getProperty(prefix + ".path");
    if (libraryDirPathProperty != null) {
      logger.log(
          Level.FINE,
          "Attempting to use native library '"
              + library
              + "' from specified path: "
              + libraryDirPathProperty);
      // TODO: Switch this to Path.of when the minimum Java version is 11.
      Path libraryFilePath = Paths.get(libraryDirPathProperty, libraryFileName).toAbsolutePath();
      if (libraryFilePath.toFile().exists()) {
        pull(libraryFilePath, target);
        staged.put(prefixedName, target);
        return target;
      } else {
        logger.log(
            Level.WARNING,
            "Library '" + libraryFilePath + "'' not found, falling back to bundeled instance.");
      }
    }

    // 5) try loading from resources or library path:
    extractFromResources(classpathDirectory, target);
    logger.log(Level.FINE, "Using native library '" + library + "' from resource path");
    staged.put(prefixedName, target);
    return target;
  }

  /**
   * Extracts a bundled native library from classpath resources to the given target path.
   *
   * <p>The resource is located at {@code <resourceDir>/<osArchStr>/<filename>} and written to
   * {@code target} using a 4 KiB copy buffer.
   *
   * @param resourceDir the classpath resource directory containing the platform-specific
   *     subdirectories
   * @param target the destination path to write the extracted library to
   * @throws FileNotFoundException if the resource cannot be found on the classpath
   * @throws IOException if reading the resource or writing the target file fails
   */
  private void extractFromResources(String resourceDir, Path target) throws IOException {
    String name = target.getFileName().toString();
    String resourcePath = resourceDir + "/" + osArchStr + "/" + name;
    try (InputStream is = OnnxRuntime.class.getResourceAsStream(resourcePath)) {
      if (is == null) {
        throw new FileNotFoundException("Library " + resourcePath + " not found on classpath!");
      } else {
        // Found in classpath resources, load via temporary file
        logger.log(
            Level.FINE,
            "Attempting to extracting native library '"
                + name
                + "' from resource path "
                + resourcePath);
        byte[] buffer = new byte[4096];
        int readBytes;
        try (FileOutputStream os = new FileOutputStream(target.toFile())) {
          while ((readBytes = is.read(buffer)) != -1) {
            os.write(buffer, 0, readBytes);
          }
        }
        logger.log(Level.FINE, "Extracted native library '" + name + "' from resource path");
      }
    }
  }

  /**
   * Makes the file at {@code from} accessible at {@code to} by creating a hard link (Windows) or a
   * symbolic link (other platforms). Falls back to a full file copy if link creation fails.
   *
   * @param from the source file path
   * @param to the destination path in the staging directory
   * @throws IOException if both link creation and the copy fallback fail
   */
  private void pull(Path from, Path to) throws IOException {
    try {
      if (isWindows()) {
        Files.createLink(to, from);
      } else {
        Files.createSymbolicLink(from, to);
      }
    } catch (Exception e) {
      logger.log(
          Level.FINE,
          "Could not create "
              + (isWindows() ? "hard link" : "soft link")
              + "due to "
              + e.getMessage()
              + ", falling back to copying...");
      Files.copy(from, to);
    }
  }

  /**
   * Maps the library name into a platform dependent library filename. Converts macOS's "jnilib" to
   * "dylib" but otherwise is the same as {@link System#mapLibraryName(String)}.
   *
   * @param library The library name
   * @return The library filename.
   */
  private String mapLibraryName(String library) {
    return System.mapLibraryName(library).replace("jnilib", "dylib");
  }

  /**
   * Returns {@code true} if the current platform is Windows.
   *
   * @return {@code true} on Windows, {@code false} otherwise
   */
  private boolean isWindows() {
    return osArchStr.startsWith("win");
  }
}
