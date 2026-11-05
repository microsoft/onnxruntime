// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import java.nio.file.Files;
import java.io.File;
import java.io.InputStream;
import java.nio.file.Path;
import java.nio.file.attribute.DosFileAttributeView;
import java.util.Arrays;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;

/** Regresses copying readonly Bazel artifacts into a deletable Maven repository. */
public final class MavenFileCopyTest {
  public static void main(String[] args) throws Exception {
    Path root = Files.createTempDirectory("ort-maven-copy-");
    String classpath = MavenPackageSmokeTest.mavenClasspath(root);
    String expected =
        root.resolve("lib") + File.separator + "*"
            + File.pathSeparator + root.resolve("boot") + File.separator + "*";
    if (!expected.equals(classpath)) {
      throw new AssertionError("Maven launcher wildcard classpath changed: " + classpath);
    }
    Path source = root.resolve("source.jar");
    Path direct = root.resolve("plugin.jar");
    Path tree = root.resolve("repository");
    Path lib = root.resolve("lib");
    Path probeJar = lib.resolve("probe.jar");
    byte[] content = new byte[] {1, 2, 3};
    Files.write(source, content);
    boolean windows = System.getProperty("os.name").startsWith("Windows");
    try {
      Files.createDirectory(lib);
      String probeClass = WildcardProbe.class.getName();
      String resource = probeClass.replace('.', '/') + ".class";
      try (JarOutputStream jar = new JarOutputStream(Files.newOutputStream(probeJar));
          InputStream input = MavenFileCopyTest.class.getResourceAsStream("/" + resource)) {
        if (input == null) {
          throw new AssertionError("Missing compiled wildcard probe: " + resource);
        }
        jar.putNextEntry(new JarEntry(resource));
        input.transferTo(jar);
        jar.closeEntry();
      }
      Path java =
          Path.of(System.getProperty("java.home"), "bin", windows ? "java.exe" : "java");
      Process probe =
          new ProcessBuilder(java.toString(), "-cp", classpath, probeClass).inheritIO().start();
      if (probe.waitFor() != 0) {
        throw new AssertionError("Java failed to resolve the Maven wildcard classpath.");
      }
      Files.delete(probeJar);
      Files.delete(lib);
      if (windows) {
        Files.getFileAttributeView(source, DosFileAttributeView.class).setReadOnly(true);
      }
      MavenPackageSmokeTest.copyFile(source, direct);
      MavenPackageSmokeTest.copyTree(source, tree);
      for (Path destination : new Path[] {direct, tree}) {
        if (!Arrays.equals(content, Files.readAllBytes(destination))) {
          throw new AssertionError("Maven fixture copy changed artifact bytes: " + destination);
        }
        Files.delete(destination);
      }
      System.out.println("Readonly Maven fixture copies are byte-identical and immediately deletable.");
    } finally {
      if (windows) {
        Files.getFileAttributeView(source, DosFileAttributeView.class).setReadOnly(false);
      }
      Files.deleteIfExists(direct);
      Files.deleteIfExists(tree);
      Files.deleteIfExists(probeJar);
      Files.deleteIfExists(lib);
      Files.delete(source);
      Files.delete(root);
    }
  }

  public static final class WildcardProbe {
    public static void main(String[] args) {
      System.out.println("Maven wildcard classpath launched the isolated probe JAR.");
    }
  }
}
