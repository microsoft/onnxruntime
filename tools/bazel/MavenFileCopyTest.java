// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.DosFileAttributeView;
import java.util.Arrays;

/** Regresses copying readonly Bazel artifacts into a deletable Maven repository. */
public final class MavenFileCopyTest {
  public static void main(String[] args) throws Exception {
    Path root = Files.createTempDirectory("ort-maven-copy-");
    Path source = root.resolve("source.jar");
    Path direct = root.resolve("plugin.jar");
    Path tree = root.resolve("repository");
    byte[] content = new byte[] {1, 2, 3};
    Files.write(source, content);
    boolean windows = System.getProperty("os.name").startsWith("Windows");
    try {
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
      Files.delete(source);
      Files.delete(root);
    }
  }
}
