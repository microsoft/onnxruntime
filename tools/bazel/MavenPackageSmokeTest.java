// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import java.io.InputStream;
import java.io.File;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/** Exercises the installed platform GAV without the source-tree API on its class path. */
public final class MavenPackageSmokeTest {
  public static void main(String[] args) throws Exception {
    if (args.length != 9) {
      throw new IllegalArgumentException(
          "Pass the Maven ZIP, version file, Maven marker, resolver plugin JAR/POM, "
              + "platform OS/architecture, CPU/GPU variant, and Add model.");
    }
    Path archive = Paths.get(args[0]).toAbsolutePath();
    String version =
        new String(Files.readAllBytes(Paths.get(args[1])), StandardCharsets.UTF_8).trim();
    Path mavenHome = Paths.get(args[2]).toAbsolutePath().getParent().getParent();
    Path pluginJar = Paths.get(args[3]).toAbsolutePath();
    Path pluginPom = Paths.get(args[4]).toAbsolutePath();
    String platform = args[5] + args[6];
    String variant = args[7];
    String artifact =
        "onnxruntime-bazel" + ("cuda".equals(variant) ? "-gpu" : "") + "-" + platform;
    Path directory = Files.createTempDirectory("ort-maven-consumer-");
    try {
      try (InputStream input = Files.newInputStream(archive);
          ZipInputStream zip = new ZipInputStream(input)) {
        ZipEntry entry;
        while ((entry = zip.getNextEntry()) != null) {
          Path destination = directory.resolve(entry.getName()).normalize();
          if (!destination.startsWith(directory)) {
            throw new IllegalArgumentException("Unsafe Maven archive member: " + entry.getName());
          }
          if (!entry.isDirectory()) {
            Files.createDirectories(destination.getParent());
            Files.copy(zip, destination);
          }
        }
      }
      Path repository =
          directory.resolve("com/microsoft/onnxruntime").resolve(artifact).resolve(version);
      String stem = artifact + "-" + version;
      for (String suffix : new String[] {".pom", ".jar", "-sources.jar", "-javadoc.jar"}) {
        if (!Files.isRegularFile(repository.resolve(stem + suffix))) {
          throw new AssertionError("Maven release omitted " + stem + suffix);
        }
      }
      Path nativeJar = repository.resolve(stem + ".jar");
      Path localRepository = directory.resolve("repository");
      copyTree(directory.resolve("com"), localRepository.resolve("com"));
      Path pluginDirectory =
          localRepository.resolve(
              "com/microsoft/onnxruntime/fixture/maven-resolution-probe/1.0.0");
      Files.createDirectories(pluginDirectory);
      copyFile(
          pluginJar,
          pluginDirectory.resolve("maven-resolution-probe-1.0.0.jar"));
      copyFile(
          pluginPom,
          pluginDirectory.resolve("maven-resolution-probe-1.0.0.pom"));
      Path consumerPom = directory.resolve("consumer-pom.xml");
      Files.write(
          consumerPom,
          ("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                  + "<project xmlns=\"http://maven.apache.org/POM/4.0.0\">\n"
                  + "  <modelVersion>4.0.0</modelVersion>\n"
                  + "  <groupId>fixture</groupId><artifactId>consumer</artifactId>"
                  + "<version>1.0.0</version>\n"
                  + "  <dependencies><dependency><groupId>com.microsoft.onnxruntime</groupId>"
                  + "<artifactId>"
                  + artifact
                  + "</artifactId><version>"
                  + version
                  + "</version></dependency></dependencies>\n"
                  + "</project>\n")
              .getBytes(StandardCharsets.UTF_8));
      Path java =
          Paths.get(
              System.getProperty("java.home"),
              "bin",
              System.getProperty("os.name").startsWith("Windows") ? "java.exe" : "java");
      String classpath = mavenClasspath(mavenHome);
      List<String> command = new ArrayList<>();
      command.add(java.toString());
      command.add("-Dmaven.home=" + mavenHome);
      command.add("-Dmaven.multiModuleProjectDirectory=" + directory);
      command.add("-cp");
      command.add(classpath);
      command.add("org.apache.maven.cli.MavenCli");
      command.add("-o");
      command.add("-q");
      command.add("-Dmaven.repo.local=" + localRepository);
      command.add("-Dort.model=" + Paths.get(args[8]).toAbsolutePath());
      command.add("-Dort.variant=" + variant);
      command.add("-f");
      command.add(consumerPom.toString());
      command.add("com.microsoft.onnxruntime.fixture:maven-resolution-probe:1.0.0:run");
      Process maven = new ProcessBuilder(command).inheritIO().start();
      if (maven.waitFor() != 0) {
        throw new AssertionError("Offline Maven dependency resolution or inference failed.");
      }
      URL testJar = MavenPackageSmokeTest.class.getProtectionDomain().getCodeSource().getLocation();
      try (URLClassLoader consumer =
          new URLClassLoader(
              new URL[] {nativeJar.toUri().toURL(), testJar},
              ClassLoader.getSystemClassLoader().getParent())) {
        Class<?> environment = consumer.loadClass("ai.onnxruntime.OrtEnvironment");
        URL origin = environment.getProtectionDomain().getCodeSource().getLocation();
        if (!origin.equals(nativeJar.toUri().toURL())) {
          throw new AssertionError("Consumer loaded the API outside the installed classifier: " + origin);
        }
        Class<?> smoke = consumer.loadClass("RuntimeSmokeTest");
        String model = Paths.get(args[8]).toAbsolutePath().toString();
        String[] inferenceArgs =
            "cuda".equals(variant) ? new String[] {model, "--cuda"} : new String[] {model};
        smoke.getMethod("main", String[].class).invoke(null, (Object) inferenceArgs);
      }
      System.out.println("Maven repository installation and isolated Java/JNI Add inference passed.");
    } finally {
      List<Path> paths;
      try (Stream<Path> files = Files.walk(directory)) {
        paths = files.sorted(Comparator.reverseOrder()).collect(Collectors.toList());
      }
      for (Path path : paths) {
        Files.delete(path);
      }
    }
  }

  static String mavenClasspath(Path mavenHome) {
    return mavenHome.resolve("lib") + File.separator + "*"
        + File.pathSeparator + mavenHome.resolve("boot") + File.separator + "*";
  }

  static void copyFile(Path source, Path destination) throws Exception {
    // Files.copy(Path, Path) preserves Bazel's DOS readonly bit on Windows.
    try (InputStream input = Files.newInputStream(source)) {
      Files.copy(input, destination, StandardCopyOption.REPLACE_EXISTING);
    }
  }

  static void copyTree(Path source, Path destination) throws Exception {
    try (Stream<Path> files = Files.walk(source)) {
      for (Path path : files.collect(Collectors.toList())) {
        Path target = destination.resolve(source.relativize(path));
        if (Files.isDirectory(path)) {
          Files.createDirectories(target);
        } else {
          Files.createDirectories(target.getParent());
          copyFile(path, target);
        }
      }
    }
  }
}
