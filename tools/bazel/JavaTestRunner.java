// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.platform.console.ConsoleLauncher;

/** Runs existing JUnit tests with Bazel-built JNI libraries and declared model fixtures. */
public final class JavaTestRunner {
  public static void main(String[] args) {
    if (args.length < 4) {
      throw new IllegalArgumentException(
          "Expected runtime library, JNI library, Java fixture directory, and test class.");
    }
    if (!args[0].equals("-")) {
      for (int i = 0; i < 2; i++) {
        if (!Files.isRegularFile(Paths.get(args[i]))) {
          throw new IllegalArgumentException(
              "Missing native library " + args[i] + ". Build the java_tensor_test target first.");
        }
      }
      System.setProperty(
          "onnxruntime.native.onnxruntime.path", Paths.get(args[0]).toAbsolutePath().toString());
      System.setProperty(
          "onnxruntime.native.onnxruntime4j_jni.path", Paths.get(args[1]).toAbsolutePath().toString());
    }
    if (!Files.isDirectory(Paths.get(args[2]))) {
      throw new IllegalArgumentException(
          "Missing Java fixture directory " + args[2] + ". Include Java test resources in runfiles.");
    }
    System.setProperty("onnxruntime.test.resources", Paths.get(args[2]).toAbsolutePath().toString());
    List<String> options = new ArrayList<>();
    options.add("execute");
    options.add("--fail-if-no-tests");
    String filter = System.getenv("TESTBRIDGE_TEST_ONLY");
    if (filter != null && !filter.isEmpty()) {
      if (!filter.equals(args[3]) && !filter.startsWith(args[3] + "#")) {
        throw new IllegalArgumentException(
            "Filter must select " + args[3] + " or " + args[3] + "#method; got: " + filter);
      }
      options.add(filter.contains("#") ? "--select-method=" + filter : "--select-class=" + filter);
    } else {
      options.add("--select-class=" + args[3]);
    }
    String reports = System.getenv("XML_OUTPUT_FILE");
    if (reports != null) {
      options.add("--reports-dir=" + Paths.get(reports).getParent());
    }
    options.addAll(Arrays.asList(args).subList(4, args.length));
    System.exit(
        ConsoleLauncher.run(
                new PrintWriter(System.out, true),
                new PrintWriter(System.err, true),
                options.toArray(new String[0]))
            .getExitCode());
  }
}
