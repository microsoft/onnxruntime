/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import ai.onnxruntime.OrtSession.SessionOptions;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/** Test for the compilation API. */
public class CompileApiTest {
  private final OrtEnvironment env = OrtEnvironment.getEnvironment();

  @Test
  public void basicUsage() throws OrtException, IOException {
    SessionOptions so = new SessionOptions();
    try (OrtModelCompilationOptions compileOptions =
        OrtModelCompilationOptions.createFromSessionOptions(so)) {
      // mainly checking these don't throw which ensures all the plumbing for the binding works.
      compileOptions.setInputModelPath("model.onnx");
      compileOptions.setOutputModelPath("compiled_model.onnx");

      compileOptions.setOutputExternalInitializersPath("external_data.bin", 512);
      compileOptions.setEpContextEmbedMode(true);
    }

    try (OrtModelCompilationOptions compileOptions =
        OrtModelCompilationOptions.createFromSessionOptions(so)) {
      Path modelPath = TestHelpers.getResourcePath("/squeezenet.onnx");
      byte[] modelBytes = Files.readAllBytes(modelPath);
      ByteBuffer modelBuffer = ByteBuffer.wrap(modelBytes);
      compileOptions.setInputModelFromBuffer(modelBuffer);
      compileOptions.setOutputModelPath("compiled_model.onnx");

      File f = new File("compiled_model.onnx");

      compileOptions.compileModel();

      // Check the compiled model is valid
      try (OrtSession session = env.createSession(f.toString(), so)) {
        Assertions.assertNotNull(session);
      }

      f.delete();
    }
  }
}
