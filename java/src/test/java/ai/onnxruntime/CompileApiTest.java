/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import ai.onnxruntime.OrtSession.SessionOptions;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

/** Test for the compilation API. */
public class CompileApiTest {
  private final OrtEnvironment env = OrtEnvironment.getEnvironment();

  @Test
  public void basicUsage() throws OrtException, IOException {
    SessionOptions so = new SessionOptions();
    try (OrtModelCompilationOptions compileOptions =
        OrtModelCompilationOptions.createFromSessionOptions(env, so)) {
      // mainly checking these don't throw which ensures all the plumbing for the binding works.
      compileOptions.setInputModelPath("model.onnx");
      compileOptions.setOutputModelPath("compiled_model.onnx");

      compileOptions.setOutputExternalInitializersPath("external_data.bin", 512);
      compileOptions.setEpContextEmbedMode(true);
    }

    try (OrtModelCompilationOptions compileOptions =
        OrtModelCompilationOptions.createFromSessionOptions(env, so)) {
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

  @Test
  public void epContextDataCallbackValidation() throws OrtException {
    try (SessionOptions sessionOptions = new SessionOptions()) {
      Assertions.assertThrows(
          NullPointerException.class,
          () -> sessionOptions.setEpContextDataReadCallback(null, 1024));
      Assertions.assertThrows(
          IllegalArgumentException.class,
          () -> sessionOptions.setEpContextDataReadCallback(name -> new byte[0], 0));
      Assertions.assertThrows(
          IllegalArgumentException.class,
          () -> sessionOptions.setEpContextDataReadCallback(name -> new byte[0], -1));

      sessionOptions.setEpContextDataReadCallback(name -> new byte[0], 1024);
      sessionOptions.setEpContextDataReadCallback(name -> new byte[] {1}, 1024);
      sessionOptions.clearEpContextDataReadCallback();

      try (OrtModelCompilationOptions compileOptions =
          OrtModelCompilationOptions.createFromSessionOptions(env, sessionOptions)) {
        Assertions.assertThrows(
            NullPointerException.class, () -> compileOptions.setEpContextDataWriteCallback(null));
        compileOptions.setEpContextDataWriteCallback((name, data) -> {});
        compileOptions.setEpContextDataWriteCallback((name, data) -> {});
        compileOptions.clearEpContextDataWriteCallback();
      }
    }
  }

  @Test
  @EnabledOnOs(OS.WINDOWS)
  public void externalEpContextDataUsesCallbacks() throws Exception {
    String libraryPath = TestHelpers.getResourcePath("/example_plugin_ep.dll").toString();
    Assertions.assertTrue(
        new File(libraryPath).exists(), "Expected library " + libraryPath + " does not exist.");

    String registrationName = "java_ep_context_callbacks";
    env.registerExecutionProviderLibrary(registrationName, libraryPath);

    Path compiledModelPath = Files.createTempFile("java_ep_context", ".onnx");
    Path failedModelPath = Files.createTempFile("java_ep_context_failed", ".onnx");
    Path clearedWriteModelPath = Files.createTempFile("java_ep_context_cleared", ".onnx");
    Files.deleteIfExists(compiledModelPath);
    Files.deleteIfExists(failedModelPath);
    Files.deleteIfExists(clearedWriteModelPath);

    AtomicReference<String> contextName = new AtomicReference<>();
    AtomicReference<byte[]> contextData = new AtomicReference<>();
    AtomicInteger replacedWriteCount = new AtomicInteger();
    AtomicInteger writeCount = new AtomicInteger();

    try {
      OrtEpDevice device =
          env.getEpDevices().stream()
              .filter(candidate -> candidate.getEpName().equals(registrationName))
              .findFirst()
              .orElseThrow(() -> new AssertionError("Registered EP device was not found"));
      byte[] inputModel = Files.readAllBytes(TestHelpers.getResourcePath("/mul_1.onnx"));

      try (SessionOptions sessionOptions = createPluginSessionOptions(device);
          OrtModelCompilationOptions compileOptions =
              OrtModelCompilationOptions.createFromSessionOptions(env, sessionOptions)) {
        configureCompilation(compileOptions, inputModel, failedModelPath);
        compileOptions.setEpContextDataWriteCallback(
            (name, data) -> {
              throw new IOException("synthetic Java EPContext write failure");
            });

        OrtException exception =
            Assertions.assertThrows(OrtException.class, compileOptions::compileModel);
        Assertions.assertTrue(
            exception.getMessage().contains("synthetic Java EPContext write failure"));
      }

      try (SessionOptions sessionOptions = createPluginSessionOptions(device);
          OrtModelCompilationOptions compileOptions =
              OrtModelCompilationOptions.createFromSessionOptions(env, sessionOptions)) {
        configureCompilation(compileOptions, inputModel, compiledModelPath);
        compileOptions.setEpContextDataWriteCallback(
            (name, data) -> replacedWriteCount.incrementAndGet());
        compileOptions.setEpContextDataWriteCallback(
            (name, data) -> {
              AtomicReference<Throwable> workerResult = new AtomicReference<>();
              Thread worker =
                  new Thread(
                      () -> {
                        try {
                          compileOptions.clearEpContextDataWriteCallback();
                        } catch (Throwable throwable) {
                          workerResult.set(throwable);
                        }
                      });
              worker.setDaemon(true);
              worker.start();
              worker.join(5000);
              Assertions.assertFalse(
                  worker.isAlive(), "Write callback reentry must not block on compilation");
              Assertions.assertInstanceOf(IllegalStateException.class, workerResult.get());
              Assertions.assertThrows(IllegalStateException.class, compileOptions::close);
              Assertions.assertThrows(
                  IllegalStateException.class,
                  () ->
                      compileOptions.setEpContextDataWriteCallback((unusedName, unusedData) -> {}));
              writeCount.incrementAndGet();
              contextName.set(name);
              contextData.set(data);
            });
        compileOptions.compileModel();
      }

      Assertions.assertEquals(0, replacedWriteCount.get());
      Assertions.assertEquals(1, writeCount.get());
      Assertions.assertNotNull(contextName.get());
      Assertions.assertFalse(contextName.get().isEmpty());
      Assertions.assertNotNull(contextData.get());
      Assertions.assertTrue(contextData.get().length > 0);
      Assertions.assertTrue(Files.exists(compiledModelPath));
      Assertions.assertFalse(Files.exists(compiledModelPath.resolveSibling(contextName.get())));

      AtomicInteger clearedWriteCount = new AtomicInteger();
      try (SessionOptions sessionOptions = createPluginSessionOptions(device);
          OrtModelCompilationOptions compileOptions =
              OrtModelCompilationOptions.createFromSessionOptions(env, sessionOptions)) {
        configureCompilation(compileOptions, inputModel, clearedWriteModelPath);
        compileOptions.setEpContextDataWriteCallback(
            (name, data) -> clearedWriteCount.incrementAndGet());
        compileOptions.clearEpContextDataWriteCallback();
        compileOptions.compileModel();
      }
      Assertions.assertEquals(0, clearedWriteCount.get());
      Assertions.assertTrue(Files.exists(clearedWriteModelPath.resolveSibling(contextName.get())));

      String supplementaryContextPrefix = "context-\uD83D\uDE80-";
      int namePaddingLength =
          contextName.get().getBytes(StandardCharsets.UTF_8).length
              - supplementaryContextPrefix.getBytes(StandardCharsets.UTF_8).length;
      Assertions.assertTrue(namePaddingLength >= 0);
      char[] namePadding = new char[namePaddingLength];
      Arrays.fill(namePadding, 'x');
      String supplementaryContextName = supplementaryContextPrefix + new String(namePadding);
      byte[] compiledModel =
          replaceUtf8(
              Files.readAllBytes(compiledModelPath), contextName.get(), supplementaryContextName);

      try (SessionOptions sessionOptions = createPluginSessionOptions(device)) {
        sessionOptions.setEpContextDataReadCallback(
            name -> {
              throw new IOException("synthetic Java EPContext read failure");
            },
            contextData.get().length);
        OrtException exception =
            Assertions.assertThrows(
                OrtException.class, () -> env.createSession(compiledModel, sessionOptions));
        Assertions.assertTrue(
            exception.getMessage().contains("synthetic Java EPContext read failure"),
            exception.getMessage());
      }

      try (SessionOptions sessionOptions = createPluginSessionOptions(device)) {
        sessionOptions.setEpContextDataReadCallback(name -> contextData.get(), 1);
        OrtException exception =
            Assertions.assertThrows(
                OrtException.class, () -> env.createSession(compiledModel, sessionOptions));
        Assertions.assertTrue(exception.getMessage().contains("configured maximum size"));
      }

      AtomicInteger emptyReadCount = new AtomicInteger();
      try (SessionOptions sessionOptions = createPluginSessionOptions(device);
          OrtSession session =
              createSessionWithReadCallback(
                  compiledModel,
                  sessionOptions,
                  name -> {
                    emptyReadCount.incrementAndGet();
                    return new byte[0];
                  },
                  contextData.get().length)) {
        Assertions.assertNotNull(session);
      }
      Assertions.assertEquals(1, emptyReadCount.get());

      AtomicInteger firstReadCount = new AtomicInteger();
      AtomicInteger secondReadCount = new AtomicInteger();
      AtomicInteger reentrantReplacementReadCount = new AtomicInteger();
      SessionOptions replacementOptions = createPluginSessionOptions(device);
      replacementOptions.setEpContextDataReadCallback(
          name -> {
            firstReadCount.incrementAndGet();
            return contextData.get();
          },
          contextData.get().length);
      replacementOptions.setEpContextDataReadCallback(
          name -> {
            secondReadCount.incrementAndGet();
            Assertions.assertEquals(supplementaryContextName, name);
            Assertions.assertThrows(IllegalStateException.class, replacementOptions::close);
            replacementOptions.setEpContextDataReadCallback(
                unusedName -> {
                  reentrantReplacementReadCount.incrementAndGet();
                  return contextData.get();
                },
                contextData.get().length);

            AtomicReference<Throwable> workerResult = new AtomicReference<>();
            Thread worker =
                new Thread(
                    () -> {
                      try {
                        replacementOptions.clearEpContextDataReadCallback();
                      } catch (Throwable throwable) {
                        workerResult.set(throwable);
                      }
                    });
            worker.setDaemon(true);
            worker.start();
            worker.join(5000);
            Assertions.assertFalse(
                worker.isAlive(), "Read callback reentry must not block on session creation");
            Assertions.assertNull(workerResult.get());
            return contextData.get();
          },
          contextData.get().length);
      try (OrtSession session = env.createSession(compiledModel, replacementOptions)) {
        replacementOptions.close();
        Assertions.assertNotNull(session);
      }
      Assertions.assertEquals(0, firstReadCount.get());
      Assertions.assertEquals(1, secondReadCount.get());
      Assertions.assertEquals(0, reentrantReplacementReadCount.get());

      AtomicInteger clearedReadCount = new AtomicInteger();
      try (SessionOptions sessionOptions = createPluginSessionOptions(device)) {
        sessionOptions.setEpContextDataReadCallback(
            name -> {
              clearedReadCount.incrementAndGet();
              return contextData.get();
            },
            contextData.get().length);
        sessionOptions.clearEpContextDataReadCallback();
        Assertions.assertThrows(
            OrtException.class, () -> env.createSession(compiledModel, sessionOptions));
      }
      Assertions.assertEquals(0, clearedReadCount.get());

      SessionOptions sourceOptions = createPluginSessionOptions(device);
      sourceOptions.setEpContextDataReadCallback(
          name -> contextData.get(), contextData.get().length);
      SessionOptions.EpContextDataReadCallbackRegistration registration =
          sourceOptions.getEpContextDataReadCallbackRegistration();
      Assertions.assertEquals(1, registration.getReferenceCount());
      OrtModelCompilationOptions retainedCompileOptions =
          OrtModelCompilationOptions.createFromSessionOptions(env, sourceOptions);
      try {
        Assertions.assertEquals(2, registration.getReferenceCount());
        sourceOptions.clearEpContextDataReadCallback();
        Assertions.assertEquals(1, registration.getReferenceCount());
        sourceOptions.close();
      } finally {
        retainedCompileOptions.close();
      }
      Assertions.assertEquals(0, registration.getReferenceCount());
    } finally {
      Files.deleteIfExists(compiledModelPath);
      Files.deleteIfExists(failedModelPath);
      Files.deleteIfExists(clearedWriteModelPath);
      if (contextName.get() != null) {
        Files.deleteIfExists(compiledModelPath.resolveSibling(contextName.get()));
        Files.deleteIfExists(failedModelPath.resolveSibling(contextName.get()));
        Files.deleteIfExists(clearedWriteModelPath.resolveSibling(contextName.get()));
      }
      env.unregisterExecutionProviderLibrary(registrationName);
    }
  }

  private SessionOptions createPluginSessionOptions(OrtEpDevice device) throws OrtException {
    SessionOptions sessionOptions = new SessionOptions();
    sessionOptions.addExecutionProvider(Collections.singletonList(device), Collections.emptyMap());
    return sessionOptions;
  }

  private static byte[] replaceUtf8(byte[] input, String oldValue, String newValue) {
    byte[] oldBytes = oldValue.getBytes(StandardCharsets.UTF_8);
    byte[] newBytes = newValue.getBytes(StandardCharsets.UTF_8);
    Assertions.assertEquals(oldBytes.length, newBytes.length);

    int match = -1;
    for (int i = 0; i <= input.length - oldBytes.length; i++) {
      boolean matches = true;
      for (int j = 0; j < oldBytes.length; j++) {
        if (input[i + j] != oldBytes[j]) {
          matches = false;
          break;
        }
      }
      if (matches) {
        Assertions.assertEquals(-1, match, "Expected the old value to occur exactly once");
        match = i;
      }
    }
    Assertions.assertNotEquals(-1, match, "Expected the old value in the compiled model");

    byte[] output = Arrays.copyOf(input, input.length);
    System.arraycopy(newBytes, 0, output, match, newBytes.length);
    return output;
  }

  private void configureCompilation(
      OrtModelCompilationOptions compileOptions, byte[] inputModel, Path outputModel)
      throws OrtException {
    compileOptions.setInputModelFromBuffer(ByteBuffer.wrap(inputModel));
    compileOptions.setOutputModelPath(outputModel.toString());
    compileOptions.setEpContextEmbedMode(false);
  }

  private OrtSession createSessionWithReadCallback(
      byte[] model,
      SessionOptions sessionOptions,
      SessionOptions.EpContextDataReadCallback callback,
      long maxDataSize)
      throws OrtException {
    sessionOptions.setEpContextDataReadCallback(callback, maxDataSize);
    return env.createSession(model, sessionOptions);
  }
}
