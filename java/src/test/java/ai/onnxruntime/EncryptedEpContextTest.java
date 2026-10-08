/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import ai.onnxruntime.OrtSession.SessionOptions;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.SecureRandom;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import javax.crypto.AEADBadTagException;
import javax.crypto.Cipher;
import javax.crypto.spec.GCMParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

public class EncryptedEpContextTest {
  @Test
  @EnabledOnOs(OS.WINDOWS)
  public void encryptedCompiledModelAndContextRunInference() throws Exception {
    OrtEnvironment env = OrtEnvironment.getEnvironment();
    String epName = "java_encrypted_ep_context";
    String library = TestHelpers.getResourcePath("/example_plugin_ep.dll").toString();
    Path directory = Files.createTempDirectory("encrypted_ep_context");
    Path plaintextModel = directory.resolve("compiled.onnx");
    Path modelFile = directory.resolve("model.aesgcm");
    Path contextFile = directory.resolve("context.aesgcm");
    byte[] key = new byte[32];
    new SecureRandom().nextBytes(key);
    AtomicReference<String> contextName = new AtomicReference<>();
    AtomicInteger writes = new AtomicInteger();
    AtomicInteger reads = new AtomicInteger();
    env.registerExecutionProviderLibrary(epName, library);
    try {
      OrtEpDevice device =
          env.getEpDevices().stream()
              .filter(candidate -> candidate.getEpName().equals(epName))
              .findFirst()
              .orElseThrow(() -> new AssertionError("Registered EP device was not found"));
      OnnxMl.ModelProto.Builder original =
          OnnxMl.ModelProto.parseFrom(
              Files.readAllBytes(TestHelpers.getResourcePath("/mul_1.onnx")))
              .toBuilder();
      OnnxMl.GraphProto.Builder graph = original.getGraphBuilder();
      Assertions.assertEquals(1, graph.getNodeCount());
      OnnxMl.NodeProto mul = graph.getNode(0);
      Assertions.assertEquals("Mul", mul.getOpType());
      OnnxMl.TypeProto inputType = graph.getInput(0).getType();
      long[] shape =
          inputType.getTensorType().getShape().getDimList().stream()
              .mapToLong(OnnxMl.TensorShapeProto.Dimension::getDimValue)
              .toArray();
      int count = 1;
      for (long dimension : shape) {
        Assertions.assertTrue(dimension > 0);
        count = Math.multiplyExact(count, Math.toIntExact(dimension));
      }
      graph.clearInitializer().clearInput();
      for (String name : mul.getInputList()) {
        graph.addInput(OnnxMl.ValueInfoProto.newBuilder().setName(name).setType(inputType));
      }
      byte[] inputModel = original.build().toByteArray();
      try (SessionOptions options = options(device);
          OrtModelCompilationOptions compile =
              OrtModelCompilationOptions.createFromSessionOptions(env, options)) {
        ByteBuffer input = ByteBuffer.allocateDirect(inputModel.length);
        input.put(inputModel).flip();
        compile.setInputModelFromBuffer(input);
        compile.setEpContextEmbedMode(false);
        // Java currently exposes only file output for the compiled ONNX model.
        compile.setOutputModelPath(plaintextModel.toString());
        compile.setEpContextDataWriteCallback(
            (name, data) -> {
              Assertions.assertEquals(0, writes.getAndIncrement());
              contextName.set(name);
              Files.write(contextFile, encrypt(data, key, name));
            });
        compile.compileModel();
      }
      Arrays.fill(inputModel, (byte) 0);
      byte[] compiled = Files.readAllBytes(plaintextModel);
      Assertions.assertEquals(
          "EPContext", OnnxMl.ModelProto.parseFrom(compiled).getGraph().getNode(0).getOpType());
      Files.write(modelFile, encrypt(compiled, key, "compiled.onnx"));
      Arrays.fill(compiled, (byte) 0);
      Files.delete(plaintextModel);
      Assertions.assertEquals(1, writes.get());
      Assertions.assertNotNull(contextName.get());
      Assertions.assertFalse(Files.exists(directory.resolve(contextName.get())));

      byte[] encryptedModel = Files.readAllBytes(modelFile);
      byte[] encryptedContext = Files.readAllBytes(contextFile);
      byte[] restored = decrypt(encryptedModel, key, "compiled.onnx");
      try {
        try (SessionOptions options = options(device)) {
          options.setEpContextDataReadCallback(
              name -> {
                Assertions.assertEquals(contextName.get(), name);
                reads.incrementAndGet();
                return decrypt(Files.readAllBytes(contextFile), key, name);
              },
              encryptedContext.length - 28);
          try (OrtSession session = env.createSession(restored, options)) {
            for (int iteration = 0; iteration < 2; ++iteration) {
              float[] left = new float[count];
              float[] right = new float[count];
              float[] expected = new float[count];
              for (int i = 0; i < count; ++i) {
                left[i] = i - iteration;
                right[i] = 2 * i + iteration;
                expected[i] = left[i] * right[i];
              }
              try (OnnxTensor x =
                      OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(left), shape);
                  OnnxTensor y =
                      OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(right), shape)) {
                Map<String, OnnxTensor> inputs = new HashMap<>();
                inputs.put(mul.getInput(0), x);
                inputs.put(mul.getInput(1), y);
                try (OrtSession.Result result = session.run(inputs)) {
                  float[] actual = new float[count];
                  ((OnnxTensor) result.get(0)).getFloatBuffer().get(actual);
                  Assertions.assertArrayEquals(expected, actual);
                }
              }
            }
          }
        }
        Assertions.assertEquals(1, reads.get());
        byte[] wrongKey = key.clone();
        wrongKey[0] ^= 1;
        Assertions.assertThrows(
            AEADBadTagException.class, () -> decrypt(encryptedModel, wrongKey, "compiled.onnx"));
        assertContextRejected(env, device, restored, encryptedContext, wrongKey);
        Arrays.fill(wrongKey, (byte) 0);
        encryptedModel[28] ^= 1;
        Assertions.assertThrows(
            AEADBadTagException.class, () -> decrypt(encryptedModel, key, "compiled.onnx"));
        encryptedContext[28] ^= 1;
        assertContextRejected(env, device, restored, encryptedContext, key);
        Assertions.assertThrows(
            AEADBadTagException.class,
            () -> decrypt(Files.readAllBytes(contextFile), key, "different-context-name"));
        try (SessionOptions options = options(device)) {
          Assertions.assertThrows(OrtException.class, () -> env.createSession(restored, options));
        }
        Assertions.assertFalse(Files.exists(plaintextModel));
        Assertions.assertFalse(Files.exists(directory.resolve(contextName.get())));
        try (java.util.stream.Stream<Path> files = Files.list(directory)) {
          Assertions.assertEquals(2, files.count());
        }
      } finally {
        Arrays.fill(restored, (byte) 0);
      }
    } finally {
      env.unregisterExecutionProviderLibrary(epName);
      Arrays.fill(key, (byte) 0);
      Files.deleteIfExists(plaintextModel);
      Files.deleteIfExists(modelFile);
      Files.deleteIfExists(contextFile);
      Files.delete(directory);
    }
  }

  private static SessionOptions options(OrtEpDevice device) throws OrtException {
    SessionOptions options = new SessionOptions();
    options.addConfigEntry("ep.example.test_execute_ep_context", "1");
    options.addExecutionProvider(Arrays.asList(device), new HashMap<>());
    return options;
  }

  private static void assertContextRejected(
      OrtEnvironment env, OrtEpDevice device, byte[] model, byte[] context, byte[] key)
      throws OrtException {
    try (SessionOptions options = options(device)) {
      options.setEpContextDataReadCallback(
          name -> {
            try {
              decrypt(context, key, name);
            } catch (AEADBadTagException failure) {
              throw new java.io.IOException("EPContext authentication failed", failure);
            }
            throw new AssertionError("Corrupt ciphertext was accepted.");
          },
          context.length - 28);
      OrtException failure =
          Assertions.assertThrows(OrtException.class, () -> env.createSession(model, options));
      Assertions.assertTrue(failure.getMessage().contains("EPContext authentication failed"));
    }
  }

  // Persist nonce || tag || ciphertext, with the logical asset name as authenticated data.
  private static byte[] encrypt(byte[] plaintext, byte[] key, String name) throws Exception {
    byte[] nonce = new byte[12];
    new SecureRandom().nextBytes(nonce);
    Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
    cipher.init(
        Cipher.ENCRYPT_MODE, new SecretKeySpec(key, "AES"), new GCMParameterSpec(128, nonce));
    cipher.updateAAD(name.getBytes(StandardCharsets.UTF_8));
    byte[] ciphertextAndTag = cipher.doFinal(plaintext);
    byte[] record = new byte[28 + plaintext.length];
    System.arraycopy(nonce, 0, record, 0, 12);
    System.arraycopy(ciphertextAndTag, plaintext.length, record, 12, 16);
    System.arraycopy(ciphertextAndTag, 0, record, 28, plaintext.length);
    return record;
  }

  private static byte[] decrypt(byte[] record, byte[] key, String name) throws Exception {
    Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
    cipher.init(
        Cipher.DECRYPT_MODE,
        new SecretKeySpec(key, "AES"),
        new GCMParameterSpec(128, record, 0, 12));
    cipher.updateAAD(name.getBytes(StandardCharsets.UTF_8));
    byte[] ciphertextAndTag = new byte[record.length - 12];
    System.arraycopy(record, 28, ciphertextAndTag, 0, record.length - 28);
    System.arraycopy(record, 12, ciphertextAndTag, record.length - 28, 16);
    return cipher.doFinal(ciphertextAndTag);
  }
}
