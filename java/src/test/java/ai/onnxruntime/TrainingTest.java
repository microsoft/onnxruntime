/*
 * Copyright (c) 2022, 2023, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import ai.onnxruntime.OrtTrainingSession.OrtCheckpointState;
import ai.onnxruntime.TensorInfo.OnnxTensorType;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/** Tests for the ORT training apis. */
@EnabledIfSystemProperty(named = "ENABLE_TRAINING_APIS", matches = "1")
public class TrainingTest {

  private static final OrtEnvironment env = OrtEnvironment.getEnvironment();

  @Test
  public void testLoadCheckpoint() throws OrtException {
    Path ckptPath = TestHelpers.getResourcePath("/checkpoint.ckpt");
    try (OrtCheckpointState ckpt = OrtCheckpointState.loadCheckpoint(ckptPath)) {
      // Must be non-null, exists so the try block isn't empty as this call will
      // throw if it fails, and throwing errors the test
      Assertions.assertNotNull(ckpt);
    }
  }

  @Test
  public void testCreateTrainingSession() throws OrtException {
    String ckptPath = TestHelpers.getResourcePath("/checkpoint.ckpt").toString();
    String trainPath = TestHelpers.getResourcePath("/training_model.onnx").toString();
    try (OrtTrainingSession trainingSession =
        env.createTrainingSession(ckptPath, trainPath, null, null)) {
      Assertions.assertNotNull(trainingSession);
      Set<String> inputNames = trainingSession.getTrainInputNames();
      Assertions.assertFalse(inputNames.isEmpty());
      Assertions.assertTrue(inputNames.contains("input-0"));
      Set<String> outputNames = trainingSession.getTrainOutputNames();
      Assertions.assertFalse(outputNames.isEmpty());
      Assertions.assertTrue(outputNames.contains("onnx::loss::21273"));
    }
  }

  @Test
  public void testCreateTrainingSessionWithEval() throws OrtException {
    String ckptPath = TestHelpers.getResourcePath("/checkpoint.ckpt").toString();
    String trainPath = TestHelpers.getResourcePath("/training_model.onnx").toString();
    String evalPath = TestHelpers.getResourcePath("/eval_model.onnx").toString();
    try (OrtTrainingSession trainingSession =
        env.createTrainingSession(ckptPath, trainPath, evalPath, null)) {
      Assertions.assertNotNull(trainingSession);
      Set<String> inputNames = trainingSession.getEvalInputNames();
      Assertions.assertFalse(inputNames.isEmpty());
      Assertions.assertTrue(inputNames.contains("input-0"));
      Set<String> outputNames = trainingSession.getEvalOutputNames();
      Assertions.assertFalse(outputNames.isEmpty());
      Assertions.assertTrue(outputNames.contains("onnx::loss::21273"));
    }
  }

  @Test
  public void testTrainingSessionTrainStep() throws OrtException {
    String checkpointPath = TestHelpers.getResourcePath("/checkpoint.ckpt").toString();
    String trainingPath = TestHelpers.getResourcePath("/training_model.onnx").toString();
    float[] expectedOutput =
        TestHelpers.loadTensorFromFile(TestHelpers.getResourcePath("/loss_1.out"));
    float[] input = TestHelpers.loadTensorFromFile(TestHelpers.getResourcePath("/input-0.in"));
    try (OrtTrainingSession trainingSession =
        env.createTrainingSession(checkpointPath, trainingPath, null, null)) {
      int[] labels = {1, 1};

      // Run train step with pinned inputs and pinned outputs
      Map<String, OnnxTensor> pinnedInputs = new HashMap<>();
      Map<String, OnnxTensor> outputMap = new HashMap<>();
      try {
        // Create inputs
        long[] inputShape = {2, 784};
        pinnedInputs.put(
            "input-0", OnnxTensor.createTensor(env, OrtUtil.reshape(input, inputShape)));

        // long[] labelsShape = {2};
        pinnedInputs.put("labels", OnnxTensor.createTensor(env, labels));

        // Prepare output buffer
        FloatBuffer output =
            ByteBuffer.allocateDirect(4 * expectedOutput.length)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        OnnxTensor outputTensor = OnnxTensor.createTensor(env, output, new long[0]);
        outputMap.put("onnx::loss::21273", outputTensor);
        try (OrtSession.Result r = trainingSession.trainStep(pinnedInputs, outputMap)) {
          Assertions.assertEquals(expectedOutput[0], (float) outputTensor.getValue(), 1e-3f);
        }
      } finally {
        OnnxValue.close(outputMap);
        OnnxValue.close(pinnedInputs);
      }
    }
  }

  void runTrainStep(OrtTrainingSession trainingSession) throws OrtException {
    float[] expectedOutput =
        TestHelpers.loadTensorFromFile(TestHelpers.getResourcePath("/loss_1.out"));
    float[] input = TestHelpers.loadTensorFromFile(TestHelpers.getResourcePath("/input-0.in"));
    int[] labels = {1, 1};

    // Run inference with pinned inputs and pinned outputs

    // Create inputs
    Map<String, OnnxTensor> pinnedInputs = new HashMap<>();
    try {
      long[] inputShape = {2, 784};
      pinnedInputs.put("input-0", OnnxTensor.createTensor(env, OrtUtil.reshape(input, inputShape)));

      // long[] labelsShape = {2};
      pinnedInputs.put("labels", OnnxTensor.createTensor(env, labels));

      try (OrtSession.Result firstOutput = trainingSession.trainStep(pinnedInputs)) {
        Assertions.assertTrue(firstOutput.size() > 0);
      }
      trainingSession.lazyResetGrad();
      try (OrtSession.Result secondOutputs = trainingSession.trainStep(pinnedInputs)) {
        OnnxValue outputBuffer = secondOutputs.get(0);

        Assertions.assertEquals(secondOutputs.get("onnx::loss::21273").get(), outputBuffer);
        Assertions.assertTrue(outputBuffer instanceof OnnxTensor);

        OnnxTensor outLabelTensor = (OnnxTensor) outputBuffer;
        Assertions.assertEquals(
            OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, outLabelTensor.getInfo().onnxType);
        Assertions.assertNotNull(outLabelTensor);
        Assertions.assertEquals(expectedOutput[0], (float) outLabelTensor.getValue(), 1e-3f);
      }
    } finally {
      OnnxValue.close(pinnedInputs);
    }
  }

  @Test
  public void testTrainingSessionTrainStepOrtOutput() throws OrtException {
    String checkpointPath = TestHelpers.getResourcePath("/checkpoint.ckpt").toString();
    String trainingPath = TestHelpers.getResourcePath("/training_model.onnx").toString();
    try (OrtTrainingSession trainingSession =
        env.createTrainingSession(checkpointPath, trainingPath, null, null)) {
      runTrainStep(trainingSession);
    }
  }

  @Test
  public void testSaveCheckpoint() throws IOException, OrtException {
    String checkpointPath = TestHelpers.getResourcePath("/checkpoint.ckpt").toString();
    String trainingPath = TestHelpers.getResourcePath("/training_model.onnx").toString();

    Path tmpPath = Files.createTempDirectory("ort-java-training-test");
    Path tmpCheckpointPath = tmpPath.resolve("checkpoint.ckpt");
    try {
      try (OrtTrainingSession trainingSession =
          env.createTrainingSession(checkpointPath, trainingPath, null, null)) {

        // Save checkpoint
        trainingSession.saveCheckpoint(tmpCheckpointPath, false);
      }

      try (OrtTrainingSession trainingSession =
          env.createTrainingSession(tmpCheckpointPath.toString(), trainingPath, null, null)) {
        // Load saved checkpoint into new session and run train step
        runTrainStep(trainingSession);
      }
    } finally {
      TestHelpers.deleteDirectoryTree(tmpPath);
    }
  }

  @Test
  public void testTrainingSessionOptimizerStep() throws OrtException {
    String checkpointPath = TestHelpers.getResourcePath("/checkpoint.ckpt").toString();
    String trainingPath = TestHelpers.getResourcePath("/training_model.onnx").toString();
    String optimizerPath = TestHelpers.getResourcePath("/adamw.onnx").toString();
    float[] expectedOutput_1 =
        TestHelpers.loadTensorFromFile(TestHelpers.getResourcePath("/loss_1.out"));
    float[] expectedOutput_2 =
        TestHelpers.loadTensorFromFile(TestHelpers.getResourcePath("/loss_2.out"));
    float[] input = TestHelpers.loadTensorFromFile(TestHelpers.getResourcePath("/input-0.in"));
    try (OrtTrainingSession trainingSession =
        env.createTrainingSession(checkpointPath, trainingPath, null, optimizerPath)) {
      int[] labels = {1, 1};

      // Run train step with pinned inputs and pinned outputs
      Map<String, OnnxTensor> pinnedInputs = new HashMap<>();
      try {
        // Create inputs
        long[] inputShape = {2, 784};
        pinnedInputs.put(
            "input-0", OnnxTensor.createTensor(env, OrtUtil.reshape(input, inputShape)));

        // long[] labelsShape = {2};
        pinnedInputs.put("labels", OnnxTensor.createTensor(env, labels));

        try (OrtSession.Result outputs = trainingSession.trainStep(pinnedInputs)) {
          Assertions.assertEquals(expectedOutput_1[0], (float) outputs.get(0).getValue(), 1e-3f);
        }

        trainingSession.lazyResetGrad();

        try (OrtSession.Result outputs = trainingSession.trainStep(pinnedInputs)) {
          Assertions.assertEquals(expectedOutput_1[0], (float) outputs.get(0).getValue(), 1e-3f);
        }

        trainingSession.optimizerStep();

        try (OrtSession.Result outputs = trainingSession.trainStep(pinnedInputs)) {
          Assertions.assertEquals(expectedOutput_2[0], (float) outputs.get(0).getValue(), 1e-3f);
        }
      } finally {
        OnnxValue.close(pinnedInputs);
      }
    }
  }

  @Test
  public void testTrainingSessionSetLearningRate() throws OrtException {
    String checkpointPath = TestHelpers.getResourcePath("/checkpoint.ckpt").toString();
    String trainingPath = TestHelpers.getResourcePath("/training_model.onnx").toString();
    String optimizerPath = TestHelpers.getResourcePath("/adamw.onnx").toString();

    try (OrtTrainingSession trainingSession =
        env.createTrainingSession(checkpointPath, trainingPath, null, optimizerPath)) {
      float learningRate = 0.245f;
      trainingSession.setLearningRate(learningRate);
      float actualLearningRate = trainingSession.getLearningRate();
      Assertions.assertEquals(learningRate, actualLearningRate);
    }
  }

  @Test
  public void testTrainingSessionLinearLRScheduler() throws OrtException {
    String checkpointPath = TestHelpers.getResourcePath("/checkpoint.ckpt").toString();
    String trainingPath = TestHelpers.getResourcePath("/training_model.onnx").toString();
    String optimizerPath = TestHelpers.getResourcePath("/adamw.onnx").toString();

    try (OrtTrainingSession trainingSession =
        env.createTrainingSession(checkpointPath, trainingPath, null, optimizerPath)) {
      float learningRate = 0.1f;
      trainingSession.registerLinearLRScheduler(2, 4, learningRate);
      runTrainStep(trainingSession);
      trainingSession.optimizerStep();
      trainingSession.schedulerStep();
      Assertions.assertEquals(0.05f, trainingSession.getLearningRate());
      trainingSession.optimizerStep();
      trainingSession.schedulerStep();
      Assertions.assertEquals(0.1f, trainingSession.getLearningRate());
      trainingSession.optimizerStep();
      trainingSession.schedulerStep();
      Assertions.assertEquals(0.05f, trainingSession.getLearningRate());
      trainingSession.optimizerStep();
      trainingSession.schedulerStep();
      Assertions.assertEquals(0.0f, trainingSession.getLearningRate());
    }
  }

  @Test
  public void testTrainingSessionExportModelForInferencing() throws IOException, OrtException {

    String ckptPath = TestHelpers.getResourcePath("/checkpoint.ckpt").toString();
    String trainPath = TestHelpers.getResourcePath("/training_model.onnx").toString();
    String evalPath = TestHelpers.getResourcePath("/eval_model.onnx").toString();
    try (OrtTrainingSession trainingSession =
        env.createTrainingSession(ckptPath, trainPath, evalPath, null)) {
      String[] graphOutputs = new String[] {"output-0"};

      Path inferencePath = Files.createTempFile("inference_model", ".onnx");

      trainingSession.exportModelForInference(inferencePath, graphOutputs);
      Assertions.assertTrue(inferencePath.toFile().exists());
      inferencePath.toFile().delete();
    }
  }

  @Test
  public void testCheckpointStateAddIntProperty() throws OrtException {
    Path ckptPath = TestHelpers.getResourcePath("/checkpoint.ckpt");
    try (OrtCheckpointState ckpt = OrtCheckpointState.loadCheckpoint(ckptPath)) {
      String propertyName = "days in a week";
      ckpt.addProperty(propertyName, 7);

      int value = ckpt.getIntProperty(env.defaultAllocator, propertyName);
      Assertions.assertEquals(7, value);

      try {
        String strVal = ckpt.getStringProperty(env.defaultAllocator, propertyName);
        Assertions.fail("Should have thrown");
      } catch (OrtException e) {
        // pass
      }

      try {
        float floatVal = ckpt.getFloatProperty(env.defaultAllocator, propertyName);
        Assertions.fail("Should have thrown");
      } catch (OrtException e) {
        // pass
      }
    }
  }

  @Test
  public void testCheckpointStateAddFloatProperty() throws OrtException {
    Path ckptPath = TestHelpers.getResourcePath("/checkpoint.ckpt");
    try (OrtCheckpointState ckpt = OrtCheckpointState.loadCheckpoint(ckptPath)) {
      String propertyName = "pi";
      ckpt.addProperty(propertyName, 3.14f);

      float value = ckpt.getFloatProperty(env.defaultAllocator, propertyName);
      Assertions.assertEquals(3.14f, value);

      try {
        String strVal = ckpt.getStringProperty(env.defaultAllocator, propertyName);
        Assertions.fail("Should have thrown");
      } catch (OrtException e) {
        // pass
      }

      try {
        int intVal = ckpt.getIntProperty(env.defaultAllocator, propertyName);
        Assertions.fail("Should have thrown");
      } catch (OrtException e) {
        // pass
      }
    }
  }

  @Test
  public void testCheckpointStateAddStringProperty() throws OrtException {
    Path ckptPath = TestHelpers.getResourcePath("/checkpoint.ckpt");
    try (OrtCheckpointState ckpt = OrtCheckpointState.loadCheckpoint(ckptPath)) {
      String propertyName = "best ai framework";
      ckpt.addProperty(propertyName, "onnxruntime");

      String value = ckpt.getStringProperty(env.defaultAllocator, propertyName);
      Assertions.assertEquals("onnxruntime", value);

      try {
        float floatVal = ckpt.getFloatProperty(env.defaultAllocator, propertyName);
        Assertions.fail("Should have thrown");
      } catch (OrtException e) {
        // pass
      }

      try {
        int intVal = ckpt.getIntProperty(env.defaultAllocator, propertyName);
        Assertions.fail("Should have thrown");
      } catch (OrtException e) {
        // pass
      }
    }
  }
}
