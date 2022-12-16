/*
 * Copyright © 2022, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import ai.onnxruntime.OrtTrainingSession.OrtCheckpointState;
import ai.onnxruntime.TensorInfo.OnnxTensorType;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Tests for the ORT training apis.
 */
@EnabledIfSystemProperty(named = "ENABLE_TRAINING", matches = "1")
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
    try (OrtTrainingSession trainingSession = env.createTrainingSession(ckptPath, trainPath, null, null)) {
      Assertions.assertNotNull(trainingSession);
      Set<String> inputNames = trainingSession.getTrainInputNames();
      Assertions.assertFalse(inputNames.isEmpty());
      Set<String> outputNames = trainingSession.getTrainOutputNames();
      Assertions.assertFalse(outputNames.isEmpty());
    }
  }

  /* this test is not completed yet as ORT Java doesn't support supplying an output buffer
  @Test
  public void TestTrainingSessionTrainStep() {
    string checkpointPath = Path.Combine(Directory.GetCurrentDirectory(), "checkpoint.ckpt");
    using(var cleanUp = new DisposableListTest<IDisposable>())
    {
      var state = new CheckpointState(checkpointPath);
      cleanUp.Add(state);
      Assertions.NotNull(state);
      string trainingPath = Path.Combine(Directory.GetCurrentDirectory(), "training_model.onnx");

      var trainingSession = new TrainingSession(state, trainingPath);
      cleanUp.Add(trainingSession);

      float[] expectedOutput = TestDataLoader.LoadTensorFromFile("loss_1.out");
      var expectedOutputDimensions = new int[]{1};
      float[] input = TestDataLoader.LoadTensorFromFile("input-0.in");
      Int32[] labels = {1, 1};

      // Run inference with pinned inputs and pinned outputs
      using(DisposableListTest < FixedBufferOnnxValue > pinnedInputs = new DisposableListTest<FixedBufferOnnxValue>(),
          pinnedOutputs = new DisposableListTest<FixedBufferOnnxValue>())
      {
        var memInfo = OrtMemoryInfo.DefaultInstance; // CPU

        // Create inputs
        long[] inputShape = {2, 784};
        pinnedInputs.Add(FixedBufferOnnxValue.CreateFromMemory < float>(memInfo, input,
          TensorElementType.Float, inputShape, input.Length * sizeof( float)));

        long[] labelsShape = {2};
        pinnedInputs.Add(FixedBufferOnnxValue.CreateFromMemory < Int32 > (memInfo, labels,
            TensorElementType.Int32, labelsShape, labels.Length * sizeof(Int32)));


        // Prepare output buffer
        long[] outputShape = {};
        float[] outputBuffer = new float[expectedOutput.Length];
        pinnedOutputs.Add(FixedBufferOnnxValue.CreateFromMemory < float>(memInfo, outputBuffer,
          TensorElementType.Float, outputShape, outputBuffer.Length * sizeof( float)));

        trainingSession.TrainStep(pinnedInputs, pinnedOutputs);
        Assertions.Equal(expectedOutput, outputBuffer, new FloatComparer());
      }
    }
  }
  */

  void runTrainStep(OrtTrainingSession trainingSession) throws OrtException {
    float[] expectedOutput = TestHelpers.loadTensorFromFile(TestHelpers.getResourcePath("./loss_1.out"));
    int[] expectedOutputDimensions = new int[]{1};
    float[] input = TestHelpers.loadTensorFromFile(TestHelpers.getResourcePath("./input-0.in"));
    int[] labels = {1, 1};

    // Run inference with pinned inputs and pinned outputs

    // Create inputs
    Map<String, OnnxTensor> pinnedInputs = new HashMap<>();
    try {
      long[] inputShape = {2, 784};
      pinnedInputs.put("features-tbd", OnnxTensor.createTensor(env, OrtUtil.reshape(input, inputShape)));

      long[] labelsShape = {2};
      pinnedInputs.put("labels-tbd", OnnxTensor.createTensor(env, labels));

      try (OrtSession.Result firstOutput = trainingSession.trainStep(pinnedInputs)) {
          Assertions.assertTrue(firstOutput.size() > 0);
      }
      trainingSession.lazyResetGrad();
      try (OrtSession.Result secondOutputs = trainingSession.trainStep(pinnedInputs)) {
        OnnxValue outputBuffer = secondOutputs.get(0);

        Assertions.assertEquals(secondOutputs.get("onnx::loss::21273").get(), outputBuffer);
        Assertions.assertTrue(outputBuffer instanceof OnnxTensor);

        OnnxTensor outLabelTensor = (OnnxTensor) outputBuffer;
        Assertions.assertEquals(OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, outLabelTensor.getInfo().onnxType);
        Assertions.assertNotNull(outLabelTensor);
        Assertions.assertArrayEquals(expectedOutput, (float[]) outLabelTensor.getValue(), 1e-3f);
      }
    } finally {
      OnnxValue.close(pinnedInputs);
    }
  }

  @Test
  public void TestTrainingSessionTrainStepOrtOutput() throws OrtException {
    String checkpointPath = TestHelpers.getResourcePath("/checkpoint.ckpt").toString();
    String trainingPath = TestHelpers.getResourcePath("/training_model.onnx").toString();
    try (OrtTrainingSession trainingSession = env.createTrainingSession(checkpointPath, trainingPath, null, null)) {
      runTrainStep(trainingSession);
    }
  }

  @Test
  public void TestSaveCheckpoint() throws IOException, OrtException {
    String checkpointPath = TestHelpers.getResourcePath("/checkpoint.ckpt").toString();
    String trainingPath = TestHelpers.getResourcePath("/training_model.onnx").toString();

    Path tmpPath = Files.createTempDirectory("ort-java-training-test");
    try {
      try (OrtTrainingSession trainingSession = env.createTrainingSession(checkpointPath, trainingPath, null, null)) {

        // Save checkpoint
        trainingSession.saveCheckpoint(tmpPath, false);
      }

      try (OrtTrainingSession trainingSession = env.createTrainingSession(tmpPath.toString(), trainingPath, null, null)) {
        // Load saved checkpoint into new session and run train step
        runTrainStep(trainingSession);
      }
    } finally {
      TestHelpers.deleteDirectoryTree(tmpPath);
    }
  }

  /*
  @Test
  public void TestTrainingSessionOptimizerStep() {
    string checkpointPath = Path.Combine(Directory.GetCurrentDirectory(), "checkpoint.ckpt");
    using(var cleanUp = new DisposableListTest<IDisposable>())
    {
      var state = new CheckpointState(checkpointPath);
      cleanUp.Add(state);
      Assertions.NotNull(state);
      string trainingPath = Path.Combine(Directory.GetCurrentDirectory(), "training_model.onnx");
      string optimizerPath = Path.Combine(Directory.GetCurrentDirectory(), "adamw.onnx");

      var trainingSession = new TrainingSession(state, trainingPath, optimizerPath);
      cleanUp.Add(trainingSession);

      float[] expectedOutput_1 = TestDataLoader.LoadTensorFromFile("loss_1.out");
      float[] expectedOutput_2 = TestDataLoader.LoadTensorFromFile("loss_2.out");
      var expectedOutputDimensions = new int[]{1};
      float[] input = TestDataLoader.LoadTensorFromFile("input-0.in");
      Int32[] labels = {1, 1};

      // Run train step with pinned inputs and pinned outputs
      using(DisposableListTest < FixedBufferOnnxValue > pinnedInputs = new DisposableListTest<FixedBufferOnnxValue>(),
          pinnedOutputs = new DisposableListTest<FixedBufferOnnxValue>())
      {
        var memInfo = OrtMemoryInfo.DefaultInstance; // CPU

        // Create inputs
        long[] inputShape = {2, 784};
        pinnedInputs.Add(FixedBufferOnnxValue.CreateFromMemory < float>(memInfo, input,
          TensorElementType.Float, inputShape, input.Length * sizeof( float)));

        long[] labelsShape = {2};
        pinnedInputs.Add(FixedBufferOnnxValue.CreateFromMemory < Int32 > (memInfo, labels,
            TensorElementType.Int32, labelsShape, labels.Length * sizeof(Int32)));


        // Prepare output buffer
        long[] outputShape = {};
        float[] outputBuffer = new float[expectedOutput_1.Length];
        pinnedOutputs.Add(FixedBufferOnnxValue.CreateFromMemory < float>(memInfo, outputBuffer,
          TensorElementType.Float, outputShape, outputBuffer.Length * sizeof( float)));

        trainingSession.TrainStep(pinnedInputs, pinnedOutputs);
        Assertions.Equal(expectedOutput_1, outputBuffer, new FloatComparer());

        trainingSession.LazyResetGrad();

        trainingSession.TrainStep(pinnedInputs, pinnedOutputs);
        Assertions.Equal(expectedOutput_1, outputBuffer, new FloatComparer());

        trainingSession.OptimizerStep();

        trainingSession.TrainStep(pinnedInputs, pinnedOutputs);
        Assertions.Equal(expectedOutput_2, outputBuffer, new FloatComparer());
      }
    }
  }
   */

  @Test
  public void TestTrainingSessionSetLearningRate() throws OrtException {
    String checkpointPath = TestHelpers.getResourcePath("/checkpoint.ckpt").toString();
    String trainingPath = TestHelpers.getResourcePath("/training_model.onnx").toString();
    String optimizerPath = TestHelpers.getResourcePath("/adamw.onnx").toString();

    try (OrtTrainingSession trainingSession = env.createTrainingSession(checkpointPath, trainingPath, null, optimizerPath)) {
      float learningRate = 0.245f;
      trainingSession.setLearningRate(learningRate);
      float actualLearningRate = trainingSession.getLearningRate();
      Assertions.assertEquals(learningRate, actualLearningRate);
    }
  }

  @Test
  public void TestTrainingSessionLinearLRScheduler() throws OrtException {
    String checkpointPath = TestHelpers.getResourcePath("/checkpoint.ckpt").toString();
    String trainingPath = TestHelpers.getResourcePath("/training_model.onnx").toString();
    String optimizerPath = TestHelpers.getResourcePath("/adamw.onnx").toString();

    try (OrtTrainingSession trainingSession = env.createTrainingSession(checkpointPath, trainingPath, null, optimizerPath)) {
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

/*
  internal

  class FloatComparer :IEqualityComparer<float>

  {
    private float atol = 1e-3f;
    private float rtol = 1.7e-2f;

    public bool Equals ( float x, float y)
    {
      return Math.Abs(x - y) <= (atol + rtol * Math.Abs(y));
    }
    public int GetHashCode ( float x)
    {
      return x.GetHashCode();
    }
  }
   */
}
