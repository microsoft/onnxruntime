// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.IO;
using Xunit;
using Xunit.Abstractions;

#if __TRAINING_ENABLED_NATIVE_BUILD__
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Generic;
using System.Linq;
#endif

// This runs in a separate package built from EndToEndTests
// and for this reason it can not refer to non-public members
// of Onnxruntime package
namespace Microsoft.ML.OnnxRuntime.Tests
{
    public partial class TrainingTest
    {
        private readonly ITestOutputHelper output;

        public TrainingTest(ITestOutputHelper o)
        {
            this.output = o;
        }

#if !__TRAINING_ENABLED_NATIVE_BUILD__
        [Fact(DisplayName = "TestLoadCheckpointThrows")]
        public void TestLoadCheckpointThrows()
        {
            string path = Path.Combine(Directory.GetCurrentDirectory(), "checkpoint.ckpt");
            var ex = Assert.Throws<InvalidOperationException>(() => { var opt = CheckpointState.LoadCheckpoint(path); });
            Assert.Contains("Please install the Microsoft.ML.OnnxRuntime.Training NuGet package.", ex.Message);
        }
#endif

#if __TRAINING_ENABLED_NATIVE_BUILD__
        [Fact(DisplayName = "TestLoadCheckpoint")]
        public void TestLoadCheckpoint()
        {
            string path = Path.Combine(Directory.GetCurrentDirectory(), "checkpoint.ckpt");
            using (var opt = CheckpointState.LoadCheckpoint(path))
            {
                Assert.NotNull(opt);
            }
        }

        [Fact(DisplayName = "TestCreateTrainingSession")]
        public void TestCreateTrainingSession()
        {
            string checkpointPath = Path.Combine(Directory.GetCurrentDirectory(), "checkpoint.ckpt");
            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                var state = CheckpointState.LoadCheckpoint(checkpointPath);
                cleanUp.Add(state);
                Assert.NotNull(state);
                string trainingPath = Path.Combine(Directory.GetCurrentDirectory(), "training_model.onnx");
                string optimizerPath = Path.Combine(Directory.GetCurrentDirectory(), "adamw.onnx");

                var trainingSession = new TrainingSession(state, trainingPath, optimizerPath);
                cleanUp.Add(trainingSession);
            }
        }

        [Fact(DisplayName = "TestTrainingSessionTrainStep")]
        public void TestTrainingSessionTrainStep()
        {
            string checkpointPath = Path.Combine(Directory.GetCurrentDirectory(), "checkpoint.ckpt");
            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                var state = CheckpointState.LoadCheckpoint(checkpointPath);
                cleanUp.Add(state);
                Assert.NotNull(state);
                string trainingPath = Path.Combine(Directory.GetCurrentDirectory(), "training_model.onnx");
                string optimizerPath = Path.Combine(Directory.GetCurrentDirectory(), "adamw.onnx");

                var trainingSession = new TrainingSession(state, trainingPath, optimizerPath);
                cleanUp.Add(trainingSession);

                float[] expectedOutput = TestDataLoader.LoadTensorFromFile("loss_1.out");
                var expectedOutputDimensions = new int[] { 1 };
                float[] input = TestDataLoader.LoadTensorFromFile("input-0.in");
                Int32[] labels = { 1, 1 };

                // Run inference with pinned inputs and pinned outputs
                using (DisposableListTest<FixedBufferOnnxValue> pinnedInputs = new DisposableListTest<FixedBufferOnnxValue>(),
                                                            pinnedOutputs = new DisposableListTest<FixedBufferOnnxValue>())
                {
                    var memInfo = OrtMemoryInfo.DefaultInstance; // CPU

                    // Create inputs
                    long[] inputShape = { 2, 784 };
                    pinnedInputs.Add(FixedBufferOnnxValue.CreateFromMemory<float>(memInfo, input,
                        TensorElementType.Float, inputShape, input.Length * sizeof(float)));

                    long[] labelsShape = { 2 };
                    pinnedInputs.Add(FixedBufferOnnxValue.CreateFromMemory<Int32>(memInfo, labels,
                        TensorElementType.Int32, labelsShape, labels.Length * sizeof(Int32)));


                    // Prepare output buffer
                    long[] outputShape = { };
                    float[] outputBuffer = new float[expectedOutput.Length];
                    pinnedOutputs.Add(FixedBufferOnnxValue.CreateFromMemory<float>(memInfo, outputBuffer,
                        TensorElementType.Float, outputShape, outputBuffer.Length * sizeof(float)));

                    trainingSession.TrainStep(pinnedInputs, pinnedOutputs);
                    Assert.Equal(expectedOutput, outputBuffer, new FloatComparer());
                }
            }
        }

        void RunTrainStep(TrainingSession trainingSession)
        {
            float[] expectedOutput = TestDataLoader.LoadTensorFromFile("loss_1.out");
            var expectedOutputDimensions = new int[] { 1 };
            float[] input = TestDataLoader.LoadTensorFromFile("input-0.in");
            Int32[] labels = { 1, 1 };

            // Run inference with pinned inputs and pinned outputs
            using (DisposableListTest<FixedBufferOnnxValue> pinnedInputs = new DisposableListTest<FixedBufferOnnxValue>())
            {
                var memInfo = OrtMemoryInfo.DefaultInstance; // CPU

                // Create inputs
                long[] inputShape = { 2, 784 };
                pinnedInputs.Add(FixedBufferOnnxValue.CreateFromMemory<float>(memInfo, input,
                    TensorElementType.Float, inputShape, input.Length * sizeof(float)));

                long[] labelsShape = { 2 };
                pinnedInputs.Add(FixedBufferOnnxValue.CreateFromMemory<Int32>(memInfo, labels,
                    TensorElementType.Int32, labelsShape, labels.Length * sizeof(Int32)));

                var outputs = trainingSession.TrainStep(pinnedInputs);
                trainingSession.LazyResetGrad();
                outputs = trainingSession.TrainStep(pinnedInputs);
                var outputBuffer = outputs.ElementAtOrDefault(0);

                Assert.Equal("onnx::loss::21273", outputBuffer.Name);
                Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, outputBuffer.ValueType);
                Assert.Equal(TensorElementType.Float, outputBuffer.ElementType);

                var outLabelTensor = outputBuffer.AsTensor<float>();
                Assert.NotNull(outLabelTensor);
                Assert.Equal(expectedOutput, outLabelTensor.ToArray(), new FloatComparer());
            }
        }

        [Fact(DisplayName = "TestTrainingSessionTrainStepOrtOutput")]
        public void TestTrainingSessionTrainStepOrtOutput()
        {
            string checkpointPath = Path.Combine(Directory.GetCurrentDirectory(), "checkpoint.ckpt");
            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                var state = CheckpointState.LoadCheckpoint(checkpointPath);
                cleanUp.Add(state);
                Assert.NotNull(state);
                string trainingPath = Path.Combine(Directory.GetCurrentDirectory(), "training_model.onnx");
                string optimizerPath = Path.Combine(Directory.GetCurrentDirectory(), "adamw.onnx");

                var trainingSession = new TrainingSession(state, trainingPath, optimizerPath);
                cleanUp.Add(trainingSession);
                RunTrainStep(trainingSession);
            }
        }


        [Fact(DisplayName = "TestSaveCheckpoint")]
        public void TestSaveCheckpoint()
        {
            string checkpointPath = Path.Combine(Directory.GetCurrentDirectory(), "checkpoint.ckpt");
            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                var state = CheckpointState.LoadCheckpoint(checkpointPath);
                cleanUp.Add(state);
                Assert.NotNull(state);
                string trainingPath = Path.Combine(Directory.GetCurrentDirectory(), "training_model.onnx");
                string optimizerPath = Path.Combine(Directory.GetCurrentDirectory(), "adamw.onnx");
                var trainingSession = new TrainingSession(state, trainingPath, optimizerPath);
                cleanUp.Add(trainingSession);

                // Save checkpoint
                string savedCheckpointPath = Path.Combine(Directory.GetCurrentDirectory(), "saved_checkpoint.ckpt");
                CheckpointState.SaveCheckpoint(state, savedCheckpointPath, true);

                // Load checkpoint and run train step
                var loadedState = CheckpointState.LoadCheckpoint(savedCheckpointPath);
                cleanUp.Add(loadedState);
                var newTrainingSession = new TrainingSession(loadedState, trainingPath, optimizerPath);
                cleanUp.Add(newTrainingSession);
                RunTrainStep(newTrainingSession);
            }
        }

        [Fact(DisplayName = "TestTrainingSessionOptimizerStep")]
        public void TestTrainingSessionOptimizerStep()
        {
            string checkpointPath = Path.Combine(Directory.GetCurrentDirectory(), "checkpoint.ckpt");
            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                var state = CheckpointState.LoadCheckpoint(checkpointPath);
                cleanUp.Add(state);
                Assert.NotNull(state);
                string trainingPath = Path.Combine(Directory.GetCurrentDirectory(), "training_model.onnx");
                string optimizerPath = Path.Combine(Directory.GetCurrentDirectory(), "adamw.onnx");

                var trainingSession = new TrainingSession(state, trainingPath, optimizerPath);
                cleanUp.Add(trainingSession);

                float[] expectedOutput_1 = TestDataLoader.LoadTensorFromFile("loss_1.out");
                float[] expectedOutput_2 = TestDataLoader.LoadTensorFromFile("loss_2.out");
                var expectedOutputDimensions = new int[] { 1 };
                float[] input = TestDataLoader.LoadTensorFromFile("input-0.in");
                Int32[] labels = { 1, 1 };

                // Run train step with pinned inputs and pinned outputs
                using (DisposableListTest<FixedBufferOnnxValue> pinnedInputs = new DisposableListTest<FixedBufferOnnxValue>(),
                                                            pinnedOutputs = new DisposableListTest<FixedBufferOnnxValue>())
                {
                    var memInfo = OrtMemoryInfo.DefaultInstance; // CPU

                    // Create inputs
                    long[] inputShape = { 2, 784 };
                    pinnedInputs.Add(FixedBufferOnnxValue.CreateFromMemory<float>(memInfo, input,
                        TensorElementType.Float, inputShape, input.Length * sizeof(float)));

                    long[] labelsShape = { 2 };
                    pinnedInputs.Add(FixedBufferOnnxValue.CreateFromMemory<Int32>(memInfo, labels,
                        TensorElementType.Int32, labelsShape, labels.Length * sizeof(Int32)));


                    // Prepare output buffer
                    long[] outputShape = { };
                    float[] outputBuffer = new float[expectedOutput_1.Length];
                    pinnedOutputs.Add(FixedBufferOnnxValue.CreateFromMemory<float>(memInfo, outputBuffer,
                        TensorElementType.Float, outputShape, outputBuffer.Length * sizeof(float)));

                    trainingSession.TrainStep(pinnedInputs, pinnedOutputs);
                    Assert.Equal(expectedOutput_1, outputBuffer, new FloatComparer());

                    trainingSession.LazyResetGrad();

                    trainingSession.TrainStep(pinnedInputs, pinnedOutputs);
                    Assert.Equal(expectedOutput_1, outputBuffer, new FloatComparer());

                    trainingSession.OptimizerStep();

                    trainingSession.TrainStep(pinnedInputs, pinnedOutputs);
                    Assert.Equal(expectedOutput_2, outputBuffer, new FloatComparer());
                }
            }
        }

        [Fact(DisplayName = "TestTrainingSessionSetLearningRate")]
        public void TestTrainingSessionSetLearningRate()
        {
            string checkpointPath = Path.Combine(Directory.GetCurrentDirectory(), "checkpoint.ckpt");
            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                var state = CheckpointState.LoadCheckpoint(checkpointPath);
                cleanUp.Add(state);
                Assert.NotNull(state);
                string trainingPath = Path.Combine(Directory.GetCurrentDirectory(), "training_model.onnx");
                string optimizerPath = Path.Combine(Directory.GetCurrentDirectory(), "adamw.onnx");

                var trainingSession = new TrainingSession(state, trainingPath, optimizerPath);
                cleanUp.Add(trainingSession);

                float learningRate = 0.245f;
                trainingSession.SetLearningRate(learningRate);
                var actualLearningRate = trainingSession.GetLearningRate();
                Assert.Equal(learningRate, actualLearningRate);
            }
        }

        [Fact(DisplayName = "TestTrainingSessionLinearLRScheduler")]
        public void TestTrainingSessionLinearLRScheduler()
        {
            string checkpointPath = Path.Combine(Directory.GetCurrentDirectory(), "checkpoint.ckpt");
            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                var state = CheckpointState.LoadCheckpoint(checkpointPath);
                cleanUp.Add(state);
                Assert.NotNull(state);
                string trainingPath = Path.Combine(Directory.GetCurrentDirectory(), "training_model.onnx");
                string optimizerPath = Path.Combine(Directory.GetCurrentDirectory(), "adamw.onnx");

                var trainingSession = new TrainingSession(state, trainingPath, optimizerPath);
                cleanUp.Add(trainingSession);

                float learningRate = 0.1f;
                trainingSession.RegisterLinearLRScheduler(2, 4, learningRate);
                RunTrainStep(trainingSession);
                trainingSession.OptimizerStep();
                trainingSession.SchedulerStep();
                Assert.Equal(0.05f, trainingSession.GetLearningRate());
                trainingSession.OptimizerStep();
                trainingSession.SchedulerStep();
                Assert.Equal(0.1f, trainingSession.GetLearningRate());
                trainingSession.OptimizerStep();
                trainingSession.SchedulerStep();
                Assert.Equal(0.05f, trainingSession.GetLearningRate());
                trainingSession.OptimizerStep();
                trainingSession.SchedulerStep();
                Assert.Equal(0.0f, trainingSession.GetLearningRate());
            }
        }

        [Fact(DisplayName = "TestTrainingSessionExportModelForInferencing")]
        public void TestTrainingSessionExportModelForInferencing()
        {
            string checkpointPath = Path.Combine(Directory.GetCurrentDirectory(), "checkpoint.ckpt");
            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                var state = CheckpointState.LoadCheckpoint(checkpointPath);
                cleanUp.Add(state);
                Assert.NotNull(state);
                string trainingPath = Path.Combine(Directory.GetCurrentDirectory(), "training_model.onnx");
                string evalPath = Path.Combine(Directory.GetCurrentDirectory(), "eval_model.onnx");
                string optimizerPath = Path.Combine(Directory.GetCurrentDirectory(), "adamw.onnx");

                var trainingSession = new TrainingSession(state, trainingPath, evalPath, optimizerPath);
                cleanUp.Add(trainingSession);

                var graphOutputs = new List<string>(){"output-0"};

                string inferencePath = Path.Combine(Directory.GetCurrentDirectory(), "inference_model.onnx");

                trainingSession.ExportModelForInferencing(inferencePath, graphOutputs);
                Assert.True(File.Exists(inferencePath));
            }
        }

        [Fact(DisplayName = "TestCheckpointStateAddProperty")]
        public void TestCheckpointStateAddProperty()
        {
            string checkpointPath = Path.Combine(Directory.GetCurrentDirectory(), "checkpoint.ckpt");
            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                var state = CheckpointState.LoadCheckpoint(checkpointPath);
                cleanUp.Add(state);
                Assert.NotNull(state);

                string propertyName = "days in a week";
                state.AddProperty(propertyName, (long)7);

                var value = state.GetProperty(propertyName);
                Assert.True(value is long);
                Assert.Equal((long)7, value);
            }
        }

        [Fact(DisplayName = "TestCheckpointStateAddFloatProperty")]
        public void TestCheckpointStateAddFloatProperty()
        {
            string checkpointPath = Path.Combine(Directory.GetCurrentDirectory(), "checkpoint.ckpt");
            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                var state = CheckpointState.LoadCheckpoint(checkpointPath);
                cleanUp.Add(state);
                Assert.NotNull(state);

                string propertyName = "pi";
                state.AddProperty(propertyName, (float)3.14);

                var value = state.GetProperty(propertyName);
                Assert.True(value is float);
                Assert.Equal((float)3.14, value);
            }
        }

        [Fact(DisplayName = "TestCheckpointStateAddStringProperty")]
        public void TestCheckpointStateAddStringProperty()
        {
            string checkpointPath = Path.Combine(Directory.GetCurrentDirectory(), "checkpoint.ckpt");
            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                var state = CheckpointState.LoadCheckpoint(checkpointPath);
                cleanUp.Add(state);
                Assert.NotNull(state);

                string propertyName = "best ai framework";
                state.AddProperty(propertyName, "onnxruntime");

                var value = state.GetProperty(propertyName);
                Assert.True(value is string);
                Assert.Equal("onnxruntime", value);
            }
        }

        [Fact(DisplayName = "TestTrainModelInputNames")]
        public void TestTrainModelInputNames()
        {
            string checkpointPath = Path.Combine(Directory.GetCurrentDirectory(), "checkpoint.ckpt");
            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                var state = CheckpointState.LoadCheckpoint(checkpointPath);
                cleanUp.Add(state);
                Assert.NotNull(state);
                string trainingPath = Path.Combine(Directory.GetCurrentDirectory(), "training_model.onnx");
                string optimizerPath = Path.Combine(Directory.GetCurrentDirectory(), "adamw.onnx");
                var trainingSession = new TrainingSession(state, trainingPath, optimizerPath);
                cleanUp.Add(trainingSession);

                var inputNames = trainingSession.InputNames(true);

                Assert.True(inputNames.Count == 2);
                Assert.Equal("input-0", inputNames[0]);
                Assert.Equal("labels", inputNames[1]);
            }
        }

        [Fact(DisplayName = "TestEvalModelInputNames")]
        public void TestEvalModelInputNames()
        {
            string checkpointPath = Path.Combine(Directory.GetCurrentDirectory(), "checkpoint.ckpt");
            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                var state = CheckpointState.LoadCheckpoint(checkpointPath);
                cleanUp.Add(state);
                Assert.NotNull(state);
                string trainingPath = Path.Combine(Directory.GetCurrentDirectory(), "training_model.onnx");
                string evalPath = Path.Combine(Directory.GetCurrentDirectory(), "eval_model.onnx");
                string optimizerPath = Path.Combine(Directory.GetCurrentDirectory(), "adamw.onnx");

                var trainingSession = new TrainingSession(state, trainingPath, evalPath, optimizerPath);
                cleanUp.Add(trainingSession);

                var inputNames = trainingSession.InputNames(false);

                Assert.True(inputNames.Count == 2);
                Assert.Equal("input-0", inputNames[0]);
                Assert.Equal("labels", inputNames[1]);
            }
        }

        [Fact(DisplayName = "TestTrainModelOutputNames")]
        public void TestTrainModelOutputNames()
        {
            string checkpointPath = Path.Combine(Directory.GetCurrentDirectory(), "checkpoint.ckpt");
            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                var state = CheckpointState.LoadCheckpoint(checkpointPath);
                cleanUp.Add(state);
                Assert.NotNull(state);
                string trainingPath = Path.Combine(Directory.GetCurrentDirectory(), "training_model.onnx");
                string optimizerPath = Path.Combine(Directory.GetCurrentDirectory(), "adamw.onnx");
                var trainingSession = new TrainingSession(state, trainingPath, optimizerPath);
                cleanUp.Add(trainingSession);

                var outputNames = trainingSession.OutputNames(true);

                Assert.Single(outputNames);
                Assert.Equal("onnx::loss::21273", outputNames[0]);
            }
        }

        [Fact(DisplayName = "TestEvalModelOutputNames")]
        public void TestEvalModelOutputNames()
        {
            string checkpointPath = Path.Combine(Directory.GetCurrentDirectory(), "checkpoint.ckpt");
            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                var state = CheckpointState.LoadCheckpoint(checkpointPath);
                cleanUp.Add(state);
                Assert.NotNull(state);
                string trainingPath = Path.Combine(Directory.GetCurrentDirectory(), "training_model.onnx");
                string evalPath = Path.Combine(Directory.GetCurrentDirectory(), "eval_model.onnx");
                string optimizerPath = Path.Combine(Directory.GetCurrentDirectory(), "adamw.onnx");

                var trainingSession = new TrainingSession(state, trainingPath, evalPath, optimizerPath);
                cleanUp.Add(trainingSession);

                var outputNames = trainingSession.OutputNames(false);

                Assert.Single(outputNames);
                Assert.Equal("onnx::loss::21273", outputNames[0]);
            }
        }

        [Fact(DisplayName = "TestToBuffer")]
        public void TestToBuffer()
        {
            string checkpointPath = Path.Combine(Directory.GetCurrentDirectory(), "checkpoint.ckpt");
            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                var state = CheckpointState.LoadCheckpoint(checkpointPath);
                cleanUp.Add(state);
                Assert.NotNull(state);
                string trainingPath = Path.Combine(Directory.GetCurrentDirectory(), "training_model.onnx");
                string evalPath = Path.Combine(Directory.GetCurrentDirectory(), "eval_model.onnx");
                string optimizerPath = Path.Combine(Directory.GetCurrentDirectory(), "adamw.onnx");

                var trainingSession = new TrainingSession(state, trainingPath, evalPath, optimizerPath);
                cleanUp.Add(trainingSession);

                var buffer = trainingSession.ToBuffer(true);
                cleanUp.Add(buffer);
            }
        }

        [Fact(DisplayName = "TestFromBuffer")]
        public void TestFromBuffer()
        {
            string checkpointPath = Path.Combine(Directory.GetCurrentDirectory(), "checkpoint.ckpt");
            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                var state = CheckpointState.LoadCheckpoint(checkpointPath);
                cleanUp.Add(state);
                Assert.NotNull(state);
                string trainingPath = Path.Combine(Directory.GetCurrentDirectory(), "training_model.onnx");
                string evalPath = Path.Combine(Directory.GetCurrentDirectory(), "eval_model.onnx");
                string optimizerPath = Path.Combine(Directory.GetCurrentDirectory(), "adamw.onnx");

                var trainingSession = new TrainingSession(state, trainingPath, evalPath, optimizerPath);
                cleanUp.Add(trainingSession);

                var buffer = trainingSession.ToBuffer(true);
                cleanUp.Add(buffer);

                trainingSession.FromBuffer(buffer);
            }
        }

        [Fact(DisplayName = "TestSetSeed")]
        public void TestSetSeed()
        {
            TrainingUtils.SetSeed(8888);
        }

        internal class FloatComparer : IEqualityComparer<float>
        {
            private float atol = 1e-3f;
            private float rtol = 1.7e-2f;

            public bool Equals(float x, float y)
            {
                return Math.Abs(x - y) <= (atol + rtol * Math.Abs(y));
            }
            public int GetHashCode(float x)
            {
                return x.GetHashCode();
            }
        }
#endif
    }
}
