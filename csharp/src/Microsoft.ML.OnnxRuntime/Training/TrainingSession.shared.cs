// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntime
{
#if __ENABLE_TRAINING_APIS__
    /// <summary>
    /// This class defines utility methods for training.
    /// </summary>
    public class TrainingUtils
    {
        /// <summary>
        /// Use this function to generate reproducible results. It should be noted that completely
        /// reproducible results are not guaranteed.
        /// </summary>
        /// <param name="seed">Manual seed to use for random number generation.</param>
        public static void SetSeed(long seed)
        {
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtSetSeed(seed));
        }
    }

    enum LRScheduler
    {
        None = 0,
        Constant = 1,
        Linear = 2
    }

    /// <summary>
    /// Trainer class that provides training, evaluation and optimizer methods for training an ONNX model.
    ///
    /// The training session requires four training artifacts
    /// - The training onnx model
    /// - The evaluation onnx model (optional)
    /// - The optimizer onnx model
    /// - The checkpoint directory
    ///
    /// These artifacts can be generated using the `onnxruntime-training` python [utility](https://github.com/microsoft/onnxruntime/blob/main/orttraining/orttraining/python/training/onnxblock/README.md).
    ///
    /// This is an IDisposable class and it must be disposed of
    /// using either an explicit call to Dispose() method or
    /// a pattern of using() block. If this is a member of another
    /// class that class must also become IDisposable and it must
    /// dispose of TrainingSession in its Dispose() method.
    /// </summary>
    public class TrainingSession : IDisposable
    {
        /// <summary>
        /// A pointer to an underlying native instance of OrtTrainingSession
        /// </summary>
        private IntPtr _nativeHandle;

        private ulong _trainOutputCount;
        private ulong _evalOutputCount;
        private List<string> _trainOutputNames;
        private List<string> _evalOutputNames;
        private List<string> _trainInputNames;
        private List<string> _evalInputNames;

        private SessionOptions _builtInSessionOptions = null;
        private RunOptions _builtInRunOptions = null;
        private LRScheduler _scheduler = LRScheduler.None;
        private bool _disposed = false;

        #region Public API

        /// <summary>
        /// Create a training session that can be used to begin or resume training.
        ///
        /// This constructor instantiates the training session based on the env and session options provided that can
        /// begin or resume training from a given checkpoint state for the given onnx models.
        /// The checkpoint state represents the parameters of the training session which will be moved
        /// to the device specified by the user through the session options (if necessary).
        /// </summary>
        /// <param name="state">Training states that the training session uses as a starting point for training.</param>
        /// <param name="trainModelPath">Model to be used to perform training.</param>
        /// <param name="evalModelPath">Model to be used to perform evaluation.</param>
        /// <param name="optimizerModelPath">Model to be used to perform weight update.</param>
        public TrainingSession(CheckpointState state, string trainModelPath, string evalModelPath, string optimizerModelPath)
        {
            Init(null, state, NativeOnnxValueHelper.GetPlatformSerializedString(trainModelPath), NativeOnnxValueHelper.GetPlatformSerializedString(evalModelPath), NativeOnnxValueHelper.GetPlatformSerializedString(optimizerModelPath));
        }

        /// <summary>
        /// Create a training session that can be used to begin or resume training.
        ///
        /// This constructor instantiates the training session based on the env and session options provided that can
        /// begin or resume training from a given checkpoint state for the given onnx models.
        /// The checkpoint state represents the parameters of the training session which will be moved
        /// to the device specified by the user through the session options (if necessary).
        /// </summary>
        /// <param name="state">Training states that the training session uses as a starting point for training.</param>
        /// <param name="trainModelPath">Model to be used to perform training.</param>
        /// <param name="optimizerModelPath">Model to be used to perform weight update.</param>
        public TrainingSession(CheckpointState state, string trainModelPath, string optimizerModelPath)
        {
            Init(null, state, NativeOnnxValueHelper.GetPlatformSerializedString(trainModelPath), null, NativeOnnxValueHelper.GetPlatformSerializedString(optimizerModelPath));
        }

        /// <summary>
        /// Create a training session that can be used to begin or resume training.
        ///
        /// This constructor instantiates the training session based on the env and session options provided that can
        /// begin or resume training from a given checkpoint state for the given onnx models.
        /// The checkpoint state represents the parameters of the training session which will be moved
        /// to the device specified by the user through the session options (if necessary).
        /// </summary>
        /// <param name="options">SessionOptions that the user can customize for this training session.</param>
        /// <param name="state">Training states that the training session uses as a starting point for training.</param>
        /// <param name="trainModelPath">Model to be used to perform training.</param>
        /// <param name="evalModelPath">Model to be used to perform evaluation.</param>
        /// <param name="optimizerModelPath">Model to be used to perform weight update.</param>
        public TrainingSession(SessionOptions options, CheckpointState state, string trainModelPath, string evalModelPath, string optimizerModelPath)
        {
            Init(options, state, NativeOnnxValueHelper.GetPlatformSerializedString(trainModelPath), NativeOnnxValueHelper.GetPlatformSerializedString(evalModelPath), NativeOnnxValueHelper.GetPlatformSerializedString(optimizerModelPath));
        }

        /// <summary>
        /// Computes the outputs of the training model and the gradients of the trainable parameters for the given inputs
        ///
        /// This function performs a training step that computes the outputs of the training model and the gradients
        /// of the trainable parameters for the given inputs. The train step is performed based on the training model
        /// that was provided to the training session.
        /// The TrainStep method is equivalent of running forward propagation and backward propagation in a single
        /// step.
        /// The gradients computed are stored inside the training session state so they can be later consumed
        /// by the OptimizerStep function.
        /// The gradients can be lazily reset by invoking the LazyResetGrad function.
        /// </summary>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values to the training model.</param>
        /// <param name="outputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the output values of the training model.</param>
        public void TrainStep(
           IReadOnlyCollection<FixedBufferOnnxValue> inputValues,
           IReadOnlyCollection<FixedBufferOnnxValue> outputValues)
        {
            TrainStep(_builtInRunOptions, inputValues, outputValues);
        }

        /// <summary>
        /// Computes the outputs of the training model and the gradients of the trainable parameters for the given inputs
        ///
        /// This function performs a training step that computes the outputs of the training model and the gradients
        /// of the trainable parameters for the given inputs. The train step is performed based on the training model
        /// that was provided to the training session.
        /// The TrainStep method is equivalent of running forward propagation and backward propagation in a single
        /// step.
        /// The gradients computed are stored inside the training session state so they can be later consumed
        /// by the OptimizerStep function.
        /// The gradients can be lazily reset by invoking the LazyResetGrad function.
        /// </summary>
        /// <param name="options">Specify <see cref="RunOptions"/> for step.</param>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values to the training model.</param>
        /// <param name="outputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the output values of the training model.</param>
        public void TrainStep(
            RunOptions options,
            IReadOnlyCollection<FixedBufferOnnxValue> inputValues,
            IReadOnlyCollection<FixedBufferOnnxValue> outputValues)
        {
            if (_trainOutputCount != (ulong)outputValues.Count())
            {
                throw new ArgumentException($"Length of {nameof(outputValues)} ({outputValues.Count}) must match that of train model ({_trainOutputCount}).");
            }
            IntPtr[] inputValuesArray = GetOrtValuesHandles(inputValues, true);

            IntPtr[] outputValuesArray = GetOrtValuesHandles(outputValues, false); /* pointers to Pre-allocated OrtValue instances */
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtTrainStep(_nativeHandle, options.Handle, (UIntPtr)inputValues.Count,
                inputValuesArray, (UIntPtr)outputValues.Count, outputValuesArray));
        }

        /// <summary>
        /// Computes the outputs of the training model and the gradients of the trainable parameters for the given inputs
        ///
        /// This function performs a training step that computes the outputs of the training model and the gradients
        /// of the trainable parameters for the given inputs. The train step is performed based on the training model
        /// that was provided to the training session.
        /// The TrainStep method is equivalent of running forward propagation and backward propagation in a single
        /// step.
        /// The gradients computed are stored inside the training session state so they can be later consumed
        /// by the OptimizerStep function.
        /// The gradients can be lazily reset by invoking the LazyResetGrad function.
        /// </summary>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values to the training model.</param>
        /// <returns>Output Tensors in a Collection of NamedOnnxValue. User must dispose the output.</returns>
        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> TrainStep(
            IReadOnlyCollection<FixedBufferOnnxValue> inputValues)
        {
            IntPtr[] inputValuesArray = GetOrtValuesHandles(inputValues, true);
            IntPtr[] outputValuesArray = new IntPtr[(int)_trainOutputCount];

            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtTrainStep(_nativeHandle, _builtInRunOptions.Handle, (UIntPtr)inputValues.Count,
                inputValuesArray, (UIntPtr)_trainOutputCount, outputValuesArray));

            // On success ortValues would contain nulls that will be
            // ignored. On failure, ortValues would contain at least
            // some valid OrtValue instances that need to be disposed.
            // It would be nice to use using() clause, but we need to upgrade to C# 8.0 for that.
            var ortValueDisposer = ConvertNativeHandlesToOrtValues(outputValuesArray);
            try
            {
                var result = new DisposableList<DisposableNamedOnnxValue>(_trainOutputNames.Count);
                try
                {
                    for (int i = 0; i < ortValueDisposer.Span.Length; i++)
                    {
                        result.Add(DisposableNamedOnnxValue.CreateFromOrtValue(_trainOutputNames[i], ref ortValueDisposer.Span[i]));
                    }
                }
                catch (OnnxRuntimeException)
                {
                    result.Dispose();
                    throw;
                }
                return result;
            }
            finally
            {
                // On success ortValues would contain nulls that will be
                // ignored. On failure, ortValues would contain at least
                // some valid OrtValue instances that need to be disposed.
                ortValueDisposer.Dispose();
            }
        }

        /// <summary>
        /// Computes the outputs of the training model and the gradients of the trainable parameters for the given inputs
        ///
        /// This function performs a training step that computes the outputs of the training model and the gradients
        /// of the trainable parameters for the given inputs. The train step is performed based on the training model
        /// that was provided to the training session.
        /// The TrainStep method is equivalent of running forward propagation and backward propagation in a single
        /// step.
        /// The gradients computed are stored inside the training session state so they can be later consumed
        /// by the OptimizerStep function.
        /// The gradients can be lazily reset by invoking the LazyResetGrad function.
        /// </summary>
        /// <param name="options">Specify <see cref="RunOptions"/> for step.</param>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values to the training model.</param>
        /// <returns>Output Tensors in a Collection of NamedOnnxValue. User must dispose the output.</returns>
        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> TrainStep(
            RunOptions options,
            IReadOnlyCollection<FixedBufferOnnxValue> inputValues)
        {
            IntPtr[] inputValuesArray = GetOrtValuesHandles(inputValues, true);
            IntPtr[] outputValuesArray = new IntPtr[(int)_trainOutputCount];

            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtTrainStep(_nativeHandle, options.Handle, (UIntPtr)inputValues.Count,
                inputValuesArray, (UIntPtr)_trainOutputCount, outputValuesArray));


            // On success ortValues would contain nulls that will be
            // ignored. On failure, ortValues would contain at least
            // some valid OrtValue instances that need to be disposed.
            // It would be nice to use using() clause, but we need to upgrade to C# 8.0 for that.
            var ortValueDisposer = ConvertNativeHandlesToOrtValues(outputValuesArray);
            try
            {
                var result = new DisposableList<DisposableNamedOnnxValue>(_trainOutputNames.Count);
                try
                {
                    for (int i = 0; i < ortValueDisposer.Span.Length; i++)
                    {
                        result.Add(DisposableNamedOnnxValue.CreateFromOrtValue(_trainOutputNames[i], ref ortValueDisposer.Span[i]));
                    }
                }
                catch (OnnxRuntimeException)
                {
                    result.Dispose();
                    throw;
                }
                return result;
            }
            finally
            {
                ortValueDisposer.Dispose();
            }
        }

        /// <summary>
        /// Convert native OrtValue handles to OrtValue instances
        /// in an exceptions safe manner.
        /// </summary>
        /// <param name="nativeHandles"></param>
        /// <returns></returns>
        private DisposableArray<OrtValue> ConvertNativeHandlesToOrtValues(IntPtr[] nativeHandles)
        {
            var diposableArray = new DisposableOrtValueHandleArray(nativeHandles);
            try
            {
                var ortValues = new OrtValue[nativeHandles.Length];
                var ortValueDisposer = new DisposableArray<OrtValue>(ortValues);
                try
                {
                    for (int i = 0; i < nativeHandles.Length; i++)
                    {
                        ortValues[i] = new OrtValue(nativeHandles[i]);
                        nativeHandles[i] = IntPtr.Zero;
                    }
                    return ortValueDisposer;
                }
                catch (Exception)
                {
                    // ortValues is the result, dispose only on exception
                    ortValueDisposer.Dispose();
                    throw;
                }
            }
            catch (Exception)
            {
                // No need to dispose on exception since the ownership is transferred to ortValues
                diposableArray.Dispose();
                throw;
            }
        }

        /// <summary>
        /// Reset the gradients of all trainable parameters to zero lazily.
        ///
        /// This function sets the internal state of the training session such that the gradients of the trainable
        /// parameters in the OrtCheckpointState will be scheduled to be reset just before the new gradients are
        /// computed on the next invocation of the next TrainStep.
        /// </summary>
        public void LazyResetGrad()
        {
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtLazyResetGrad(_nativeHandle));
        }

        /// <summary>
        /// Computes the outputs for the eval model for the given inputs
        /// This function performs an eval step that computes the outputs of the eval model for the given inputs.
        /// The eval step is performed based on the eval model that was provided to the training session.
        /// </summary>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values to the eval model.</param>
        /// <param name="outputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the output values of the eval model.</param>
        public void EvalStep(
            IReadOnlyCollection<FixedBufferOnnxValue> inputValues,
            IReadOnlyCollection<FixedBufferOnnxValue> outputValues)
        {
            EvalStep(_builtInRunOptions, inputValues, outputValues);
        }

        /// <summary>
        /// Computes the outputs for the eval model for the given inputs
        /// This function performs an eval step that computes the outputs of the eval model for the given inputs.
        /// The eval step is performed based on the eval model that was provided to the training session.
        /// </summary>
        /// <param name="options">Specify <see cref="RunOptions"/> for step.</param>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values to the eval model.</param>
        /// <param name="outputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the output values of the eval model.</param>
        public void EvalStep(
            RunOptions options,
            IReadOnlyCollection<FixedBufferOnnxValue> inputValues,
            IReadOnlyCollection<FixedBufferOnnxValue> outputValues)
        {
            if (_evalOutputCount != (ulong)outputValues.Count())
            {
                throw new ArgumentException($"Length of {nameof(outputValues)} ({outputValues.Count}) must match that of eval model ({_evalOutputCount}).");
            }
            const bool isInput = true;
            IntPtr[] inputValuesArray = GetOrtValuesHandles(inputValues, isInput);

            IntPtr[] outputValuesArray = GetOrtValuesHandles(outputValues, !isInput); /* pointers to Pre-allocated OrtValue instances */
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtEvalStep(_nativeHandle, options.Handle, (UIntPtr)inputValues.Count,
                inputValuesArray, (UIntPtr)outputValues.Count, outputValuesArray));
        }


        /// <summary>
        /// Sets the learning rate for this training session.
        ///
        /// This function allows users to set the learning rate for the training session. The current
        /// learning rate is maintained by the training session and can be overwritten by invoking
        /// this function with the desired learning rate. This function should not be used when a valid
        /// learning rate scheduler is registered. It should be used either to set the learning rate
        /// derived from a custom learning rate scheduler or to set a constant learning rate to be used
        /// throughout the training session.
        /// <note type="note">
        /// Please note that this function does not set the initial learning rate that may be needed
        /// by the predefined learning rate schedulers. To set the initial learning rate for learning
        /// rate schedulers, please look at the function RegisterLinearLRScheduler.
        /// </note>
        /// </summary>
        /// <param name="learningRate">Desired learning rate to be set.</param>
        public void SetLearningRate(float learningRate)
        {
            if (_scheduler != LRScheduler.None && _scheduler != LRScheduler.Constant)
            {
                throw new InvalidOperationException("Cannot set constant LR while using LR scheduler.");
            }
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtSetLearningRate(_nativeHandle, learningRate));
            _scheduler = LRScheduler.Constant;
        }

        /// <summary>
        /// Gets the current learning rate for this training session.
        ///
        /// This function allows users to get the learning rate for the training session. The current
        /// learning rate is maintained by the training session, and users can query it for the purpose
        /// of implementing their own learning rate schedulers.
        /// </summary>
        /// <returns>float representing the current learning rate.</returns>
        public float GetLearningRate()
        {
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtGetLearningRate(_nativeHandle, out float lr));
            return lr;
        }

        /// <summary>
        /// Registers a linear learning rate scheduler for the training session.
        ///
        /// Register a linear learning rate scheduler that decays the learning rate by linearly updated
        /// multiplicative factor from the initial learning rate set on the training session to 0. The decay
        /// is performed after the initial warm up phase where the learning rate is linearly incremented
        /// from 0 to the initial learning rate provided.
        /// </summary>
        /// <param name="warmupStepCount"> Number of warmup steps</param>
        /// <param name="totalStepCount"> Number of total steps</param>
        /// <param name="initialLearningRate"> Initial learning rate</param>
        public void RegisterLinearLRScheduler(long warmupStepCount,
                                              long totalStepCount,
                                              float initialLearningRate)
        {
            if (_scheduler != LRScheduler.None && _scheduler != LRScheduler.Constant)
            {
                throw new InvalidOperationException("Cannot set LR scheduler while using constant LR.");
            }

            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtRegisterLinearLRScheduler(_nativeHandle, warmupStepCount, totalStepCount, initialLearningRate));
            _scheduler = LRScheduler.Linear;
        }

        /// <summary>
        /// Update the learning rate based on the registered learning rate scheduler.
        ///
        /// Takes a scheduler step that updates the learning rate that is being used by the training session.
        /// This function should typically be called before invoking the optimizer step for each round,
        /// or as determined necessary to update the learning rate being used by the training session.
        /// <note type="note">
        /// Please note that a valid predefined learning rate scheduler must be first registered to invoke this function.
        /// </note>
        /// </summary>
        public void SchedulerStep()
        {
            if (_scheduler == LRScheduler.Constant || _scheduler == LRScheduler.None)
            {
                throw new InvalidOperationException("Cannot take scheduler step without registering a valid LR scheduler.");
            }
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtSchedulerStep(_nativeHandle));
        }

        /// <summary>
        /// Performs the weight updates for the trainable parameters using the optimizer model.
        ///
        /// This function performs the weight update step that updates the trainable parameters such that they
        /// take a step in the direction of their gradients (gradient descent). The optimizer step is performed
        /// based on the optimizer model that was provided to the training session.
        /// The updated parameters are stored inside the training state so that they can be used by the next
        /// TrainStep function call.
        /// </summary>
        public void OptimizerStep()
        {
            OptimizerStep(_builtInRunOptions);
        }

        /// <summary>
        /// Performs the weight updates for the trainable parameters using the optimizer model.
        ///
        /// This function performs the weight update step that updates the trainable parameters such that they
        /// take a step in the direction of their gradients (gradient descent). The optimizer step is performed
        /// based on the optimizer model that was provided to the training session.
        /// The updated parameters are stored inside the training state so that they can be used by the next
        /// TrainStep function call.
        /// </summary>
        /// <param name="options">Specify <see cref="RunOptions"/> for step.</param>
        public void OptimizerStep(RunOptions options)
        {
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtOptimizerStep(_nativeHandle, options.Handle));

        }

        /// <summary>
        /// Export a model that can be used for inferencing.
        /// If the training session was provided with an eval model, the training session can generate
        /// an inference model if it knows the inference graph outputs. The input inference graph outputs
        /// are used to prune the eval model so that the inference model's outputs align with the provided outputs.
        /// The exported model is saved at the path provided and can be used for inferencing with InferenceSession.
        /// Note that the function re-loads the eval model from the path provided to TrainingSession
        /// and expects that this path still be valid.
        /// </summary>
        /// <param name="inferenceModelPath">Path where the inference model should be serialized to.</param>
        /// <param name="graphOutputNames">Names of the outputs that are needed in the inference model.</param>
        public void ExportModelForInferencing(string inferenceModelPath, IReadOnlyCollection<string> graphOutputNames)
        {
            using (var cleanupList = new DisposableList<IDisposable>())
            {
                var outputNamesArray = ConvertNamesToUtf8(graphOutputNames, cleanupList);
                NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtExportModelForInferencing(
                    _nativeHandle, NativeOnnxValueHelper.GetPlatformSerializedString(inferenceModelPath),
                    (UIntPtr)graphOutputNames.Count, outputNamesArray));
            }
        }

        /// <summary>
        /// Returns a contiguous buffer that holds a copy of all training state parameters
        /// </summary>
        /// <param name="onlyTrainable">Whether to only copy trainable parameters or to copy all parameters.</param>
        public OrtValue ToBuffer(bool onlyTrainable)
        {
            UIntPtr bufferSize = UIntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtGetParametersSize(_nativeHandle, out bufferSize, onlyTrainable));

            float[] bufferMemory = new float[bufferSize.ToUInt64()];

            var memInfo = OrtMemoryInfo.DefaultInstance; // CPU
            var shape = new long[] { (long)bufferSize.ToUInt64() };
            var buffer = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, Tensors.TensorElementType.Float, shape);

            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtCopyParametersToBuffer(_nativeHandle, buffer.Handle, onlyTrainable));

            return buffer;
        }

        /// <summary>
        /// Loads the training session model parameters from a contiguous buffer
        /// </summary>
        /// <param name="buffer">Contiguous buffer to load the parameters from.</param>
        public void FromBuffer(OrtValue buffer)
        {
            if (buffer.OnnxType != OnnxValueType.ONNX_TYPE_TENSOR)
            {
                throw new ArgumentException("Incorrect buffer received. Expected a tensor buffer.");
            }

            IntPtr typeAndShapeInfo = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorTypeAndShape(buffer.Handle, out typeAndShapeInfo));
            UIntPtr numDimensions = UIntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetDimensionsCount(typeAndShapeInfo, out numDimensions));
            if (numDimensions.ToUInt64() != 1)
            {
                string errorMessage = "Incorrect buffer shape received. Expected a contiguous tensor buffer. Expected number of dimensions: 1, Actual: " + numDimensions.ToString();
                throw new ArgumentException(errorMessage);
            }

            // Here buffer size represents the number of elements in the buffer
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorShapeElementCount(typeAndShapeInfo, out UIntPtr bufferSize));

            // OrtGetParametersSize returns the total number of elements in the model's parameters.
            UIntPtr numElementsTrainingOnly = UIntPtr.Zero;
            const bool onlyTrainable = true;
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtGetParametersSize(_nativeHandle, out numElementsTrainingOnly, onlyTrainable));
            if ((ulong)bufferSize == (ulong)numElementsTrainingOnly)
            {
                NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtCopyBufferToParameters(_nativeHandle, buffer.Handle, onlyTrainable));
                return;
            }

            UIntPtr numElements = UIntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtGetParametersSize(_nativeHandle, out numElements, !onlyTrainable));
            if ((ulong)bufferSize != (ulong)numElements)
            {
                string errorMessage = "Incorrect buffer size received. Expected size to be one of " + numElementsTrainingOnly.ToString() + " (training only) or " + numElements.ToString() + " (all parameters). Actual size: " + bufferSize.ToString();
                throw new ArgumentException(errorMessage);
            }

            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtCopyBufferToParameters(_nativeHandle, buffer.Handle, !onlyTrainable));
        }

        /// <summary>
        /// Retrieves the names of the user outputs for the training and eval models.
        /// </summary>
        /// <param name="training">Whether the training model output names are requested or eval model output names.</param>
        public List<string> OutputNames(bool training)
        {
            return training ? _trainOutputNames : _evalOutputNames;
        }

        /// <summary>
        /// Retrieves the names of the user inputs for the training and eval models.
        /// </summary>
        /// <param name="training">Whether the training model input names are requested or eval model input names.</param>
        public List<string> InputNames(bool training)
        {
            return training ? _trainInputNames : _evalInputNames;
        }

        #endregion
        #region private methods

        private void Init(SessionOptions sessOptions, CheckpointState state, byte[] trainModelPath, byte[] evalModelPath, byte[] optimizerModelPath)
        {
            if (!NativeTrainingMethods.TrainingEnabled())
            {
                throw new InvalidOperationException("This package does not contain the training API. Please install the Microsoft.ML.OnnxRuntime.Training NuGet package.\n");
            }
            var options = sessOptions;
            if (sessOptions == null)
            {
                _builtInSessionOptions = new SessionOptions();
                options = _builtInSessionOptions;
            }
            var envHandle = OrtEnv.Instance().Handle;
            try
            {
                NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtCreateTrainingSession(envHandle, options.Handle, state.Handle, trainModelPath,
                                                                                     evalModelPath, optimizerModelPath, out _nativeHandle));

                UIntPtr outputCount = UIntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtGetTrainingModelOutputCount(_nativeHandle, out outputCount));
                _trainOutputCount = outputCount.ToUInt64();

                // get all the output names and metadata
                _trainOutputNames = new List<string>();
                for (ulong i = 0; i < _trainOutputCount; i++)
                {
                    _trainOutputNames.Add(GetOutputName(i, true));
                }

                _trainInputNames = new List<string>();
                UIntPtr inputCount = UIntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtGetTrainingModelInputCount(_nativeHandle, out inputCount));
                for (ulong i = 0; i < inputCount.ToUInt64(); i++)
                {
                    _trainInputNames.Add(GetInputName(i, true));
                }

                if (evalModelPath != null)
                {
                    outputCount = UIntPtr.Zero;
                    NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtGetEvalModelOutputCount(_nativeHandle, out outputCount));
                    _evalOutputCount = outputCount.ToUInt64();
                    _evalOutputNames = new List<string>();
                    for (ulong i = 0; i < _evalOutputCount; i++)
                    {
                        _evalOutputNames.Add(GetOutputName(i, false));
                    }

                    _evalInputNames = new List<string>();
                    inputCount = UIntPtr.Zero;
                    NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtGetEvalModelInputCount(_nativeHandle, out inputCount));
                    for (ulong i = 0; i < inputCount.ToUInt64(); i++)
                    {
                        _evalInputNames.Add(GetInputName(i, false));
                    }
                }

                _builtInRunOptions = new RunOptions();  // create a default built-in run option, and avoid creating a new one every run() call
            }
            catch (Exception)
            {
                CleanupHelper(true);
                throw;
            }
        }

        private string GetOutputName(ulong index, bool training)
        {
            var allocator = OrtAllocator.DefaultInstance;
            IntPtr nameHandle;
            if (training)
            {
                NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtGetTrainingModelOutputName(
                                           _nativeHandle,
                                           (UIntPtr)index,
                                           allocator.Pointer,
                                           out nameHandle));
            }
            else
            {
                NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtGetEvalModelOutputName(
                                           _nativeHandle,
                                           (UIntPtr)index,
                                           allocator.Pointer,
                                           out nameHandle));
            }
            return NativeOnnxValueHelper.StringFromNativeUtf8(nameHandle, allocator);
        }

        private string GetInputName(ulong index, bool training)
        {
            var allocator = OrtAllocator.DefaultInstance;
            IntPtr nameHandle;
            if (training)
            {
                NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtGetTrainingModelInputName(
                                           _nativeHandle,
                                           (UIntPtr)index,
                                           allocator.Pointer,
                                           out nameHandle));
            }
            else
            {
                NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtGetEvalModelInputName(
                                           _nativeHandle,
                                           (UIntPtr)index,
                                           allocator.Pointer,
                                           out nameHandle));
            }
            return NativeOnnxValueHelper.StringFromNativeUtf8(nameHandle, allocator);
        }

        private IntPtr[] GetOrtValuesHandles(IReadOnlyCollection<FixedBufferOnnxValue> values, bool input)
        {
            var valuesArray = new IntPtr[values.Count];
            for (int index = 0; index < values.Count; ++index)
            {
                var v = values.ElementAt(index);
                if (!input && v.ElementType == Tensors.TensorElementType.String)
                {
                    throw new NotSupportedException("Using string type FixedBufferOnnxValue in outputs is not supported.");
                }
                valuesArray[index] = v.Value.Handle;
            }
            return valuesArray;
        }

        private IntPtr[] ConvertNamesToUtf8(IReadOnlyCollection<string> names, DisposableList<IDisposable> cleanupList)
        {
            cleanupList.Capacity += names.Count;
            var result = new IntPtr[names.Count];
            for (int i = 0; i < names.Count; ++i)
            {
                var name = names.ElementAt(i);
                var utf8Name = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(name);
                var pinnedHandle = new Memory<byte>(utf8Name).Pin();
                unsafe
                {
                    result[i] = (IntPtr)pinnedHandle.Pointer;
                }
                cleanupList.Add(pinnedHandle);
            }
            return result;
        }

        /// <summary>
        /// Other classes access
        /// </summary>
        internal IntPtr Handle
        {
            get
            {
                return _nativeHandle;
            }
        }

        #endregion

        #region IDisposable

        /// <summary>
        /// Finalizer.
        /// </summary>
        ~TrainingSession()
        {
            Dispose(false);
        }

        /// <summary>
        /// IDisposable implementation
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// IDisposable implementation
        /// </summary>
        /// <param name="disposing">true if invoked from Dispose() method</param>
        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
            {
                return;
            }
            CleanupHelper(disposing);
            _disposed = true;
        }

        private void CleanupHelper(bool disposing)
        {
            if (disposing)
            {
                if (_builtInRunOptions != null)
                {
                    _builtInRunOptions.Dispose();
                    _builtInRunOptions = null;
                }

                if (_builtInSessionOptions != null)
                {
                    _builtInSessionOptions.Dispose();
                    _builtInSessionOptions = null;
                }
            }

            // cleanup unmanaged resources
            if (_nativeHandle != IntPtr.Zero)
            {
                NativeTrainingMethods.OrtReleaseTrainingSession(_nativeHandle);
                _nativeHandle = IntPtr.Zero;
            }
        }

        #endregion
    }
#endif
}
