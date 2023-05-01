// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Linq;
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
    /// Represents a Training Session on an ONNX Model.
    /// This is a IDisposable class and it must be disposed of
    /// using either a explicit call to Dispose() method or
    /// a pattern of using() block. If this is a member of another
    /// class that class must also become IDisposable and it must
    /// dispose of TrainingSession in its Dispose() method.
    /// </summary>
    public class TrainingSession : IDisposable
    {
        /// <summary>
        /// A pointer to a underlying native instance of OrtTrainingSession
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
        /// Creates TrainingSession from the model and checkpoint in <paramref name="state"/>.
        /// </summary>
        /// <param name="state">Model checkpoint loaded into <see cref="CheckpointState"/>.</param>
        /// <param name="trainModelPath">Specify path to training model graph.</param>
        /// <param name="evalModelPath">Specify path to eval model graph.</param>
        /// <param name="optimizerModelPath">Specify path to optimizer model graph.</param>
        public TrainingSession(CheckpointState state, string trainModelPath, string evalModelPath, string optimizerModelPath)
        {
            Init(null, state, NativeOnnxValueHelper.GetPlatformSerializedString(trainModelPath), NativeOnnxValueHelper.GetPlatformSerializedString(evalModelPath), NativeOnnxValueHelper.GetPlatformSerializedString(optimizerModelPath));
        }

        /// <summary>
        /// Creates TrainingSession from the model and checkpoint in <paramref name="state"/>.
        /// </summary>
        /// <param name="state">Model checkpoint loaded into <see cref="CheckpointState"/>.</param>
        /// <param name="trainModelPath">Specify path to training model graph.</param>
        /// <param name="optimizerModelPath">Specify path to optimizer model graph.</param>
        public TrainingSession(CheckpointState state, string trainModelPath, string optimizerModelPath)
        {
            Init(null, state, NativeOnnxValueHelper.GetPlatformSerializedString(trainModelPath), null, NativeOnnxValueHelper.GetPlatformSerializedString(optimizerModelPath));
        }

        /// <summary>
        /// Creates TrainingSession from the model and checkpoint in <paramref name="state"/>.
        /// </summary>
        /// <param name="state">Model checkpoint loaded into <see cref="CheckpointState"/>.</param>
        /// <param name="trainModelPath">Specify path to training model graph.</param>
        public TrainingSession(CheckpointState state, string trainModelPath)
        {
            Init(null, state, NativeOnnxValueHelper.GetPlatformSerializedString(trainModelPath), null, null);
        }


        /// <summary>
        /// Creates TrainingSession from the model and checkpoint in <paramref name="state"/>.
        /// </summary>
        /// <param name="options">Session options</param>
        /// <param name="state">Model checkpoint loaded into <see cref="CheckpointState"/>.</param>
        /// <param name="trainModelPath">Specify path to training model graph.</param>
        /// <param name="evalModelPath">Specify path to eval model graph.</param>
        /// <param name="optimizerModelPath">Specify path to optimizer model graph.</param>
        public TrainingSession(SessionOptions options, CheckpointState state, string trainModelPath, string evalModelPath, string optimizerModelPath)
        {
            Init(options, state, NativeOnnxValueHelper.GetPlatformSerializedString(trainModelPath), NativeOnnxValueHelper.GetPlatformSerializedString(evalModelPath), NativeOnnxValueHelper.GetPlatformSerializedString(optimizerModelPath));
        }

        /// <summary>
        /// Runs a train step on the loaded model for the given inputs.
        /// </summary>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the output values.</param>
        public void TrainStep(
           IReadOnlyCollection<FixedBufferOnnxValue> inputValues,
           IReadOnlyCollection<FixedBufferOnnxValue> outputValues)
        {
            TrainStep(_builtInRunOptions, inputValues, outputValues);
        }

        /// <summary>
        /// Runs a train step on the loaded model for the given inputs. Uses the given RunOptions for this run.
        /// </summary>
        /// <param name="options">Specify <see cref="RunOptions"/> for step.</param>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the output values.</param>
        public void TrainStep(
            RunOptions options,
            IReadOnlyCollection<FixedBufferOnnxValue> inputValues,
            IReadOnlyCollection<FixedBufferOnnxValue> outputValues)
        {
            if (_trainOutputCount!= (ulong)outputValues.Count())
            {
                throw new ArgumentException($"Length of {nameof(outputValues)} ({outputValues.Count}) must match that of train model ({_trainOutputCount}).");
            }
            IntPtr[] inputValuesArray = GetOrtValuesHandles(inputValues, true);

            IntPtr[] outputValuesArray = GetOrtValuesHandles(outputValues, false); /* pointers to Pre-allocated OrtValue instances */
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtTrainStep(_nativeHandle, options.Handle, (UIntPtr)inputValues.Count,
                inputValuesArray, (UIntPtr)outputValues.Count, outputValuesArray));
        }

        /// <summary>
        /// Runs the loaded model for the given inputs, and fetches the graph outputs.
        /// </summary>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values.</param>
        /// <returns>Output Tensors in a Collection of NamedOnnxValue. User must dispose the output.</returns>
        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> TrainStep(
            IReadOnlyCollection<FixedBufferOnnxValue> inputValues)
        {
            using (var ortValues = new DisposableList<OrtValue>((int)_trainOutputCount))
            {
                IntPtr[] inputValuesArray = GetOrtValuesHandles(inputValues, true);
                IntPtr[] outputValuesArray = new IntPtr[(int)_trainOutputCount];

                NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtTrainStep(_nativeHandle, _builtInRunOptions.Handle, (UIntPtr)inputValues.Count,
                    inputValuesArray, (UIntPtr)_trainOutputCount, outputValuesArray));
                foreach (var v in outputValuesArray)
                {
                    ortValues.Add(new OrtValue(v));
                }

                var result = new DisposableList<DisposableNamedOnnxValue>(_trainOutputNames.Count);
                try
                {
                    for (int i = 0; i < ortValues.Count; i++)
                    {
                        var ortValue = ortValues[i];
                        result.Add(DisposableNamedOnnxValue.CreateFromOrtValue(_trainOutputNames[i], ortValue));
                    }
                }
                catch (OnnxRuntimeException)
                {
                    result.Dispose();
                    throw;
                }
                return result;
            }
        }

        /// <summary>
        /// Runs the loaded model for the given inputs, and fetches the specified outputs in <paramref name="outputNames"/>. Uses the given RunOptions for this run.
        /// </summary>
        /// <param name="options">Specify <see cref="RunOptions"/> for step.</param>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values.</param>
        /// <returns>Output Tensors in a Collection of NamedOnnxValue. User must dispose the output.</returns>
        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> TrainStep(
            RunOptions options,
            IReadOnlyCollection<FixedBufferOnnxValue> inputValues)
        {
            using (var ortValues = new DisposableList<OrtValue>((int)_trainOutputCount))
            {
                IntPtr[] inputValuesArray = GetOrtValuesHandles(inputValues, true);
                IntPtr[] outputValuesArray = new IntPtr[(int)_trainOutputCount];

                NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtTrainStep(_nativeHandle, options.Handle, (UIntPtr)inputValues.Count,
                    inputValuesArray, (UIntPtr)_trainOutputCount, outputValuesArray));
                foreach (var v in outputValuesArray)
                {
                    ortValues.Add(new OrtValue(v));
                }

                var result = new DisposableList<DisposableNamedOnnxValue>(_trainOutputNames.Count);
                try
                {
                    for (int i = 0; i < ortValues.Count; i++)
                    {
                        var ortValue = ortValues[i];
                        result.Add(DisposableNamedOnnxValue.CreateFromOrtValue(_trainOutputNames[i], ortValue));
                    }
                }
                catch (OnnxRuntimeException)
                {
                    result.Dispose();
                    throw;
                }
                return result;
            }
        }

        /// <summary>
        /// Sets the reset grad flag on the training graph. The gradient buffers will be reset while executing the
        /// next train step.
        /// </summary>
        public void LazyResetGrad()
        {
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtLazyResetGrad(_nativeHandle));
        }

        /// <summary>
        /// Runs an eval step on the loaded model for the given inputs. The eval graph must be passed while TrainingSession creation.
        /// </summary>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the output values.</param>
        public void EvalStep(
            IReadOnlyCollection<FixedBufferOnnxValue> inputValues,
            IReadOnlyCollection<FixedBufferOnnxValue> outputValues)
        {
            EvalStep(_builtInRunOptions, inputValues, outputValues);
        }

        /// <summary>
        /// Runs an eval step on the loaded model for the given inputs. The eval graph must be passed while TrainingSession creation.
        /// </summary>
        /// <param name="options">Specify <see cref="RunOptions"/> for step.</param>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the output values.</param>
        public void EvalStep(
            RunOptions options,
            IReadOnlyCollection<FixedBufferOnnxValue> inputValues,
            IReadOnlyCollection<FixedBufferOnnxValue> outputValues)
        {
            if (!_evalOutputCount.Equals(outputValues.Count))
            {
                throw new ArgumentException($"Length of {nameof(outputValues)} ({outputValues.Count}) must match that of train model ({_trainOutputCount}).");
            }
            IntPtr[] inputValuesArray = GetOrtValuesHandles(inputValues, true);

            IntPtr[] outputValuesArray = GetOrtValuesHandles(outputValues, false); /* pointers to Pre-allocated OrtValue instances */
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtEvalStep(_nativeHandle, options.Handle, (UIntPtr)inputValues.Count,
                inputValuesArray, (UIntPtr)outputValues.Count, outputValuesArray));
        }


        /// <summary>
        /// Sets a constant learning rate for the session. LR must be controlled by either this method
        /// or by registering a LR scheduler.
        /// </summary>
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
        /// Gets the current learning rate for the session.
        /// </summary>
        public float GetLearningRate()
        {
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtGetLearningRate(_nativeHandle, out float lr));
            return lr;
        }

        /// <summary>
        /// Registers a linear learning rate scheduler for the session. LR must be controlled by either
        /// the SetLearningRate method or by registering a LR scheduler.
        /// <param name="warmupStepCount"> Number of warmup steps</param>
        /// <param name="totalStepCount"> Number of total steps</param>
        /// <param name="initialLearningRate"> Initial learning rate</param>
        /// </summary>
        public void RegisterLinearLRScheduler(long warmupStepCount,
                                              long totalStepCount,
                                              float initialLearningRate)
        {
            if (_scheduler != LRScheduler.None && _scheduler != LRScheduler.Constant)
            {
                throw new InvalidOperationException("Cannot set LR scheduler while using constant LR.");
            }

            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtRegisterLinearLRScheduler(_nativeHandle, warmupStepCount,totalStepCount, initialLearningRate));
            _scheduler = LRScheduler.Linear;
        }

        /// <summary>
        /// Runs a LR scheduler step. There must be a valid LR scheduler registered for the training session.
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
        /// Runs an optimizer step on the loaded model for the given inputs. The optimizer graph must be passed while TrainingSession creation.
        /// </summary>
        public void OptimizerStep()
        {
            OptimizerStep(_builtInRunOptions);
        }

        /// <summary>
        /// Runs an eval step on the loaded model for the given inputs. The eval graph must be passed while TrainingSession creation.
        /// </summary>
        /// <param name="options">Specify <see cref="RunOptions"/> for step.</param>
        /// <param name="outputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the output values.</param>
        public void OptimizerStep(RunOptions options)
        {
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtOptimizerStep(_nativeHandle, options.Handle));

        }

        /// <summary>
        /// Export a model that can be used for inferencing.
        /// If the training session was provided with an eval model, the training session can generate
        /// an inference model if it knows the inference graph outputs. The input inference graph outputs
        /// are used to prune the eval model so that the inference model's outputs align with the provided outputs.
        /// The exported model is saved at the path provided and can be used for inferencing with Ort::Session.
        /// Note that the function re-loads the eval model from the path provided to Ort::TrainingSession
        /// and expects that this path still be valid.
        /// </summary>
        /// <param name="inference_model_path">Path where the inference model should be serialized to.</param>
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
        /// <param name="only_trainable">Whether to only copy trainable parameters or to copy all parameters.</param>
        public FixedBufferOnnxValue ToBuffer(bool only_trainable)
        {
            UIntPtr bufferSize = UIntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtGetParametersSize(_nativeHandle, out bufferSize, only_trainable));

            float[] bufferMemory = new float[bufferSize.ToUInt64()];

            var memInfo = OrtMemoryInfo.DefaultInstance; // CPU
            var shape = new long[] {(long)bufferSize.ToUInt64()};
            var buffer = FixedBufferOnnxValue.CreateFromMemory<float>(memInfo, bufferMemory, Tensors.TensorElementType.Float, shape, (long)bufferSize.ToUInt64() * sizeof(float));

            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtCopyParametersToBuffer(_nativeHandle, buffer.Value.Handle, only_trainable));

            return buffer;
        }

        /// <summary>
        /// Loads the training session model parameters from a contiguous buffer
        /// </summary>
        /// <param name="buffer">Contiguous buffer to load the parameters from.</param>
        public void FromBuffer(FixedBufferOnnxValue buffer)
        {
            if (buffer.OnnxValueType != OnnxValueType.ONNX_TYPE_TENSOR)
            {
                throw new ArgumentException("Incorrect buffer received. Expected a tensor buffer.");
            }

            IntPtr typeAndShapeInfo = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorTypeAndShape(buffer.Value.Handle, out typeAndShapeInfo));
            UIntPtr numDimensions = UIntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetDimensionsCount(typeAndShapeInfo, out numDimensions));
            if (numDimensions.ToUInt64() != 1)
            {
                string errorMessage = "Incorrect buffer shape received. Expected a contiguous tensor buffer. Expected number of dimensions: 1, Actual: " + numDimensions.ToString();
                throw new ArgumentException(errorMessage);
            }

            IntPtr numElementsTrainingOnly = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorShapeElementCount(typeAndShapeInfo, out numElementsTrainingOnly));

            UIntPtr bufferSize = UIntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtGetParametersSize(_nativeHandle, out bufferSize, true));
            if ((long)bufferSize.ToUInt64() == numElementsTrainingOnly.ToInt64())
            {
                NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtCopyBufferToParameters(_nativeHandle, buffer.Value.Handle, true));
                return;
            }

            IntPtr numElements = IntPtr.Zero;
            bufferSize = UIntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtGetParametersSize(_nativeHandle, out bufferSize, false));
            if ((long)bufferSize.ToUInt64() != numElements.ToInt64())
            {
                string errorMessage = "Incorrect buffer size received. Expected size to be one of " + numElementsTrainingOnly.ToString() + " (training only) or " + numElements.ToString() + " (all parameters). Actual size: " + bufferSize.ToString();
                throw new ArgumentException(errorMessage);
            }

            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtCopyBufferToParameters(_nativeHandle, buffer.Value.Handle, false));
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
