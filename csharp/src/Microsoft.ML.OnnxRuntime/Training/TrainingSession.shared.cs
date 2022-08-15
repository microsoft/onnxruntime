// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.OnnxRuntime
{
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
        /// A pointer to a underlying native instance of OrtSession
        /// </summary>
        private IntPtr _nativeHandle;

        private ulong _trainOutputCount;
        private ulong _evalOutputCount;
        private List<string> _trainOutputNames;
        private List<string> _evalOutputNames;

        private SessionOptions _builtInSessionOptions = null;
        private RunOptions _builtInRunOptions = null;
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
            Init(null, state, NativeMethods.GetPlatformSerializedString(trainModelPath), NativeMethods.GetPlatformSerializedString(evalModelPath), NativeMethods.GetPlatformSerializedString(optimizerModelPath));
        }

        /// <summary>
        /// Creates TrainingSession from the model and checkpoint in <paramref name="state"/>.
        /// </summary>
        /// <param name="state">Model checkpoint loaded into <see cref="CheckpointState"/>.</param>
        /// <param name="trainModelPath">Specify path to training model graph.</param>
        /// <param name="optimizerModelPath">Specify path to optimizer model graph.</param>
        public TrainingSession(CheckpointState state, string trainModelPath, string optimizerModelPath)
        {
            Init(null, state, NativeMethods.GetPlatformSerializedString(trainModelPath), null, NativeMethods.GetPlatformSerializedString(optimizerModelPath));
        }

        /// <summary>
        /// Creates TrainingSession from the model and checkpoint in <paramref name="state"/>.
        /// </summary>
        /// <param name="state">Model checkpoint loaded into <see cref="CheckpointState"/>.</param>
        /// <param name="trainModelPath">Specify path to training model graph.</param>
        public TrainingSession(CheckpointState state, string trainModelPath)
        {
            Init(null, state, NativeMethods.GetPlatformSerializedString(trainModelPath), null, null);
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
            Init(options, state, NativeMethods.GetPlatformSerializedString(trainModelPath), NativeMethods.GetPlatformSerializedString(evalModelPath), NativeMethods.GetPlatformSerializedString(optimizerModelPath));
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
            using (var cleanupList = new DisposableList<IDisposable>())
            {
                IntPtr[] inputValuesArray = GetOrtValuesHandles(inputValues, true);

                var ortValues = new DisposableList<OrtValue>((int)_trainOutputCount);
                cleanupList.Add(ortValues);

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
            using (var cleanupList = new DisposableList<IDisposable>())
            {
                IntPtr[] inputValuesArray = GetOrtValuesHandles(inputValues, true);

                var ortValues = new DisposableList<OrtValue>((int)_trainOutputCount);
                cleanupList.Add(ortValues);

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
        public void ResetGrad()
        {
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtResetGrad(_nativeHandle));
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
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtTrainStep(_nativeHandle, options.Handle, (UIntPtr)inputValues.Count,
                inputValuesArray, (UIntPtr)outputValues.Count, outputValuesArray));
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
        /// Saves a checkpoint to path. It can be loaded into <see cref="CheckpointState"/>
        /// </summary>
        /// <param name="path">Specify path for saving the checkpoint.</param>
        /// <param name="saveOptimizerState">SFlag indicating whether to save optimizer state or not.</param>
        public void SaveCheckpoint(string path, bool saveOptimizerState = false)
        {
            NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtSaveCheckpoint(NativeMethods.GetPlatformSerializedString(path),_nativeHandle, saveOptimizerState));
        }

        #endregion
        #region private methods

        private void Init(SessionOptions sessOptions, CheckpointState state, byte[] trainModelPath, byte[] evalModelPath, byte[] optimizerModelPath)
        {
            if (!NativeTrainingMethods.TrainingEnabled())
            {
                throw new InvalidOperationException("Training is disabled in the current build.");
            }
            var options = sessOptions;
            if (sessOptions == null)
            {
                _builtInSessionOptions = new SessionOptions();
                options = _builtInSessionOptions;
            }
            var envHandle = OrtEnv.Handle;
            try
            {
                NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtCreateTrainingSession(envHandle, options.Handle, state.Handle, trainModelPath,
                                                                                     evalModelPath, optimizerModelPath, out _nativeHandle));

                UIntPtr outputCount = UIntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtGetTrainModelOutputCount(_nativeHandle, out outputCount));
                _trainOutputCount = outputCount.ToUInt64();

                // get all the output names and metadata
                _trainOutputNames = new List<string>();
                for (ulong i = 0; i < _trainOutputCount; i++)
                {
                    _trainOutputNames.Add(GetOutputName(i, true));
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
                }

                _builtInRunOptions = new RunOptions();  // create a default built-in run option, and avoid creating a new one every run() call
            } catch
            {
                Dispose();
            }
        }

        private string GetOutputName(ulong index, bool training)
        {
            var allocator = OrtAllocator.DefaultInstance;
            IntPtr nameHandle = IntPtr.Zero;
            string str = null;
            if (training)
            { NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtGetTrainModelOutputName(
                                           _nativeHandle,
                                           (UIntPtr)index,
                                           allocator.Pointer,
                                           out nameHandle));
            } else
            { NativeApiStatus.VerifySuccess(NativeTrainingMethods.OrtGetEvalModelOutputName(
                                           _nativeHandle,
                                           (UIntPtr)index,
                                           allocator.Pointer,
                                           out nameHandle));
            }

            using (var ortAllocation = new OrtMemoryAllocation(allocator, nameHandle, 0))
            {
                str = NativeOnnxValueHelper.StringFromNativeUtf8(nameHandle);
            }

            return str;
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
        /// Finalizer. to cleanup session in case it runs
        /// and the user forgets to Dispose() of the session
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

            if (disposing)
            {
                // cleanup managed resources
                if (_builtInSessionOptions != null)
                {
                    _builtInSessionOptions.Dispose();
                    _builtInSessionOptions = null;
                }

                if (_builtInRunOptions != null)
                {
                    _builtInRunOptions.Dispose();
                    _builtInRunOptions = null;
                }
            }

            // cleanup unmanaged resources
            if (_nativeHandle != IntPtr.Zero)
            {
                NativeTrainingMethods.OrtReleaseTrainingSession(_nativeHandle);
                _nativeHandle = IntPtr.Zero;
            }
            _disposed = true;
        }

        #endregion
    }
}
