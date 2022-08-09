// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Buffers;

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

        public TrainingSession(CheckpointState state, string trainModelPath, string evalModelPath, string optimizerModelPath)
        {
            _builtInSessionOptions = new SessionOptions(); // need to be disposed
            Init(_builtInSessionOptions, state, NativeMethods.GetPlatformSerializedString(trainModelPath), NativeMethods.GetPlatformSerializedString(evalModelPath), NativeMethods.GetPlatformSerializedString(optimizerModelPath));
        }

        public TrainingSession(CheckpointState state, string trainModelPath, string optimizerModelPath)
        {
            _builtInSessionOptions = new SessionOptions(); // need to be disposed
            Init(_builtInSessionOptions, state, NativeMethods.GetPlatformSerializedString(trainModelPath), null, NativeMethods.GetPlatformSerializedString(optimizerModelPath));
        }

        public TrainingSession(CheckpointState state, string trainModelPath)
        {
            _builtInSessionOptions = new SessionOptions(); // need to be disposed
            Init(_builtInSessionOptions, state, NativeMethods.GetPlatformSerializedString(trainModelPath), null, null);
        }


        /// <summary>
        /// Constructs an TrainingSession from a model file
        /// </summary>
        /// <param name="modelPath"></param>
        public TrainingSession(SessionOptions options, CheckpointState state, string trainModelPath, string evalModelPath, string optimizerModelPath)
        {
            Init(options, state, NativeMethods.GetPlatformSerializedString(trainModelPath), NativeMethods.GetPlatformSerializedString(evalModelPath), NativeMethods.GetPlatformSerializedString(optimizerModelPath));
        }

        public void TrainStep(
           IReadOnlyCollection<FixedBufferOnnxValue> inputValues,
           IReadOnlyCollection<FixedBufferOnnxValue> outputValues)
        {
            TrainStep(_builtInRunOptions, inputValues, outputValues);
        }

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
            NativeApiStatus.VerifySuccess(NativeMethods.OrtTrainStep(_nativeHandle, options.Handle, (UIntPtr)inputValues.Count,
                inputValuesArray, (UIntPtr)outputValues.Count, outputValuesArray));
        }

        /// <summary>
        /// Runs the loaded model for the given inputs, and fetches the specified outputs in <paramref name="outputNames"/>. Uses the given RunOptions for this run.
        /// </summary>
        /// <param name="inputNames">Specify a collection of string that indicates the input names. Should match <paramref name="inputValues"/>.</param>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputNames">Specify a collection of string that indicates the output names to fetch.</param>
        /// <param name="options"></param>
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
                NativeApiStatus.VerifySuccess(NativeMethods.OrtTrainStep(_nativeHandle, options.Handle, (UIntPtr)inputValues.Count,
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

        public void ResetGrad()
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtResetGrad(_nativeHandle));
        }

        public void EvalStep(
            IReadOnlyCollection<FixedBufferOnnxValue> inputValues,
            IReadOnlyCollection<FixedBufferOnnxValue> outputValues)
        {
            EvalStep(_builtInRunOptions, inputValues, outputValues);
        }

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
            NativeApiStatus.VerifySuccess(NativeMethods.OrtTrainStep(_nativeHandle, options.Handle, (UIntPtr)inputValues.Count,
                inputValuesArray, (UIntPtr)outputValues.Count, outputValuesArray));
        }

        public void OptimizerStep()
        {   
            OptimizerStep(_builtInRunOptions);
        }
        public void OptimizerStep(RunOptions options)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtOptimizerStep(_nativeHandle, options.Handle));

        }

        public void SaveCheckpoint(string path, bool saveOptimizerState = false)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSaveCheckpoint(NativeMethods.GetPlatformSerializedString(path),_nativeHandle, saveOptimizerState));
        }

        #endregion
        #region private methods

        private void Init(SessionOptions options, CheckpointState state, byte[] trainModelPath, byte[] evalModelPath, byte[] optimizerModelPath)
        {
            var envHandle = OrtEnv.Handle;

            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateTrainingSession(envHandle, options.Handle, state.Handle, trainModelPath,
                                                                                 evalModelPath, optimizerModelPath, out _nativeHandle));

            UIntPtr outputCount = UIntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTrainModelOutputCount(_nativeHandle, out outputCount));
            _trainOutputCount = outputCount.ToUInt64();

            // get all the output names and metadata
            for (ulong i = 0; i < _trainOutputCount; i++)
            {
                _trainOutputNames.Add(GetOutputName(i, true));
            }

            if (evalModelPath != null)
            {
                outputCount = UIntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetEvalModelOutputCount(_nativeHandle, out outputCount));
                _evalOutputCount = outputCount.ToUInt64();
                for (ulong i = 0; i < _evalOutputCount; i++)
                {
                    _evalOutputNames.Add(GetOutputName(i, false));
                }
            }

            _builtInRunOptions = new RunOptions();  // create a default built-in run option, and avoid creating a new one every run() call
        }

        private string GetOutputName(ulong index, bool training)
        {
            var allocator = OrtAllocator.DefaultInstance;
            IntPtr nameHandle = IntPtr.Zero;
            string str = null;
            if (training)
            { NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTrainModelOutputName(
                                           _nativeHandle,
                                           (UIntPtr)index,
                                           allocator.Pointer,
                                           out nameHandle));
            } else
            { NativeApiStatus.VerifySuccess(NativeMethods.OrtGetEvalModelOutputName(
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
                NativeMethods.OrtReleaseTrainingSession(_nativeHandle);
                _nativeHandle = IntPtr.Zero;
            }
            _disposed = true;
        }

        #endregion
    }
}
