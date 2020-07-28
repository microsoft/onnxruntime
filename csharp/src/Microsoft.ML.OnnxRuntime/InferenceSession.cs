// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Represents an Inference Session on an ONNX Model
    /// </summary>
    public class InferenceSession : IDisposable
    {
        protected IntPtr _nativeHandle;
        protected Dictionary<string, NodeMetadata> _inputMetadata, _outputMetadata, _overridableInitializerMetadata;
        private SessionOptions _builtInSessionOptions = null;
        private RunOptions _builtInRunOptions = null;


        #region Public API

        /// <summary>
        /// Constructs an InferenceSession from a model file
        /// </summary>
        /// <param name="modelPath"></param>
        public InferenceSession(string modelPath)
        {
            _builtInSessionOptions = new SessionOptions(); // need to be disposed
            Init(modelPath, _builtInSessionOptions);
        }


        /// <summary>
        /// Constructs an InferenceSession from a model file, with some additional session options
        /// </summary>
        /// <param name="modelPath"></param>
        /// <param name="options"></param>
        public InferenceSession(string modelPath, SessionOptions options)
        {
            Init(modelPath, options);
        }

        /// <summary>
        /// Constructs an InferenceSession from a model data in byte array
        /// </summary>
        /// <param name="model"></param>
        public InferenceSession(byte[] model)
        {
            _builtInSessionOptions = new SessionOptions(); // need to be disposed
            Init(model, _builtInSessionOptions);
        }

        /// <summary>
        /// Constructs an InferenceSession from a model data in byte array, with some additional session options
        /// </summary>
        /// <param name="model"></param>
        /// <param name="options"></param>
        public InferenceSession(byte[] model, SessionOptions options)
        {
            Init(model, options);
        }

        /// <summary>
        /// Meta data regarding the input nodes, keyed by input names
        /// </summary>
        public IReadOnlyDictionary<string, NodeMetadata> InputMetadata
        {
            get
            {
                return _inputMetadata;
            }
        }

        /// <summary>
        /// Metadata regarding the output nodes, keyed by output names
        /// </summary>
        public IReadOnlyDictionary<string, NodeMetadata> OutputMetadata
        {
            get
            {
                return _outputMetadata;
            }
        }

        /// <summary>
        /// Metadata regarding the overridable initializers, keyed by node names
        /// </summary>
        public IReadOnlyDictionary<string, NodeMetadata> OverridableInitializerMetadata
        {
            get
            {
                return _overridableInitializerMetadata;
            }
        }

        /// <summary>
        /// Runs the loaded model for the given inputs, and fetches all the outputs.
        /// </summary>
        /// <param name="inputs">specify a collection of <see cref="NamedOnnxValue"/> that indicates the input values.</param>
        /// <returns>Output Tensors in a Collection of NamedOnnxValue. User must dispose the output.</returns>
        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs)
        {
            string[] outputNames = new string[_outputMetadata.Count];
            _outputMetadata.Keys.CopyTo(outputNames, 0);
            return Run(inputs, outputNames);
        }

        /// <summary>
        /// Runs the loaded model for the given inputs, and fetches the outputs specified in <paramref name="outputNames"/>.
        /// </summary>
        /// <param name="inputs">Specify a collection of <see cref="NamedOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputNames">Specify a collection of string that indicates the output names to fetch.</param>
        /// <returns>Output Tensors in a Collection of NamedOnnxValue. User must dispose the output.</returns>
        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs, IReadOnlyCollection<string> outputNames)
        {
            return Run(inputs, outputNames, _builtInRunOptions);
        }

        /// <summary>
        /// Runs the loaded model for the given inputs, and fetches the specified outputs in <paramref name="outputNames"/>. Uses the given RunOptions for this run.
        /// </summary>
        /// <param name="inputs">Specify a collection of <see cref="NamedOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputNames">Specify a collection of string that indicates the output names to fetch.</param>
        /// <param name="options"></param>
        /// <returns>Output Tensors in a Collection of NamedOnnxValue. User must dispose the output.</returns>
        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs, IReadOnlyCollection<string> outputNames, RunOptions options)
        {
            // prepare inputs
            var inputNamesArray = new string[inputs.Count];
            var inputValuesArray = new IntPtr[inputs.Count];
            var pinnedInputBufferHandles = new System.Buffers.MemoryHandle[inputs.Count];
            var disposeInputs = new bool[inputs.Count];

            int inputIndex = 0;
            foreach (var input in inputs)
            {
                inputNamesArray[inputIndex] = input.Name;

                // create Tensor from the input if feasible, else throw notsupported exception for now
                input.ToNativeOnnxValue(
                    out inputValuesArray[inputIndex],
                    out pinnedInputBufferHandles[inputIndex],
                    out disposeInputs[inputIndex]);

                inputIndex++;
            }

            // prepare outputs
            string[] outputNamesArray = outputNames as string[] ?? outputNames.ToArray();
            IntPtr[] outputValuesArray = new IntPtr[outputNames.Count];

            IntPtr status = NativeMethods.OrtRun(
                                                _nativeHandle,
                                                options.Handle,
                                                inputNamesArray,
                                                inputValuesArray,
                                                (UIntPtr)inputs.Count,
                                                outputNamesArray,
                                                (UIntPtr)outputNames.Count,
                                                outputValuesArray /* Empty array is passed in to receive output OrtValue pointers */
                                                );

            try
            {
                NativeApiStatus.VerifySuccess(status);
                var result = new DisposableList<DisposableNamedOnnxValue>(outputValuesArray.Length);
                for (int i = 0; i < outputValuesArray.Length; i++)
                {
                    result.Add(DisposableNamedOnnxValue.CreateFromOnnxValue(outputNamesArray[i], outputValuesArray[i]));
                }

                return result;
            }
            catch (OnnxRuntimeException e)
            {
                //clean up the individual output tensors if it is not null;
                for (int i = 0; i < outputValuesArray.Length; i++)
                {
                    if (outputValuesArray[i] != IntPtr.Zero)
                    {
                        NativeMethods.OrtReleaseValue(outputValuesArray[i]);
                    }
                }
                throw e;
            }
            finally
            {
                for (int i = 0; i < inputs.Count; i++)
                {
                    if (disposeInputs[i])
                    {
                        NativeMethods.OrtReleaseValue(inputValuesArray[i]); // For elementary type Tensors, this should not release the buffer, but should delete the native tensor object.
                                                                            // For string tensors, this releases the native memory allocated for the tensor, including the buffer
                        pinnedInputBufferHandles[i].Dispose();
                    }
                }
            }

        }

        /// <summary>
        /// Runs the loaded model for the given inputs, and fetches all the outputs.
        /// </summary>
        /// <param name="inputNames">Specify a collection of string that indicates the input names. Should match <paramref name="inputValues"/>.</param>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values.</param>
        /// <returns>Output Tensors in a Collection of NamedOnnxValue. User must dispose the output.</returns>
        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(
            IReadOnlyCollection<string> inputNames,
            IReadOnlyCollection<FixedBufferOnnxValue> inputValues)
        {
            string[] outputNames = new string[_outputMetadata.Count];
            _outputMetadata.Keys.CopyTo(outputNames, 0);
            return Run(inputNames, inputValues, outputNames, _builtInRunOptions);
        }

        /// <summary>
        /// Runs the loaded model for the given inputs, and fetches the outputs specified in <paramref name="outputNames"/>.
        /// </summary>
        /// <param name="inputNames">Specify a collection of string that indicates the input names. Should match <paramref name="inputValues"/>.</param>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputNames">Specify a collection of string that indicates the output names to fetch.</param>
        /// <returns>Output Tensors in a Collection of NamedOnnxValue. User must dispose the output.</returns>
        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(
            IReadOnlyCollection<string> inputNames,
            IReadOnlyCollection<FixedBufferOnnxValue> inputValues,
            IReadOnlyCollection<string> outputNames)
        {
            return Run(inputNames, inputValues, outputNames, _builtInRunOptions);
        }

        /// <summary>
        /// Runs the loaded model for the given inputs, and fetches the specified outputs in <paramref name="outputNames"/>. Uses the given RunOptions for this run.
        /// </summary>
        /// <param name="inputNames">Specify a collection of string that indicates the input names. Should match <paramref name="inputValues"/>.</param>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputNames">Specify a collection of string that indicates the output names to fetch.</param>
        /// <param name="options"></param>
        /// <returns>Output Tensors in a Collection of NamedOnnxValue. User must dispose the output.</returns>
        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(
            IReadOnlyCollection<string> inputNames,
            IReadOnlyCollection<FixedBufferOnnxValue> inputValues,
            IReadOnlyCollection<string> outputNames,
            RunOptions options)
        {
            if (inputNames.Count != inputValues.Count)
            {
                throw new ArgumentException($"Length of {nameof(inputNames)} ({inputNames.Count}) must match that of {nameof(inputValues)} ({inputValues.Count}).");
            }

            // prepare inputs
            string[] inputNamesArray = inputNames as string[] ?? inputNames.ToArray();
            IntPtr[] inputValuesArray = new IntPtr[inputNames.Count];
            int inputIndex = 0;
            foreach (var input in inputValues)
            {
                inputValuesArray[inputIndex] = input.Value;

                inputIndex++;
            }

            // prepare outputs
            string[] outputNamesArray = outputNames as string[] ?? outputNames.ToArray();
            IntPtr[] outputValuesArray = new IntPtr[outputNames.Count];

            IntPtr status = NativeMethods.OrtRun(
                                                _nativeHandle,
                                                options.Handle,
                                                inputNamesArray,
                                                inputValuesArray,
                                                (UIntPtr)inputNames.Count,
                                                outputNamesArray,
                                                (UIntPtr)outputNames.Count,
                                                outputValuesArray /* Empty array is passed in to receive output OrtValue pointers */
                                                );

            try
            {
                NativeApiStatus.VerifySuccess(status);
                var result = new DisposableList<DisposableNamedOnnxValue>(outputValuesArray.Length);
                for (int i = 0; i < outputValuesArray.Length; i++)
                {
                    result.Add(DisposableNamedOnnxValue.CreateFromOnnxValue(outputNamesArray[i], outputValuesArray[i]));
                }

                return result;
            }
            catch (OnnxRuntimeException e)
            {
                //clean up the individual output tensors if it is not null;
                for (uint i = 0; i < outputValuesArray.Length; i++)
                {
                    if (outputValuesArray[i] != IntPtr.Zero)
                    {
                        NativeMethods.OrtReleaseValue(outputValuesArray[i]);
                    }
                }
                throw e;
            }

        }

        /// <summary>
        /// Runs the loaded model for the given inputs and outputs.
        /// 
        /// Outputs need to be created with correct type and dimension to accept the fetched data.
        /// </summary>
        /// <param name="inputNames">Specify a collection of string that indicates the input names. Should match <paramref name="inputValues"/>.</param>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputNames">Specify a collection of string that indicates the output names. Should match <paramref name="outputValues"/>.</param>
        /// <param name="outputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the output values.</param>
        public void Run(
            IReadOnlyCollection<string> inputNames,
            IReadOnlyCollection<FixedBufferOnnxValue> inputValues,
            IReadOnlyCollection<string> outputNames,
            IReadOnlyCollection<FixedBufferOnnxValue> outputValues)
        {
            Run(inputNames, inputValues, outputNames, outputValues, _builtInRunOptions);
        }

        /// <summary>
        /// Runs the loaded model for the given inputs and outputs. Uses the given RunOptions for this run.
        /// 
        /// Outputs need to be created with correct type and dimension to accept the fetched data.
        /// </summary>
        /// <param name="inputNames">Specify a collection of string that indicates the input names. Should match <paramref name="inputValues"/>.</param>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputNames">Specify a collection of string that indicates the output names. Should match <paramref name="outputValues"/>.</param>
        /// <param name="outputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the output values.</param>
        /// <param name="options"></param>
        public void Run(
            IReadOnlyCollection<string> inputNames,
            IReadOnlyCollection<FixedBufferOnnxValue> inputValues,
            IReadOnlyCollection<string> outputNames,
            IReadOnlyCollection<FixedBufferOnnxValue> outputValues,
            RunOptions options)
        {
            if (inputNames.Count != inputValues.Count)
            {
                throw new ArgumentException($"Length of {nameof(inputNames)} ({inputNames.Count}) must match that of {nameof(inputValues)} ({inputValues.Count}).");
            }
            if (outputNames.Count != outputValues.Count)
            {
                throw new ArgumentException($"Length of {nameof(outputNames)} ({outputNames.Count}) must match that of {nameof(outputValues)} ({outputValues.Count}).");
            }

            // prepare inputs
            string[] inputNamesArray = inputNames as string[] ?? inputNames.ToArray();
            IntPtr[] inputValuesArray = new IntPtr[inputNames.Count];
            int inputIndex = 0;
            foreach (var input in inputValues)
            {
                inputValuesArray[inputIndex] = input.Value;

                inputIndex++;
            }

            // prepare outputs
            string[] outputNamesArray = outputNames as string[] ?? outputNames.ToArray();
            IntPtr[] outputValuesArray = new IntPtr[outputNames.Count];
            int outputIndex = 0;
            foreach (var output in outputValues)
            {
                if (output.ElementType == Tensors.TensorElementType.String)
                {
                    throw new NotSupportedException("Using string type FixedBufferOnnxValue in outputs is not supported.");
                }

                outputValuesArray[outputIndex] = output.Value;

                outputIndex++;
            }

            IntPtr status = NativeMethods.OrtRun(
                                                _nativeHandle,
                                                options.Handle,
                                                inputNamesArray,
                                                inputValuesArray,
                                                (UIntPtr)inputNames.Count,
                                                outputNamesArray,
                                                (UIntPtr)outputNames.Count,
                                                outputValuesArray /* pointers to Pre-allocated OrtValue instances */
                                                );

            NativeApiStatus.VerifySuccess(status);
        }

        /// <summary>
        /// Runs the loaded model for the given inputs and outputs.
        /// 
        /// Outputs need to be created with correct type and dimension to receive the fetched data.
        /// </summary>
        /// <param name="inputs">Specify a collection of <see cref="NamedOnnxValue"/> that indicates the input values.</param>
        /// <param name="output">Specify a collection of <see cref="NamedOnnxValue"/> that indicates the output values.</param>
        public void Run(
            IReadOnlyCollection<NamedOnnxValue> inputs,
            IReadOnlyCollection<NamedOnnxValue> outputs)
        {
            Run(inputs, outputs, _builtInRunOptions);
        }

        /// <summary>
        /// Runs the loaded model for the given inputs and outputs. Uses the given RunOptions for this run.
        /// 
        /// Outputs need to be created with correct type and dimension to receive the fetched data.
        /// </summary>
        /// <param name="inputs">Specify a collection of <see cref="NamedOnnxValue"/> that indicates the input values.</param>
        /// <param name="output">Specify a collection of <see cref="NamedOnnxValue"/> that indicates the output values.</param>
        /// <param name="options"></param>
        public void Run(
            IReadOnlyCollection<NamedOnnxValue> inputs,
            IReadOnlyCollection<NamedOnnxValue> outputs,
            RunOptions options)
        {
            var inputNamesArray = new string[inputs.Count];
            var inputValuesArray = new IntPtr[inputs.Count];
            var pinnedInputBufferHandles = new System.Buffers.MemoryHandle[inputs.Count];
            var disposeInputs = new bool[inputs.Count];

            var outputNamesArray = new string[outputs.Count];
            var outputValuesArray = new IntPtr[outputs.Count];
            var pinnedOutputBufferHandles = new System.Buffers.MemoryHandle[outputs.Count];
            var disposeOutputs = new bool[outputs.Count];

            try
            {
                // prepare inputs
                int inputIndex = 0;
                foreach (var input in inputs)
                {
                    inputNamesArray[inputIndex] = input.Name;

                    // create native OrtValue from the input if feasible, else throw notsupported exception for now
                    input.ToNativeOnnxValue(
                        out inputValuesArray[inputIndex],
                        out pinnedInputBufferHandles[inputIndex],
                        out disposeInputs[inputIndex]);

                    inputIndex++;
                }

                // prepare outputs
                int outputIndex = 0;
                foreach (var output in outputs)
                {
                    outputNamesArray[outputIndex] = output.Name;

                    // create native OrtValue from the output if feasible, else throw notsupported exception for now
                    output.ToNativeOnnxValue(
                        out outputValuesArray[outputIndex],
                        out pinnedOutputBufferHandles[outputIndex],
                        out disposeOutputs[outputIndex]);

                    outputIndex++;
                }

                IntPtr status = NativeMethods.OrtRun(
                                                    _nativeHandle,
                                                    options.Handle,
                                                    inputNamesArray,
                                                    inputValuesArray,
                                                    (UIntPtr)inputs.Count,
                                                    outputNamesArray,
                                                    (UIntPtr)outputs.Count,
                                                    outputValuesArray /* pointers to Pre-allocated OrtValue instances */
                                                    );

                NativeApiStatus.VerifySuccess(status);
            }
            finally
            {
                for (int i = 0; i < inputs.Count; i++)
                {
                    if (disposeInputs[i])
                    {
                        NativeMethods.OrtReleaseValue(inputValuesArray[i]); // For elementary type Tensors, this should not release the buffer, but should delete the native tensor object.
                                                                            // For string tensors, this releases the native memory allocated for the tensor, including the buffer
                        pinnedInputBufferHandles[i].Dispose();
                    }
                }

                for (int i = 0; i < outputs.Count; i++)
                {
                    if (disposeOutputs[i])
                    {
                        NativeMethods.OrtReleaseValue(outputValuesArray[i]); // For elementary type Tensors, this should not release the buffer, but should delete the native tensor object.
                                                                             // For string tensors, this releases the native memory allocated for the tensor, including the buffer
                        pinnedOutputBufferHandles[i].Dispose();
                    }
                }
            }
        }

        /// <summary>
        /// Runs the loaded model for the given inputs and outputs.
        /// 
        /// Outputs need to be created with correct type and dimension to receive the fetched data.
        /// </summary>
        /// <param name="inputs">Specify a collection of <see cref="NamedOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputNames">Specify a collection of string that indicates the output names. Should match <paramref name="outputValues"/>.</param>
        /// <param name="outputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the output values.</param>
        public void Run(
            IReadOnlyCollection<NamedOnnxValue> inputs,
            IReadOnlyCollection<string> outputNames,
            IReadOnlyCollection<FixedBufferOnnxValue> outputValues)
        {
            Run(inputs, outputNames, outputValues, _builtInRunOptions);
        }

        /// <summary>
        /// Runs the loaded model for the given inputs and outputs. Uses the given RunOptions for this run.
        /// 
        /// Outputs need to be created with correct type and dimension to receive the fetched data.
        /// </summary>
        /// <param name="inputs">Specify a collection of <see cref="NamedOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputNames">Specify a collection of string that indicates the output names. Should match <paramref name="outputValues"/>.</param>
        /// <param name="outputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the output values.</param>
        /// <param name="options"></param>
        public void Run(
            IReadOnlyCollection<NamedOnnxValue> inputs,
            IReadOnlyCollection<string> outputNames,
            IReadOnlyCollection<FixedBufferOnnxValue> outputValues,
            RunOptions options)
        {
            if (outputNames.Count != outputValues.Count)
            {
                throw new ArgumentException($"Length of {nameof(outputNames)} ({outputNames.Count}) must match that of {nameof(outputValues)} ({outputValues.Count}).");
            }


            var inputNamesArray = new string[inputs.Count];
            var inputValuesArray = new IntPtr[inputs.Count];
            var pinnedInputBufferHandles = new System.Buffers.MemoryHandle[inputs.Count];
            var disposeInputs = new bool[inputs.Count];

            try
            {
                // prepare inputs
                int inputIndex = 0;
                foreach (var input in inputs)
                {
                    inputNamesArray[inputIndex] = input.Name;

                    // create native OrtValue from the input if feasible, else throw notsupported exception for now
                    input.ToNativeOnnxValue(
                        out inputValuesArray[inputIndex],
                        out pinnedInputBufferHandles[inputIndex],
                        out disposeInputs[inputIndex]);

                    inputIndex++;
                }

                // prepare outputs
                string[] outputNamesArray = outputNames as string[] ?? outputNames.ToArray();
                IntPtr[] outputValuesArray = new IntPtr[outputNames.Count];
                int outputIndex = 0;
                foreach (var output in outputValues)
                {
                    if (output.ElementType == TensorElementType.String)
                    {
                        throw new NotSupportedException("Using string type FixedBufferOnnxValue in outputs is not supported.");
                    }

                    outputValuesArray[outputIndex] = output.Value;

                    outputIndex++;
                }

                IntPtr status = NativeMethods.OrtRun(
                                                    _nativeHandle,
                                                    options.Handle,
                                                    inputNamesArray,
                                                    inputValuesArray,
                                                    (UIntPtr)inputs.Count,
                                                    outputNamesArray,
                                                    (UIntPtr)outputNames.Count,
                                                    outputValuesArray /* pointers to Pre-allocated OrtValue instances */
                                                    );


                NativeApiStatus.VerifySuccess(status);
            }
            finally
            {
                for (int i = 0; i < inputs.Count; i++)
                {
                    if (disposeInputs[i])
                    {
                        NativeMethods.OrtReleaseValue(inputValuesArray[i]); // For elementary type Tensors, this should not release the buffer, but should delete the native tensor object.
                                                                            // For string tensors, this releases the native memory allocated for the tensor, including the buffer
                        pinnedInputBufferHandles[i].Dispose();
                    }
                }
            }
        }

        /// <summary>
        /// Runs the loaded model for the given inputs and outputs.
        /// 
        /// Outputs need to be created with correct type and dimension to receive the fetched data.
        /// </summary>
        /// <param name="inputNames">Specify a collection of string that indicates the input names. Should match <paramref name="inputValues"/>.</param>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values.</param>
        /// <param name="output">Specify a collection of <see cref="NamedOnnxValue"/> that indicates the output values.</param>
        public void Run(
            IReadOnlyCollection<string> inputNames,
            IReadOnlyCollection<FixedBufferOnnxValue> inputValues,
            IReadOnlyCollection<NamedOnnxValue> outputs)
        {
            Run(inputNames, inputValues, outputs, _builtInRunOptions);
        }

        /// <summary>
        /// Runs the loaded model for the given inputs and outputs. Uses the given RunOptions for this run.
        /// 
        /// Outputs need to be created with correct type and dimension to receive the fetched data.
        /// </summary>
        /// <param name="inputNames">Specify a collection of string that indicates the input names. Should match <paramref name="inputValues"/>.</param>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values.</param>
        /// <param name="output">Specify a collection of <see cref="NamedOnnxValue"/> that indicates the output values.</param>
        /// <param name="options"></param>
        public void Run(
            IReadOnlyCollection<string> inputNames,
            IReadOnlyCollection<FixedBufferOnnxValue> inputValues,
            IReadOnlyCollection<NamedOnnxValue> outputs,
            RunOptions options)
        {
            if (inputNames.Count != inputValues.Count)
            {
                throw new ArgumentException($"Length of {nameof(inputNames)} ({inputNames.Count}) must match that of {nameof(inputValues)} ({inputValues.Count}).");
            }

            var outputNamesArray = new string[outputs.Count];
            var outputValuesArray = new IntPtr[outputs.Count];
            var pinnedOutputBufferHandles = new System.Buffers.MemoryHandle[outputs.Count];
            var disposeOutputs = new bool[outputs.Count];

            try
            {
                // prepare inputs
                string[] inputNamesArray = inputNames as string[] ?? inputNames.ToArray();
                IntPtr[] inputValuesArray = new IntPtr[inputNames.Count];
                int inputIndex = 0;
                foreach (var input in inputValues)
                {
                    inputValuesArray[inputIndex] = input.Value;

                    inputIndex++;
                }

                // prepare outputs

                int outputIndex = 0;
                foreach (var output in outputs)
                {
                    outputNamesArray[outputIndex] = output.Name;

                    // create native OrtValue from the output if feasible, else throw notsupported exception for now
                    output.ToNativeOnnxValue(
                        out outputValuesArray[outputIndex],
                        out pinnedOutputBufferHandles[outputIndex],
                        out disposeOutputs[outputIndex]);

                    outputIndex++;
                }

                IntPtr status = NativeMethods.OrtRun(
                                                    _nativeHandle,
                                                    options.Handle,
                                                    inputNamesArray,
                                                    inputValuesArray,
                                                    (UIntPtr)inputNames.Count,
                                                    outputNamesArray,
                                                    (UIntPtr)outputs.Count,
                                                    outputValuesArray /* pointers to Pre-allocated OrtValue instances */
                                                    );


                NativeApiStatus.VerifySuccess(status);
            }
            finally
            {
                for (int i = 0; i < outputs.Count; i++)
                {
                    if (disposeOutputs[i])
                    {
                        NativeMethods.OrtReleaseValue(outputValuesArray[i]); // For elementary type Tensors, this should not release the buffer, but should delete the native tensor object.
                                                                             // For string tensors, this releases the native memory allocated for the tensor, including the buffer
                        pinnedOutputBufferHandles[i].Dispose();
                    }
                }
            }
        }

        /// <summary>
        /// Create OrtIoBinding instance to bind pre-allocated buffers
        /// to input/output
        /// </summary>
        /// <returns></returns>
        public OrtIoBinding CreateIoBinding()
        {
            return new OrtIoBinding(this);
        }

        /// <summary>
        /// This method runs inference on the OrtIoBinding instance
        /// The method does not return anything. This is a lightweight version of 
        /// RunWithBindingAndNames(). When you bind pre-allocated buffers to the output values
        /// you may not want to fetch the outputs since you already have access to them so you can spare
        /// the expense of fetching them and pairing with names.
        /// You can still fetch the outputs by calling OrtIOBinding.GetOutputValues()
        /// </summary>
        /// <param name="runOptions"></param>
        /// <param name="ioBinding"></param>
        public void RunWithBinding(RunOptions runOptions, OrtIoBinding ioBinding)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtRunWithBinding(Handle, runOptions.Handle, ioBinding.Handle));
        }

        /// <summary>
        ///  This method return a collection of DisposableNamedOnnxValue as in other interfaces
        ///  Query names from OrtIoBinding object and pair then with the array of OrtValues returned
        /// from OrtIoBinding.GetOutputValues()
        /// 
        /// </summary>
        /// <param name="runOptions">RunOptions</param>
        /// <param name="ioBinding">OrtIoBinding instance with bindings</param>
        /// <param name="names">optional parameter. If you already know the names of the outputs you can save a native
        /// call to retrieve output names. They will be paired with the returned OrtValues and combined into DisposbleNamedOnnxValues.
        /// Otherwise, the method will retrieve output names from the OrtIoBinding instance.
        /// It is an error if you supply a different number of names than the returned outputs</param>
        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> RunWithBindingAndNames(RunOptions runOptions, OrtIoBinding ioBinding, string[] names = null)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtRunWithBinding(Handle, runOptions.Handle, ioBinding.Handle));
            using (var ortValues = ioBinding.GetOutputValues())
            {
                string[] outputNames = names;
                if (outputNames == null)
                {
                    outputNames = ioBinding.GetOutputNames();
                }

                if (outputNames.Length != ortValues.Count)
                {
                    throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                        "Number of specified names: " + names.Length + " does not match the output number: " +
                        ortValues.Count);
                }

                var result = new DisposableList<DisposableNamedOnnxValue>(outputNames.Length);
                try
                {
                    for (int i = 0; i < outputNames.Length; ++i)
                    {
                        var ortValue = ortValues.ElementAt(i);
                        result.Add(DisposableNamedOnnxValue.CreateTensorFromOnnxValue(outputNames[i], ortValue.Handle));
                        // We transferred ownership of the handle.
                        // Make sure it is not disposed here
                        ortValue.Disown();
                    }
                } catch(Exception e)
                {
                    result.Dispose();
                    throw e;
                }
                return result;
            }
        }

        /// <summary>
        /// Ends profiling for the session. Returns the profile file name.
        /// 
        public string EndProfiling()
        {
            IntPtr nameHandle = IntPtr.Zero;
            var allocator = OrtAllocator.DefaultInstance;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionEndProfiling(_nativeHandle,
                                                                   allocator.Pointer,
                                                                   out nameHandle));
            using(var allocation = new OrtMemoryAllocation(allocator, nameHandle, 0))
            {
                return NativeOnnxValueHelper.StringFromNativeUtf8(nameHandle);
            }
        }

        //TODO: kept internal until implemented
        internal ModelMetadata ModelMetadata
        {
            get
            {
                return new ModelMetadata(); //TODO: implement
            }
        }

        #endregion

        #region private methods

        private void Init(string modelPath, SessionOptions options)
        {
            var envHandle = OnnxRuntime.Handle;
            var session = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateSession(envHandle, NativeMethods.GetPlatformSerializedString(modelPath), options.Handle, out session));

            InitWithSessionHandle(session, options);
        }

        private void Init(byte[] modelData, SessionOptions options)
        {
            var envHandle = OnnxRuntime.Handle;
            var session = IntPtr.Zero;

            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateSessionFromArray(envHandle, modelData, (UIntPtr)modelData.Length, options.Handle, out session));

            InitWithSessionHandle(session, options);
        }

        /// <summary>
        /// Initializes the session object with a native session handle
        /// </summary>
        /// <param name="session">Handle of a native session object</param>
        /// <param name="options">Session options</param>
        private void InitWithSessionHandle(IntPtr session, SessionOptions options)
        {
            _nativeHandle = session;
            try
            {

                // Initialize input/output metadata
                _inputMetadata = new Dictionary<string, NodeMetadata>();
                _outputMetadata = new Dictionary<string, NodeMetadata>();
                _overridableInitializerMetadata = new Dictionary<string, NodeMetadata>();

                // get input count
                UIntPtr inputCount = UIntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetInputCount(_nativeHandle, out inputCount));

                // get all the input names and metadata
                for (ulong i = 0; i < (ulong)inputCount; i++)
                {
                    var iname = GetInputName(i);
                    _inputMetadata[iname] = GetInputMetadata(i);
                }
                // get output count
                UIntPtr outputCount = UIntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetOutputCount(_nativeHandle, out outputCount));

                // get all the output names and metadata
                for (ulong i = 0; i < (ulong)outputCount; i++)
                {
                    _outputMetadata[GetOutputName(i)] = GetOutputMetadata(i);
                }

                // get overridable initializer count
                UIntPtr initilaizerCount = UIntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetOverridableInitializerCount(_nativeHandle, out initilaizerCount));

                // get all the overridable initializer names and metadata
                for (ulong i = 0; i < (ulong)initilaizerCount; i++)
                {
                    _overridableInitializerMetadata[GetOverridableInitializerName(i)] = GetOverridableInitializerMetadata(i);
                }

            }
            catch (OnnxRuntimeException e)
            {
                if (_nativeHandle != IntPtr.Zero)
                {
                    NativeMethods.OrtReleaseSession(_nativeHandle);
                    _nativeHandle = IntPtr.Zero;
                }
                throw e;
            }

            _builtInRunOptions = new RunOptions();  // create a default built-in run option, and avoid creating a new one every run() call
        }


        private string GetOutputName(ulong index)
        {
            IntPtr nameHandle = IntPtr.Zero;
            string str = null;

            try
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetOutputName(
                                               _nativeHandle,
                                               (UIntPtr)index,
                                               OrtAllocator.DefaultInstance.Pointer,
                                               out nameHandle));

                str = NativeOnnxValueHelper.StringFromNativeUtf8(nameHandle);
            }
            finally
            {
                if (nameHandle != IntPtr.Zero)
                {
                    OrtAllocator.DefaultInstance.FreeMemory(nameHandle);
                }
            }

            return str;
        }

        private string GetInputName(ulong index)
        {
            IntPtr nameHandle = IntPtr.Zero;
            string str = null;

            try
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetInputName(
                                               _nativeHandle,
                                               (UIntPtr)index,
                                               OrtAllocator.DefaultInstance.Pointer,
                                               out nameHandle));

                str = NativeOnnxValueHelper.StringFromNativeUtf8(nameHandle);
            }
            finally
            {
                if (nameHandle != IntPtr.Zero)
                {
                    OrtAllocator.DefaultInstance.FreeMemory(nameHandle);
                }
            }
            return str;
        }

        private string GetOverridableInitializerName(ulong index)
        {
            IntPtr nameHandle = IntPtr.Zero;
            string str = null;

            try
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetOverridableInitializerName(
                                                _nativeHandle,
                                                (UIntPtr)index,
                                                OrtAllocator.DefaultInstance.Pointer,
                                                out nameHandle));

                str = NativeOnnxValueHelper.StringFromNativeUtf8(nameHandle);
            }
            finally
            {
                if (nameHandle != IntPtr.Zero)
                {
                    OrtAllocator.DefaultInstance.FreeMemory(nameHandle);
                }
            }
            return str;
        }

        private NodeMetadata GetInputMetadata(ulong index)
        {
            IntPtr typeInfo = IntPtr.Zero;
            try
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetInputTypeInfo(_nativeHandle, (UIntPtr)index, out typeInfo));
                return GetMetadataFromTypeInfo(typeInfo);
            }
            finally
            {
                if (typeInfo != IntPtr.Zero)
                {
                    NativeMethods.OrtReleaseTypeInfo(typeInfo);
                }
            }
        }

        private NodeMetadata GetOutputMetadata(ulong index)
        {
            IntPtr typeInfo = IntPtr.Zero;
            try
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetOutputTypeInfo(_nativeHandle, (UIntPtr)index, out typeInfo));
                return GetMetadataFromTypeInfo(typeInfo);
            }
            finally
            {
                if (typeInfo != IntPtr.Zero)
                {
                    NativeMethods.OrtReleaseTypeInfo(typeInfo);
                }
            }
        }

        private NodeMetadata GetOverridableInitializerMetadata(ulong index)
        {
            IntPtr typeInfo = IntPtr.Zero;
            try
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetOverridableInitializerTypeInfo(_nativeHandle, (UIntPtr)index, out typeInfo));
                return GetMetadataFromTypeInfo(typeInfo);
            }
            finally
            {
                if (typeInfo != IntPtr.Zero)
                {
                    NativeMethods.OrtReleaseTypeInfo(typeInfo);
                }
            }
        }

        internal static NodeMetadata GetMetadataFromTypeInfo(IntPtr typeInfo)
        {
            OnnxValueType valueType;
            unsafe
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetOnnxTypeFromTypeInfo(typeInfo, new IntPtr(&valueType)));
            }
            if (valueType != OnnxValueType.ONNX_TYPE_TENSOR && valueType != OnnxValueType.ONNX_TYPE_SPARSETENSOR)
            {
                return new NodeMetadata(valueType, new int[] { }, new string[] { }, typeof(NamedOnnxValue));
            }

            // This should not be released
            IntPtr tensorInfo;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCastTypeInfoToTensorInfo(typeInfo, out tensorInfo)); //(IntPtr)(int)(uint)
            // Convert the newly introduced OrtTypeInfo* to the older OrtTypeAndShapeInfo*

            if (tensorInfo == IntPtr.Zero)
                return null;

            TensorElementType type;
            {
                IntPtr el_type;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorElementType(tensorInfo, out el_type));
                type = (TensorElementType)el_type;
            }
            Type dotnetType = null;
            int width = 0;
            TensorElementTypeConverter.GetTypeAndWidth(type, out dotnetType, out width);
            UIntPtr numDimensions;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetDimensionsCount(tensorInfo, out numDimensions));

            long[] dimensions = new long[(int)numDimensions];
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetDimensions(tensorInfo, dimensions, numDimensions));
            int[] intDimensions = new int[(int)numDimensions];
            for (var i = 0; i < (long)numDimensions; i++)
            {
                intDimensions[i] = (int)dimensions[i];
            }

            IntPtr[] dimensionNamePtrs = new IntPtr[(int)numDimensions];
            NativeApiStatus.VerifySuccess(
                NativeMethods.OrtGetSymbolicDimensions(tensorInfo, dimensionNamePtrs, numDimensions));

            string[] symbolicDimensions = new string[(int)numDimensions];
            for (var i = 0; i < (int)numDimensions; i++)
            {
                symbolicDimensions[i] = Marshal.PtrToStringAnsi(dimensionNamePtrs[i]); //assumes charset = ANSI
            }

            return new NodeMetadata(valueType, intDimensions, symbolicDimensions, dotnetType);
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

        #region destructors disposers


        ~InferenceSession()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            GC.SuppressFinalize(this);
            Dispose(true);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                // cleanup managed resources
                if (_builtInSessionOptions != null)
                {
                    _builtInSessionOptions.Dispose();
                }

                if (_builtInRunOptions != null)
                {
                    _builtInRunOptions.Dispose();
                }
            }

            // cleanup unmanaged resources
            if (_nativeHandle != IntPtr.Zero)
            {
                NativeMethods.OrtReleaseSession(_nativeHandle);
            }
        }

        #endregion

    }


    /// <summary>
    /// Resembles type and shape information of session-graph nodes, used for communicating the shape/type of input/output nodes
    /// </summary>
    public class NodeMetadata
    {
        private OnnxValueType _onnxValueType;
        private int[] _dimensions;
        private string[] _symbolicDimensions;
        private Type _type;

        internal NodeMetadata(OnnxValueType onnxValueType, int[] dimensions, string[] symbolicDimensions, Type type)
        {
            _onnxValueType = onnxValueType;
            _dimensions = dimensions;
            _symbolicDimensions = symbolicDimensions;
            _type = type;
        }

        public OnnxValueType OnnxValueType
        {
            get
            {
                return _onnxValueType;
            }
        }

        public int[] Dimensions
        {
            get
            {
                return _dimensions;
            }
        }

        public string[] SymbolicDimensions
        {
            get
            {
                return _symbolicDimensions;
            }
        }

        public System.Type ElementType
        {
            get
            {
                return _type;
            }
        }

        public bool IsTensor
        {
            get
            {
                return true; // currently only Tensor nodes are supported
            }
        }
    }


    internal class ModelMetadata
    {
        //TODO: placeholder for Model metadata. Currently C-API does not expose this.
    }


}
