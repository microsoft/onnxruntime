// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Buffers;
using System.Diagnostics;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Represents an Inference Session on an ONNX Model.
    /// This is a IDisposable class and it must be disposed of
    /// using either a explicit call to Dispose() method or
    /// a pattern of using() block. If this is a member of another
    /// class that class must also become IDisposable and it must
    /// dispose of InferfenceSession in its Dispose() method.
    /// </summary>
    public class InferenceSession : IDisposable
    {
        /// <summary>
        /// A pointer to a underlying native instance of OrtSession
        /// </summary>
        private IntPtr _nativeHandle;

        /// <summary>
        /// Dictionary that represents input metadata
        /// </summary>
        private Dictionary<string, NodeMetadata> _inputMetadata;

        /// <summary>
        /// Ordered list of input names
        /// </summary>
        private List<string> _inputNames;

        /// <summary>
        /// Dictionary that represent output metadata
        /// </summary>
        private Dictionary<string, NodeMetadata> _outputMetadata;

        /// <summary>
        /// Ordered list of output names
        /// </summary>
        private List<string> _outputNames;

        /// <summary>
        /// Dictionary that represents overridableInitializers metadata
        /// </summary>
        private Dictionary<string, NodeMetadata> _overridableInitializerMetadata;

        private SessionOptions _builtInSessionOptions = null;
        private RunOptions _builtInRunOptions = null;
        private ModelMetadata _modelMetadata = null;
        private bool _disposed = false;
        private ulong _profilingStartTimeNs = 0;

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
        /// Constructs an InferenceSession from a model file and it will use 
        /// the provided pre-packed weights container to store and share pre-packed buffers 
        /// of shared initializers across sessions if any.
        /// </summary>
        /// <param name="modelPath">Model path</param>
        /// <param name="prepackedWeightsContainer">Instance of PrepackedWeightsContainer. 
        /// Lifetime of 'prepackedWeightsContainer' must be
        /// managed by the user and it must outlive any sessions reliant on it</param>
        public InferenceSession(string modelPath, PrePackedWeightsContainer prepackedWeightsContainer)
        {
            _builtInSessionOptions = new SessionOptions(); // need to be disposed
            Init(modelPath, _builtInSessionOptions, prepackedWeightsContainer);
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
        /// Constructs an InferenceSession from a model file, with some additional session options
        /// and it will use the provided pre-packed weights container to store and share pre-packed buffers 
        /// of shared initializers across sessions if any.
        /// </summary>
        /// <param name="modelPath">Model path</param>
        /// <param name="options">Session options</param>
        /// <param name="prepackedWeightsContainer">Instance of PrepackedWeightsContainer. 
        /// Lifetime of 'prepackedWeightsContainer' must be
        /// managed by the user and it must outlive any sessions reliant on it</param>
        public InferenceSession(string modelPath, SessionOptions options,
            PrePackedWeightsContainer prepackedWeightsContainer)
        {
            Init(modelPath, options, prepackedWeightsContainer);
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
        /// Constructs an InferenceSession from a model data (in byte array) and it will use 
        /// the provided pre-packed weights container to store and share pre-packed buffers 
        /// of shared initializers across sessions if any.
        /// </summary>
        /// <param name="model">Model as byte array</param>
        /// <param name="prepackedWeightsContainer">Instance of PrepackedWeightsContainer. 
        /// Lifetime of 'prepackedWeightsContainer' must be
        /// managed by the user and it must outlive any sessions reliant on it</param>
        public InferenceSession(byte[] model, PrePackedWeightsContainer prepackedWeightsContainer)
        {
            _builtInSessionOptions = new SessionOptions(); // need to be disposed
            Init(model, _builtInSessionOptions, prepackedWeightsContainer);
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
        /// Constructs an InferenceSession from a model data (in byte array) with some additional
        /// session options and it will use the provided pre-packed weights container to store
        /// and share pre-packed buffers of shared initializers across sessions if any.
        /// </summary>
        /// <param name="model">Model as byte array</param>
        /// <param name="options">Session Options</param>
        /// <param name="prepackedWeightsContainer">Instance of PrepackedWeightsContainer. 
        /// Lifetime of 'prepackedWeightsContainer' must be
        /// managed by the user and it must outlive any sessions reliant on it</param>
        public InferenceSession(byte[] model, SessionOptions options,
                                PrePackedWeightsContainer prepackedWeightsContainer)
        {
            Init(model, options, prepackedWeightsContainer);
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
        /// Ordered list of input names that can be accessed by index;
        /// </summary>
        public IReadOnlyList<string> InputNames { get { return _inputNames; } }

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
        /// Ordered list of output names that can be accessed by index.
        /// </summary>
        public IReadOnlyList<string> OutputNames { get { return _outputNames; } }

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
        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs,
            IReadOnlyCollection<string> outputNames)
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
        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs,
            IReadOnlyCollection<string> outputNames,
            RunOptions options)
        {
            using (var cleanupList = new DisposableList<IDisposable>())
            {
                var inputNamesArray = ConvertNamesToUtf8(inputs, v => v.Name, LookupInputMetadata, cleanupList);
                var inputValuesArray = GetOrtValuesHandles(inputs, LookupInputMetadata, ExtractOrtValueForInput, cleanupList);
                var outputNamesArray = ConvertNamesToUtf8(outputNames, n => n, LookupOutputMetadata, cleanupList);

                var ortValues = RunImpl(options, inputNamesArray, inputValuesArray, outputNamesArray, cleanupList);
                return CreateDisposableResult(ortValues, outputNames);
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
            return Run(inputNames, inputValues, _outputNames, _builtInRunOptions);
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

            using (var cleanupList = new DisposableList<IDisposable>())
            {
                var inputNamesArray = ConvertNamesToUtf8(inputNames, n => n, LookupInputMetadata, cleanupList);
                IntPtr[] inputValuesArray = GetOrtValuesHandles(inputValues, true);
                var outputNamesArray = ConvertNamesToUtf8(outputNames, n => n, LookupOutputMetadata, cleanupList);


                var ortValues = RunImpl(options, inputNamesArray, inputValuesArray, outputNamesArray, cleanupList);
                return CreateDisposableResult(ortValues, outputNames);
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

            using (var cleanupList = new DisposableList<IDisposable>())
            {
                // prepare inputs
                var inputNamesArray = ConvertNamesToUtf8(inputNames, n => n, LookupInputMetadata, cleanupList);
                IntPtr[] inputValuesArray = GetOrtValuesHandles(inputValues, true);

                // prepare outputs
                var outputNamesArray = ConvertNamesToUtf8(outputNames, n => n, LookupOutputMetadata, cleanupList);
                IntPtr[] outputValuesArray = GetOrtValuesHandles(outputValues, false);

                NativeApiStatus.VerifySuccess(NativeMethods.OrtRun(
                                                    _nativeHandle,
                                                    options.Handle,
                                                    inputNamesArray,
                                                    inputValuesArray,
                                                    (UIntPtr)inputNames.Count,
                                                    outputNamesArray,
                                                    (UIntPtr)outputNames.Count,
                                                    outputValuesArray /* pointers to Pre-allocated OrtValue instances */
                                                    ));
            }
        }

        /// <summary>
        /// Runs the loaded model for the given inputs and outputs.
        /// 
        /// Outputs need to be created with correct type and dimension to receive the fetched data.
        /// </summary>
        /// <param name="inputs">Specify a collection of <see cref="NamedOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputs">Specify a collection of <see cref="NamedOnnxValue"/> that indicates the output values.</param>
        public void Run(
            IReadOnlyCollection<NamedOnnxValue> inputs,
            IReadOnlyCollection<NamedOnnxValue> outputs)
        {
            Run(inputs, outputs, _builtInRunOptions);
        }

        /// <summary>
        /// 
        /// Runs the loaded model for the given inputs and outputs. Uses the given RunOptions for this run.
        ///
        /// Outputs need to be created with correct type and dimension to receive the fetched data.
        /// </summary>
        /// <param name="inputs">Specify a collection of <see cref="NamedOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputs">Specify a collection of <see cref="NamedOnnxValue"/> that indicates the output values.</param>
        /// <param name="options"></param>
        public void Run(
            IReadOnlyCollection<NamedOnnxValue> inputs,
            IReadOnlyCollection<NamedOnnxValue> outputs,
            RunOptions options)
        {
            using (var cleanupList = new DisposableList<IDisposable>())
            {
                var inputNamesArray = ConvertNamesToUtf8(inputs, i => i.Name, LookupInputMetadata, cleanupList);
                var inputValuesArray = GetOrtValuesHandles(inputs, LookupInputMetadata, ExtractOrtValueForInput, cleanupList);

                var outputNamesArray = ConvertNamesToUtf8(outputs, o => o.Name, LookupOutputMetadata, cleanupList);
                var outputValuesArray = GetOrtValuesHandles(outputs, LookupOutputMetadata, ExtractOrtValueForOutput, cleanupList);

                NativeApiStatus.VerifySuccess(NativeMethods.OrtRun(
                                                    _nativeHandle,
                                                    options.Handle,
                                                    inputNamesArray,
                                                    inputValuesArray,
                                                    (UIntPtr)inputs.Count,
                                                    outputNamesArray,
                                                    (UIntPtr)outputs.Count,
                                                    outputValuesArray /* pointers to Pre-allocated OrtValue instances */
                                                    ));
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

            using (var cleanupList = new DisposableList<IDisposable>())
            {
                // prepare inputs
                var inputNamesArray = ConvertNamesToUtf8(inputs, i => i.Name, LookupInputMetadata, cleanupList);
                var inputValuesArray = GetOrtValuesHandles(inputs, LookupInputMetadata, ExtractOrtValueForInput, cleanupList);

                // prepare outputs
                var outputNamesArray = ConvertNamesToUtf8(outputNames, n => n, LookupOutputMetadata, cleanupList);
                var outputValuesArray = GetOrtValuesHandles(outputValues, false);

                NativeApiStatus.VerifySuccess(NativeMethods.OrtRun(
                                                    _nativeHandle,
                                                    options.Handle,
                                                    inputNamesArray,
                                                    inputValuesArray,
                                                    (UIntPtr)inputs.Count,
                                                    outputNamesArray,
                                                    (UIntPtr)outputNames.Count,
                                                    outputValuesArray /* pointers to Pre-allocated OrtValue instances */
                                                    ));
            }
        }

        /// <summary>
        ///
        /// Runs the loaded model for the given inputs and outputs.
        ///
        /// Outputs need to be created with correct type and dimension to receive the fetched data.
        /// </summary>
        /// <param name="inputNames">Specify a collection of string that indicates the input names. Should match <paramref name="inputValues"/>.</param>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputs">Specify a collection of <see cref="NamedOnnxValue"/> that indicates the output values.</param>
        public void Run(
            IReadOnlyCollection<string> inputNames,
            IReadOnlyCollection<FixedBufferOnnxValue> inputValues,
            IReadOnlyCollection<NamedOnnxValue> outputs)
        {
            Run(inputNames, inputValues, outputs, _builtInRunOptions);
        }

        /// <summary>
        /// 
        /// Runs the loaded model for the given inputs and outputs. Uses the given RunOptions for this run.
        ///
        /// Outputs need to be created with correct type and dimension to receive the fetched data.
        /// </summary>
        /// <param name="inputNames">Specify a collection of string that indicates the input names. Should match <paramref name="inputValues"/>.</param>
        /// <param name="inputValues">Specify a collection of <see cref="FixedBufferOnnxValue"/> that indicates the input values.</param>
        /// <param name="outputs">Specify a collection of <see cref="NamedOnnxValue"/> that indicates the output values.</param>
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

            using (var cleanupList = new DisposableList<IDisposable>())
            {
                // prepare inputs
                var inputNamesArray = ConvertNamesToUtf8(inputNames, n => n, LookupInputMetadata, cleanupList);
                var inputValuesArray = GetOrtValuesHandles(inputValues, true);

                // prepare outputs
                var outputNamesArray = ConvertNamesToUtf8(outputs, o => o.Name, LookupOutputMetadata, cleanupList);
                var outputValuesArray = GetOrtValuesHandles(outputs, LookupOutputMetadata, ExtractOrtValueForOutput, cleanupList);

                NativeApiStatus.VerifySuccess(NativeMethods.OrtRun(
                                                    _nativeHandle,
                                                    options.Handle,
                                                    inputNamesArray,
                                                    inputValuesArray,
                                                    (UIntPtr)inputNames.Count,
                                                    outputNamesArray,
                                                    (UIntPtr)outputs.Count,
                                                    outputValuesArray /* pointers to Pre-allocated OrtValue instances */
                                                    ));
            }
        }

        /// <summary>
        /// Create OrtIoBinding instance to bind pre-allocated buffers
        /// to input/output
        /// </summary>
        /// <returns>A new instance of OrtIoBinding</returns>
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
        /// <param name="runOptions">runOptions</param>
        /// <param name="ioBinding">ioBinding instance to use</param>
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
        /// <returns>A disposable collection of DisposableNamedOnnxValue that encapsulate output OrtValues</returns>
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
                        result.Add(DisposableNamedOnnxValue.CreateFromOrtValue(outputNames[i], ortValue));
                    }
                }
                catch (Exception)
                {
                    result.Dispose();
                    throw;
                }
                return result;
            }
        }

        /// <summary>
        /// Ends profiling for the session.
        /// </summary>
        /// <returns> Returns the profile file name.</returns>
        public string EndProfiling()
        {
            IntPtr nameHandle = IntPtr.Zero;
            var allocator = OrtAllocator.DefaultInstance;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionEndProfiling(_nativeHandle,
                                                                   allocator.Pointer,
                                                                   out nameHandle));
            using (var allocation = new OrtMemoryAllocation(allocator, nameHandle, 0))
            {
                return NativeOnnxValueHelper.StringFromNativeUtf8(nameHandle);
            }
        }

        // Delegate for string extraction from an arbitrary input/output object
        private delegate string NameExtractor<in TInput>(TInput input);

        // delegate to fetch input/output OrtValue
        private delegate OrtValue OrtValueExtractor(NamedOnnxValue value, NodeMetadata metadata, out IDisposable memOwner);

        // Delegate to lookup metadata for input/initializers/output
        private delegate NodeMetadata MetadataLookup(string nodeName);

        /// <summary>
        /// Checks if the name is a known input or overridable initializer name
        /// and if so, returns metadata for it.
        /// metadata
        /// </summary>
        /// <param name="nodeName"></param>
        /// <returns>NodeMetadata for the nodeName</returns>
        /// <exception cref="OnnxRuntimeException"></exception>
        private NodeMetadata LookupInputMetadata(string nodeName)
        {
            NodeMetadata meta;
            if (!_inputMetadata.TryGetValue(nodeName, out meta) &&
                !_overridableInitializerMetadata.TryGetValue(nodeName, out meta))
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument, $"Input name: '{nodeName}' is not in the metadata");
            }
            return meta;
        }

        /// <summary>
        /// Checks if the nodeName is a known output name and if so returns metadata for it.
        /// </summary>
        /// <param name="nodeName"></param>
        /// <returns></returns>
        /// <exception cref="OnnxRuntimeException"></exception>
        private NodeMetadata LookupOutputMetadata(string nodeName)
        {
            NodeMetadata meta;
            if (!_outputMetadata.TryGetValue(nodeName, out meta))
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument, $"Output name: '{nodeName}' is not in the metadata");
            }
            return meta;
        }

        /// <summary>
        /// Fetches/creates OrtValue for the content of the input
        /// </summary>
        /// <param name="input"></param>
        /// <param name="metadata"></param>
        /// <param name="memOwner"></param>
        /// <returns></returns>
        private static OrtValue ExtractOrtValueForInput(NamedOnnxValue input, NodeMetadata metadata, out IDisposable memOwner)
        {
            return input.InputToOrtValue(metadata, out memOwner);
        }

        /// <summary>
        /// Fetches/Creates OrtValue for output
        /// </summary>
        /// <param name="output"></param>
        /// <param name="metadata"></param>
        /// <param name="memOwner"></param>
        /// <returns>May return null if the onnx value type does not support pre-creation of output OrtValues</returns>
        private static OrtValue ExtractOrtValueForOutput(NamedOnnxValue output, NodeMetadata metadata, out IDisposable memOwner)
        {
            return output.OutputToOrtValue(metadata, out memOwner);
        }

        /// <summary>
        /// Run helper
        /// </summary>
        /// <param name="values">names to convert to zero terminated utf8 and pin</param>
        /// <param name="nameExtractor">extractor functor that helps extracting names from inputs</param>
        /// <param name="metaDict">inputs/outputs metadata</param>
        /// <param name="cleanupList">list to add pinned memory to for later disposal</param>
        /// <returns></returns>
        private IntPtr[] ConvertNamesToUtf8<T>(IReadOnlyCollection<T> values, NameExtractor<T> nameExtractor,
            MetadataLookup metaLookup,
            DisposableList<IDisposable> cleanupList)
        {
            cleanupList.Capacity += values.Count;
            var result = new IntPtr[values.Count];
            for (int i = 0; i < values.Count; ++i)
            {
                var name = nameExtractor(values.ElementAt(i));
                NodeMetadata meta = metaLookup(name);
                var utf8Name = meta.ZeroTerminatedUtf8Name;
                Debug.Assert(utf8Name != null);
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
        /// This function obtains ortValues for NamedOnnxValue.
        /// The problem with NamedOnnxValue is that it is not disposable and can not contain any disposable items.
        /// so calling InputToOrtValue creates a new instance of OrtValue that needs to be disposed.
        /// The deriving object DisposableNamedValue actually contains and owns OrtValue and it returns
        /// it.
        /// </summary>
        /// <param name="values">a collection of NamedOnnxValues</param>
        /// <param name="metaLookup">Metadata lookup function (input/initializers/output)</param>
        /// <param name="cleanupList">list to cleanup in an exception safe manner</param>
        /// <returns></returns>
        private IntPtr[] GetOrtValuesHandles(IReadOnlyCollection<NamedOnnxValue> values, MetadataLookup metaLookup,
            OrtValueExtractor ortValueExtractor,
            DisposableList<IDisposable> cleanupList)
        {
            cleanupList.Capacity += values.Count * 2;
            IntPtr[] result = new IntPtr[values.Count];
            for (int valueIndex = 0; valueIndex < values.Count; ++valueIndex)
            {
                var value = values.ElementAt(valueIndex);
                var meta = metaLookup(value.Name);
                var ortValue = ortValueExtractor(value, meta, out IDisposable memHolder);
                if (memHolder != null)
                {
                    cleanupList.Add(memHolder);
                }
                if (ortValue != null)
                {
                    if (ortValue.IsOwned)
                        cleanupList.Add(ortValue);

                    result[valueIndex] = ortValue.Handle;
                }
                else
                {
                    result[valueIndex] = IntPtr.Zero;
                }
            }
            return result;
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


        private DisposableList<OrtValue> RunImpl(RunOptions options, IntPtr[] inputNames, IntPtr[] inputValues, IntPtr[] outputNames,
           DisposableList<IDisposable> cleanupList)
        {
            cleanupList.Capacity += 1;
            var ortValues = new DisposableList<OrtValue>(outputNames.Length + 1);
            cleanupList.Add(ortValues);

            IntPtr[] outputValuesArray = new IntPtr[outputNames.Length];
            NativeApiStatus.VerifySuccess(NativeMethods.OrtRun(
                                                _nativeHandle,
                                                options.Handle,
                                                inputNames,
                                                inputValues,
                                                (UIntPtr)inputNames.Length,
                                                outputNames,
                                                (UIntPtr)outputNames.Length,
                                                outputValuesArray /* Empty array is passed in to receive output OrtValue pointers */
                                                ));

            foreach (var v in outputValuesArray)
            {
                ortValues.Add(new OrtValue(v));
            }
            return ortValues;
        }

        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> CreateDisposableResult(List<OrtValue> ortValues,
            IReadOnlyCollection<string> outputNames)
        {
            var result = new DisposableList<DisposableNamedOnnxValue>(outputNames.Count);
            try
            {
                for (int i = 0; i < ortValues.Count; i++)
                {
                    result.Add(DisposableNamedOnnxValue.CreateFromOrtValue(outputNames.ElementAt(i), ortValues[i]));
                }
            }
            catch (OnnxRuntimeException)
            {
                result.Dispose();
                throw;
            }
            return result;
        }

        /// <summary>
        /// This property queries model metadata, constructs
        /// an instance of ModelMetadata and caches it
        /// </summary>
        /// <returns>Instance of ModelMetdata</returns>
        public ModelMetadata ModelMetadata
        {
            get
            {
                if (_modelMetadata != null)
                {
                    return _modelMetadata;
                }

                _modelMetadata = new ModelMetadata(this);
                return _modelMetadata;
            }
        }

        /// <summary>
        /// Return the nanoseconds of profiling's start time
        /// On some platforms, this timer may not be as precise as nanoseconds
        /// For instance, on Windows and MacOS, the precision will be ~100ns
        /// </summary>
        public ulong ProfilingStartTimeNs
        {
            get
            {
                return _profilingStartTimeNs;
            }
        }

        #endregion

        #region private methods

        private void Init(string modelPath, SessionOptions options,
                          PrePackedWeightsContainer prepackedWeightsContainer = null)
        {
            var envHandle = OrtEnv.Instance().Handle;
            IntPtr session;
            if (prepackedWeightsContainer == null)
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateSession(envHandle, NativeOnnxValueHelper.GetPlatformSerializedString(modelPath),
                options.Handle, out session));
            }

            else
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateSessionWithPrepackedWeightsContainer(
                    envHandle, NativeOnnxValueHelper.GetPlatformSerializedString(modelPath),
                    options.Handle, prepackedWeightsContainer.Pointer, out session));
            }

            InitWithSessionHandle(session, options);
        }

        private void Init(byte[] modelData, SessionOptions options,
                          PrePackedWeightsContainer prepackedWeightsContainer = null)
        {
            var envHandle = OrtEnv.Instance().Handle;
            IntPtr session;
            if (prepackedWeightsContainer == null)
            {

                NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateSessionFromArray(envHandle, modelData, (UIntPtr)modelData.Length, options.Handle, out session));
            }

            else
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateSessionFromArrayWithPrepackedWeightsContainer(
                    envHandle, modelData, (UIntPtr)modelData.Length, options.Handle, prepackedWeightsContainer.Pointer,
                    out session));

            }

            InitWithSessionHandle(session, options);
        }

        /// <summary>
        /// Initializes the session object with a native session handle
        /// </summary>
        /// <param name="session">Value of a native session object</param>
        /// <param name="options">Session options</param>
        private void InitWithSessionHandle(IntPtr session, SessionOptions options)
        {
            _nativeHandle = session;
            try
            {
                // Initialize input/output metadata

                // get input count
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetInputCount(_nativeHandle, out UIntPtr inputCount));

                // get all the input names and metadata
                _inputMetadata = new Dictionary<string, NodeMetadata>((int)inputCount);
                _inputNames = new List<string>((int)inputCount);

                for (ulong i = 0; i < (ulong)inputCount; i++)
                {
                    var inputMeta = GetInputMetadata(i);
                    var iname = GetInputName(i, out byte[] utf8);
                    _inputNames.Add(iname);
                    inputMeta.ZeroTerminatedUtf8Name = utf8;
                    _inputMetadata[iname] = inputMeta;
                }
                // get output count
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetOutputCount(_nativeHandle, out UIntPtr outputCount));

                // get all the output names and metadata
                _outputMetadata = new Dictionary<string, NodeMetadata>((int)outputCount);
                _outputNames = new List<string>((int)outputCount);

                for (ulong i = 0; i < (ulong)outputCount; i++)
                {
                    var outputMeta = GetOutputMetadata(i);
                    var oname = GetOutputName(i, out byte[] utf8);
                    _outputNames.Add(oname);
                    outputMeta.ZeroTerminatedUtf8Name = utf8;
                    _outputMetadata[oname] = outputMeta;
                }

                // get overridable initializer count
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetOverridableInitializerCount(_nativeHandle, out UIntPtr initilaizerCount));

                _overridableInitializerMetadata = new Dictionary<string, NodeMetadata>((int)initilaizerCount);
                // get all the overridable initializer names and metadata
                for (ulong i = 0; i < (ulong)initilaizerCount; i++)
                {
                    var meta = GetOverridableInitializerMetadata(i);
                    var iname = GetOverridableInitializerName(i, out byte[] utf8);
                    meta.ZeroTerminatedUtf8Name = utf8;
                    _overridableInitializerMetadata[iname] = meta;
                }
                // set profiling's start time
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetProfilingStartTimeNs(_nativeHandle,
                                                                    out UIntPtr startTime));
                _profilingStartTimeNs = (ulong)startTime;
            }
            catch (Exception)
            {
                if (_nativeHandle != IntPtr.Zero)
                {
                    NativeMethods.OrtReleaseSession(_nativeHandle);
                    _nativeHandle = IntPtr.Zero;
                }
                throw;
            }

            _builtInRunOptions = new RunOptions();  // create a default built-in run option, and avoid creating a new one every run() call  
        }


        private string GetOutputName(ulong index, out byte[] utf8)
        {
            string str;
            var allocator = OrtAllocator.DefaultInstance;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetOutputName(
                                           _nativeHandle,
                                           (UIntPtr)index,
                                           allocator.Pointer,
                                           out IntPtr nameHandle));

            using (var ortAllocation = new OrtMemoryAllocation(allocator, nameHandle, 0))
            {
                NativeOnnxValueHelper.StringAndUtf8FromNative(nameHandle, out str, out utf8);
            }

            return str;
        }

        private string GetInputName(ulong index, out byte[] utf8)
        {
            string str;
            var allocator = OrtAllocator.DefaultInstance;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetInputName(
                                           _nativeHandle,
                                           (UIntPtr)index,
                                           allocator.Pointer,
                                           out IntPtr nameHandle));

            using (var ortAllocation = new OrtMemoryAllocation(allocator, nameHandle, 0))
            {
                NativeOnnxValueHelper.StringAndUtf8FromNative(nameHandle, out str, out utf8);
            }
            return str;
        }

        private string GetOverridableInitializerName(ulong index, out byte[] utf8)
        {
            string str;
            var allocator = OrtAllocator.DefaultInstance;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetOverridableInitializerName(
                                            _nativeHandle,
                                            (UIntPtr)index,
                                            allocator.Pointer,
                                            out IntPtr nameHandle));
            using (var ortAllocation = new OrtMemoryAllocation(allocator, nameHandle, 0))
            {
                NativeOnnxValueHelper.StringAndUtf8FromNative(nameHandle, out str, out utf8);
            }
            return str;
        }

        private NodeMetadata GetInputMetadata(ulong index)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetInputTypeInfo(_nativeHandle, (UIntPtr)index, out IntPtr typeInfo));
            try
            {
                return GetMetadataFromTypeInfo(typeInfo);
            }
            finally
            {
                NativeMethods.OrtReleaseTypeInfo(typeInfo);
            }
        }

        private NodeMetadata GetOutputMetadata(ulong index)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetOutputTypeInfo(_nativeHandle, (UIntPtr)index, out IntPtr typeInfo));
            try
            {
                return GetMetadataFromTypeInfo(typeInfo);
            }
            finally
            {
                NativeMethods.OrtReleaseTypeInfo(typeInfo);
            }
        }

        private NodeMetadata GetOverridableInitializerMetadata(ulong index)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetOverridableInitializerTypeInfo(_nativeHandle, (UIntPtr)index, out IntPtr typeInfo));
            try
            {
                return GetMetadataFromTypeInfo(typeInfo);
            }
            finally
            {
                NativeMethods.OrtReleaseTypeInfo(typeInfo);
            }
        }

        internal static NodeMetadata GetMetadataFromTypeInfo(IntPtr typeInfo)
        {
            OnnxValueType valueType;
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetOnnxTypeFromTypeInfo(typeInfo, out IntPtr valType));
                valueType = (OnnxValueType)valType;
            }

            switch (valueType)
            {
                case OnnxValueType.ONNX_TYPE_TENSOR:
                case OnnxValueType.ONNX_TYPE_SPARSETENSOR:
                    return GetTensorNodeMetadata(valueType, typeInfo);
                case OnnxValueType.ONNX_TYPE_SEQUENCE:
                    return GetSequenceMetadataFromTypeInfo(typeInfo);
                case OnnxValueType.ONNX_TYPE_MAP:
                    return GetMapMetadataFromTypeInfo(typeInfo);
                case OnnxValueType.ONNX_TYPE_OPTIONAL:
                    return GetOptionalMetadataFromTypeInfo(typeInfo);
            }

            throw new OnnxRuntimeException(ErrorCode.NotImplemented, $"Value type: '{valueType}' not supported in this code");
        }

        internal static NodeMetadata GetSequenceMetadataFromTypeInfo(IntPtr typeInfo)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCastTypeInfoToSequenceTypeInfo(typeInfo, out IntPtr sequenceTypeInfo));
            // Casts API are broken. Always return success, but may return null for the result.
            if (sequenceTypeInfo == IntPtr.Zero)
            {
                throw new OnnxRuntimeException(ErrorCode.Fail, "TypeInfo cast to SequenceTypeInfo failed. The object does not represent a sequence");
            }

            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetSequenceElementType(sequenceTypeInfo, out IntPtr elementType));
            try
            {
                var elementMeta = GetMetadataFromTypeInfo(elementType);
                var seqMeta = new SequenceMetadata(elementMeta);
                return new NodeMetadata(seqMeta);
            }
            finally
            {
                NativeMethods.OrtReleaseTypeInfo(elementType);
            }
        }

        internal static NodeMetadata GetMapMetadataFromTypeInfo(IntPtr typeInfo)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCastTypeInfoToMapTypeInfo(typeInfo, out IntPtr mapTypeInfo));
            // Casts API are broken. Always return success, but may return null for the result.
            if (mapTypeInfo == IntPtr.Zero)
            {
                throw new OnnxRuntimeException(ErrorCode.Fail, "TypeInfo cast to MapTypeInfo failed. The object does not represent a map");
            }

            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetMapKeyType(mapTypeInfo, out IntPtr keyType));

            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetMapValueType(mapTypeInfo, out IntPtr valueTypeInfo));
            try
            {
                var valueMetadata = GetMetadataFromTypeInfo(valueTypeInfo);
                var mapMeta = new MapMetadata((TensorElementType)keyType, valueMetadata);
                return new NodeMetadata(mapMeta);
            }
            finally
            {
                NativeMethods.OrtReleaseTypeInfo(valueTypeInfo);
            }
        }

        internal static NodeMetadata GetOptionalMetadataFromTypeInfo(IntPtr typeInfo)
        {
            // This should not be destroyed
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCastTypeInfoToOptionalTypeInfo(typeInfo, out IntPtr optTypeInfo));
            // Casts API are broken. Always return success, but may return null for the result.
            if (optTypeInfo == IntPtr.Zero)
            {
                throw new OnnxRuntimeException(ErrorCode.Fail, "TypeInfo cast to OptionalTypeInfo failed. The object does not represent a optional value");
            }

            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetOptionalContainedTypeInfo(optTypeInfo, out IntPtr elementTypeInfo));
            try
            {
                var elementMetadata = GetMetadataFromTypeInfo(elementTypeInfo);
                var optMetadata = new OptionalMetadata(elementMetadata);
                return new NodeMetadata(optMetadata);
            }
            finally
            {
                NativeMethods.OrtReleaseTypeInfo(elementTypeInfo);
            }
        }

        internal static NodeMetadata GetTensorNodeMetadata(OnnxValueType valueType, IntPtr typeInfo)
        {
            // Fetch tensor type and shape from the TypeInfo
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCastTypeInfoToTensorInfo(typeInfo, out IntPtr tensorInfo)); //(IntPtr)(int)(uint)
            // Casts API are broken. Always return success, but may return null for the result.
            if (tensorInfo == IntPtr.Zero)
            {
                throw new OnnxRuntimeException(ErrorCode.Fail, "TypeInfo cast to TensorTypeInfo failed. The object does not represent a tensor");
            }

            TensorElementType type;
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorElementType(tensorInfo, out IntPtr el_type));
                type = (TensorElementType)el_type;
            }

            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetDimensionsCount(tensorInfo, out UIntPtr numDimensions));

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
                symbolicDimensions[i] = NativeOnnxValueHelper.StringFromNativeUtf8(dimensionNamePtrs[i]);
            }

            var tensorTypeAndShape = new TensorTypeAndShape(type, intDimensions, symbolicDimensions);
            return new NodeMetadata(valueType, tensorTypeAndShape);
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
        ~InferenceSession()
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
                NativeMethods.OrtReleaseSession(_nativeHandle);
                _nativeHandle = IntPtr.Zero;
            }
            _disposed = true;
        }

        #endregion
    }


    /// <summary>
    /// Represents tensor element type and its shapes
    /// </summary>
    public class TensorTypeAndShape
    {
        internal TensorTypeAndShape(TensorElementType elementType, int[] dimensions, string[] symbolicDimensions)
        {
            ElementTypeInfo = TensorBase.GetElementTypeInfo(elementType);
            if (ElementTypeInfo == null)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument, "Unregistered TensorElementType value of: " + elementType.ToString());
            }
            ElementDataType = elementType;
            Dimensions = dimensions;
            SymbolicDimensions = symbolicDimensions;
        }

        /// <summary>
        /// Tensor Element type
        /// </summary>
        /// <value>TensorElementType enum</value>
        public TensorElementType ElementDataType { get; }

        /// <summary>
        /// Shape
        /// </summary>
        /// <value>Array of dimensions</value>
        public int[] Dimensions { get; }

        /// <summary>
        /// Symbolic dimensions
        /// </summary>
        /// <value>Array of symbolic dimensions if present.</value>
        public string[] SymbolicDimensions { get; }

        /// <summary>
        /// Tensor element metadata
        /// </summary>
        public TensorElementTypeInfo ElementTypeInfo { get; }
    }

    /// <summary>
    /// Represents sequnce metdata
    /// </summary>
    public class SequenceMetadata
    {
        /// <summary>
        /// __ctor
        /// </summary>
        /// <param name="elementData"></param>
        internal SequenceMetadata(NodeMetadata elementData)
        {
            ElementMeta = elementData;
        }
        /// <summary>
        /// Element Metatada, recursive definition with a Tensor being a base case
        /// may contain maps, tensors and other sequences
        /// </summary>
        public NodeMetadata ElementMeta { get; }
    }

    /// <summary>
    /// The class contains metadata for an optional input/output
    /// </summary>
    public class OptionalMetadata
    {
        /// <summary>
        /// __ctor
        /// </summary>
        /// <param name="elementData"></param>
        internal OptionalMetadata(NodeMetadata elementData)
        {
            ElementMeta = elementData;
        }

        /// <summary>
        /// Element Metatada, recursive definition with a Tensor being a base case
        /// may contain maps, tensors and sequences
        /// </summary>
        public NodeMetadata ElementMeta { get; }
    }

    /// <summary>
    /// Represents Map MetaData.
    /// Key is always a tensor denoted by an element type
    /// with value type being a recursive structure that may
    /// contain other maps, sequences or tensors.
    /// </summary>
    public class MapMetadata
    {
        internal MapMetadata(TensorElementType keyDataType, NodeMetadata valueMetadata)
        {
            KeyDataType = keyDataType;
            ValueMetadata = valueMetadata;
        }

        /// <summary>
        /// Key tensor data type
        /// </summary>
        /// <value>A value of TensorElementType enum</value>
        public TensorElementType KeyDataType { get; }

        /// <summary>
        /// Value metadata
        /// </summary>
        /// /// <value>Instance of Nodemetadata for the value of the map</value>
        public NodeMetadata ValueMetadata { get; }
    }

    /// <summary>
    /// Resembles type and shape information of session-graph nodes, used for communicating the shape/type of input/output nodes
    /// </summary>
    public class NodeMetadata
    {
        private readonly Object _metadata;
        /// <summary>
        /// Constructs NodeMetadata for tensor
        /// </summary>
        /// <param name="onnxValueType">either ONNX_TYPE_TENSOR or ONNX_TYPE_SPARSETENSOR</param>
        /// <param name="typeAndShape">Tensor type and shape information</param>
        internal NodeMetadata(OnnxValueType onnxValueType, TensorTypeAndShape typeAndShape)
        {
            OnnxValueType = onnxValueType;
            CheckTensor();
            _metadata = typeAndShape;
        }

        /// <summary>
        /// __ctor for map metadata
        /// </summary>
        /// <param name="mapMetadata"></param>
        internal NodeMetadata(MapMetadata mapMetadata)
        {
            OnnxValueType = OnnxValueType.ONNX_TYPE_MAP;
            _metadata = mapMetadata;
        }

        /// <summary>
        /// __ctor for sequence metadata
        /// </summary>
        /// <param name="sequenceMetadata"></param>
        internal NodeMetadata(SequenceMetadata sequenceMetadata)
        {
            OnnxValueType = OnnxValueType.ONNX_TYPE_SEQUENCE;
            _metadata = sequenceMetadata;
        }

        /// <summary>
        /// __ctor
        /// </summary>
        /// <param name="optMetadata"></param>
        internal NodeMetadata(OptionalMetadata optMetadata)
        {
            OnnxValueType = OnnxValueType.ONNX_TYPE_OPTIONAL;
            _metadata = optMetadata;
        }

        private void CheckTensor()
        {
            if (!IsTensor)
            {
                throw new OnnxRuntimeException(ErrorCode.Fail, "OnnxValueType must either be a tensor or sparse tensor");
            }
        }

        /// <summary>
        /// Retrieves MapMetadata, valid only if this node represents a Map.
        /// </summary>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException">when the instance does not contain map metadata</exception>
        public MapMetadata AsMapMetadata()
        {
            if (OnnxValueType != OnnxValueType.ONNX_TYPE_MAP)
            {
                throw new OnnxRuntimeException(ErrorCode.Fail, "Instance does not contain Map metadata");
            }
            return _metadata as MapMetadata;
        }

        /// <summary>
        /// Retrieves SequenceMetadata, valid only if this node represents a Sequence
        /// </summary>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException">when the instance does not contain sequence metadata</exception>
        public SequenceMetadata AsSequenceMetadata()
        {
            if (OnnxValueType != OnnxValueType.ONNX_TYPE_SEQUENCE)
            {
                throw new OnnxRuntimeException(ErrorCode.Fail, "Instance does not contain Sequence metadata");
            }
            return _metadata as SequenceMetadata;
        }

        /// <summary>
        /// Retrieves Optional type metadata, valid if this node is optional
        /// Optional metadata is nothing more than just a container for all the usual
        /// element types.
        /// </summary>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
        public OptionalMetadata AsOptionalMetadata()
        {
            if (OnnxValueType != OnnxValueType.ONNX_TYPE_OPTIONAL)
            {
                throw new OnnxRuntimeException(ErrorCode.Fail, "Instance does not contain Optional metadata");
            }
            return _metadata as OptionalMetadata;
        }

        /// <summary>
        /// Type value of the node
        /// </summary>
        /// <value>A value of OnnxValueType enum</value>
        public OnnxValueType OnnxValueType { get; }

        /// <summary>
        /// Zero terminated UTF-8 name of the input/output
        /// Present only on the top-level instance
        /// metadata dictionary entries.
        /// 
        /// Used to avoid utf8 conversions on every run and associated allocations
        /// </summary>
        internal byte[] ZeroTerminatedUtf8Name { get; set; }

        /// <summary>
        /// Tensor shape valid only if this is a Tensor.
        /// Preserved for API compatibility
        /// </summary>
        /// <value>Array of dimensions</value>
        public int[] Dimensions
        {
            get
            {
                CheckTensor();
                return (_metadata as TensorTypeAndShape).Dimensions;
            }
        }

        /// <summary>
        /// Symbolic dimensions valid only if this is a Tensor.
        /// Preserved for API compatibility
        /// </summary>
        /// <value>Array of symbolic dimensions if present.</value>
        public string[] SymbolicDimensions
        {
            get
            {
                CheckTensor();
                return (_metadata as TensorTypeAndShape).SymbolicDimensions;
            }
        }

        /// <summary>
        /// .NET type that corresponds to the primitive Tensor data type.
        /// Valid only if this is a Tensor.
        /// </summary>
        /// <value>System.Type</value>
        public System.Type ElementType
        {
            get
            {
                CheckTensor();
                return (_metadata as TensorTypeAndShape).ElementTypeInfo.TensorType;
            }
        }

        /// <summary>
        /// Tensor Element Type. Valid if tensor
        /// </summary>
        public TensorElementType ElementDataType
        {
            get
            {
                CheckTensor();
                return (_metadata as TensorTypeAndShape).ElementDataType;
            }
        }

        /// <summary>
        /// Convinience method to check for string
        /// </summary>
        public bool IsString
        {
            get
            {
                CheckTensor();
                return (_metadata as TensorTypeAndShape).ElementTypeInfo.IsString;
            }
        }

        /// <summary>
        /// Whether it is a Tensor
        /// </summary>
        /// <value>currently always returns true</value>
        public bool IsTensor
        {
            get
            {
                return (OnnxValueType == OnnxValueType.ONNX_TYPE_TENSOR) || (OnnxValueType == OnnxValueType.ONNX_TYPE_SPARSETENSOR);
            }
        }
    }


    /// <summary>
    /// A class that queries and caches model metadata and exposes
    /// it as properties
    /// </summary>
    public class ModelMetadata
    {
        private string _producerName;
        private string _graphName;
        private string _domain;
        private string _description;
        private string _graphDescription;
        private long _version;
        private Dictionary<string, string> _customMetadataMap = new Dictionary<string, string>();

        internal ModelMetadata(InferenceSession session)
        {
            IntPtr modelMetadataHandle = IntPtr.Zero;

            var allocator = OrtAllocator.DefaultInstance;

            // Get the native ModelMetadata instance associated with the InferenceSession

            NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetModelMetadata(session.Handle, out modelMetadataHandle));

            try
            {

                // Process producer name
                IntPtr producerNameHandle = IntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtModelMetadataGetProducerName(modelMetadataHandle, allocator.Pointer, out producerNameHandle));
                using (var ortAllocation = new OrtMemoryAllocation(allocator, producerNameHandle, 0))
                {
                    _producerName = NativeOnnxValueHelper.StringFromNativeUtf8(producerNameHandle);
                }

                // Process graph name
                IntPtr graphNameHandle = IntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtModelMetadataGetGraphName(modelMetadataHandle, allocator.Pointer, out graphNameHandle));
                using (var ortAllocation = new OrtMemoryAllocation(allocator, graphNameHandle, 0))
                {
                    _graphName = NativeOnnxValueHelper.StringFromNativeUtf8(graphNameHandle);
                }


                // Process domain
                IntPtr domainHandle = IntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtModelMetadataGetDomain(modelMetadataHandle, allocator.Pointer, out domainHandle));
                using (var ortAllocation = new OrtMemoryAllocation(allocator, domainHandle, 0))
                {
                    _domain = NativeOnnxValueHelper.StringFromNativeUtf8(domainHandle);
                }

                // Process description
                IntPtr descriptionHandle = IntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtModelMetadataGetDescription(modelMetadataHandle, allocator.Pointer, out descriptionHandle));
                using (var ortAllocation = new OrtMemoryAllocation(allocator, descriptionHandle, 0))
                {
                    _description = NativeOnnxValueHelper.StringFromNativeUtf8(descriptionHandle);
                }

                // Process graph description
                IntPtr graphDescriptionHandle = IntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtModelMetadataGetGraphDescription(modelMetadataHandle, allocator.Pointer, out graphDescriptionHandle));
                using (var ortAllocation = new OrtMemoryAllocation(allocator, graphDescriptionHandle, 0))
                {
                    _graphDescription = NativeOnnxValueHelper.StringFromNativeUtf8(graphDescriptionHandle);
                }

                // Process version
                NativeApiStatus.VerifySuccess(NativeMethods.OrtModelMetadataGetVersion(modelMetadataHandle, out _version));


                // Process CustomMetadata Map
                IntPtr customMetadataMapKeysHandle = IntPtr.Zero;
                long numKeys;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtModelMetadataGetCustomMetadataMapKeys(modelMetadataHandle, allocator.Pointer, out customMetadataMapKeysHandle, out numKeys));

                // We have received an array of null terminated C strings which are the keys that we can use to lookup the custom metadata map
                // The OrtAllocator will finally free the customMetadataMapKeysHandle
                using (var ortAllocationKeysArray = new OrtMemoryAllocation(allocator, customMetadataMapKeysHandle, 0))
                using (var ortAllocationKeys = new DisposableList<OrtMemoryAllocation>((int)numKeys))
                {
                    // Put all the handles to each key in the DisposableList to be disposed off in an exception-safe manner
                    for (int i = 0; i < (int)numKeys; ++i)
                    {
                        ortAllocationKeys.Add(new OrtMemoryAllocation(allocator, Marshal.ReadIntPtr(customMetadataMapKeysHandle, IntPtr.Size * i), 0));
                    }

                    // Process each key via the stored key handles
                    foreach (var allocation in ortAllocationKeys)
                    {
                        IntPtr keyHandle = allocation.Pointer;
                        IntPtr valueHandle = IntPtr.Zero;
                        NativeApiStatus.VerifySuccess(NativeMethods.OrtModelMetadataLookupCustomMetadataMap(modelMetadataHandle, allocator.Pointer, keyHandle, out valueHandle));

                        using (var ortAllocationValue = new OrtMemoryAllocation(allocator, valueHandle, 0))
                        {
                            var key = NativeOnnxValueHelper.StringFromNativeUtf8(keyHandle);
                            var value = NativeOnnxValueHelper.StringFromNativeUtf8(valueHandle);

                            // Put the key/value pair into the dictionary
                            _customMetadataMap[key] = value;

                        }
                    }
                }
            }

            finally
            {

                // Free ModelMetadata handle
                NativeMethods.OrtReleaseModelMetadata(modelMetadataHandle);

            }

        }

        /// <summary>
        /// Producer name string
        /// </summary>
        /// <value>producer name string</value>
        public string ProducerName
        {
            get
            {
                return _producerName;
            }
        }

        /// <summary>
        /// Graph name for this model
        /// </summary>
        /// <value>graph name string</value>
        public string GraphName
        {
            get
            {
                return _graphName;
            }
        }

        /// <summary>
        /// Domain for this model
        /// </summary>
        /// <value>domain name string</value>
        public string Domain
        {
            get
            {
                return _domain;
            }
        }

        /// <summary>
        /// Unstructured model description
        /// </summary>
        /// <value>description string</value>
        public string Description
        {
            get
            {
                return _description;
            }
        }

        /// <summary>
        /// Unstructured graph description
        /// </summary>
        /// <value>description string</value>
        public string GraphDescription
        {
            get
            {
                return _graphDescription;
            }
        }

        /// <summary>
        /// Version number
        /// </summary>
        /// <value>long version integer</value>
        public long Version
        {
            get
            {
                return _version;
            }
        }

        /// <summary>
        /// Custom metadata key/value pairs
        /// </summary>
        /// <value>An instance of a Dictionary<string,string></value>
        public Dictionary<string, string> CustomMetadataMap
        {
            get
            {
                return _customMetadataMap;
            }
        }
    }


}
