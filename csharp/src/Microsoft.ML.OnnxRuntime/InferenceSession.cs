// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.IO;
using System.Linq;


namespace Microsoft.ML.OnnxRuntime
{


    /// <summary>
    /// Represents an Inference Session on an ONNX Model
    /// </summary>
    public class InferenceSession: IDisposable
    {
        protected IntPtr _nativeHandle;
        protected Dictionary<string, NodeMetadata> _inputMetadata, _outputMetadata;


        #region Public API

        /// <summary>
        /// Constructs an InferenceSession from a model file
        /// </summary>
        /// <param name="modelPath"></param>
        public InferenceSession(string modelPath)
            : this(modelPath, SessionOptions.Default)
        {
        }

        /// <summary>
        /// Constructs an InferenceSession from a model file, with some additional session options
        /// </summary>
        /// <param name="modelPath"></param>
        /// <param name="options"></param>
        public InferenceSession(string modelPath, SessionOptions options)
        {
            var envHandle = OnnxRuntime.Handle;

            _nativeHandle = IntPtr.Zero;
            try
            {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateSession(envHandle, System.Text.Encoding.Unicode.GetBytes(modelPath), options._nativePtr, out _nativeHandle));
                else
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateSession(envHandle, System.Text.Encoding.UTF8.GetBytes(modelPath), options._nativePtr, out _nativeHandle));

                // Initialize input/output metadata
                _inputMetadata = new Dictionary<string, NodeMetadata>();
                _outputMetadata = new Dictionary<string, NodeMetadata>();

                // get input count
                ulong inputCount = 0;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetInputCount(_nativeHandle, out inputCount));

                // get all the output names
                for (ulong i = 0; i < inputCount; i++)
                {
                    _inputMetadata[GetInputName(i)] = GetInputMetadata(i);
                }

                // get output count
                ulong outputCount = 0;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetOutputCount(_nativeHandle, out outputCount));

                // get all the output names
                for (ulong i = 0; i < outputCount; i++)
                {
                    _outputMetadata[GetOutputName(i)] = GetOutputMetadata(i);
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
        }

        public IReadOnlyDictionary<string, NodeMetadata> InputMetadata
        {
            get
            {
                return _inputMetadata;  
            }
        }

        public IReadOnlyDictionary<string, NodeMetadata> OutputMetadata
        {
            get
            {
                return _outputMetadata; 
            }
        }

        /// <summary>
        /// Runs the loaded model for the given inputs, and fetches all the outputs.
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns>Output Tensors in a Collection of NamedOnnxValue</returns>
        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs)
        {
            string[] outputNames = new string[_outputMetadata.Count];
            _outputMetadata.Keys.CopyTo(outputNames, 0);
            return Run(inputs, outputNames);
        }

        /// <summary>
        /// Runs the loaded model for the given inputs, and fetches the outputs specified in <paramref name="outputNames"/>.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="outputNames"></param>
        /// <returns>Output Tensors in a Collection of NamedOnnxValue</returns>
        public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs, IReadOnlyCollection<string> outputNames)
        {
            return Run(inputs, outputNames, RunOptions.Default);
        }

        /// <summary>
        /// Runs the loaded model for the given inputs, and fetches the specified outputs in <paramref name="outputNames"/>.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="outputNames"></param>
        /// <param name="options"></param>
        /// <returns>Output Tensors in a Collection of NamedOnnxValue</returns>
        //TODO: kept internal until RunOptions is made public
        internal IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs, IReadOnlyCollection<string> outputNames, RunOptions options)
        {
            var inputNames = new string[inputs.Count];
            var inputTensors = new IntPtr[inputs.Count];
            var pinnedBufferHandles = new System.Buffers.MemoryHandle[inputs.Count];

            int offset = 0;
            foreach (var input in inputs)
            {
                inputNames[offset] = input.Name;

                // create Tensor from the input if feasible, else throw notsupported exception for now
                input.ToNativeOnnxValue(out inputTensors[offset], out pinnedBufferHandles[offset]);

                offset++;
            }

            string[] outputNamesArray = outputNames.ToArray();
            IntPtr[] outputValueArray = new IntPtr[outputNames.Count];

            IntPtr status = NativeMethods.OrtRun(
                                                this._nativeHandle,
                                                IntPtr.Zero,  // TODO: use Run options when Run options creation API is available
                                                              // Passing null uses the default run options in the C-api
                                                inputNames,
                                                inputTensors,
                                                (ulong)(inputTensors.Length),  /* TODO: size_t, make it portable for x86 arm */
                                                outputNamesArray,
                                                (ulong)outputNames.Count,  /* TODO: size_t, make it portable for x86 and arm */
                                                outputValueArray /* An array of output value pointers. Array must be allocated by the caller */
                                                );

            try
            {
                NativeApiStatus.VerifySuccess(status);
                var result = new DisposableList<DisposableNamedOnnxValue>();
                for (uint i = 0; i < outputValueArray.Length; i++)
                {
                    result.Add(DisposableNamedOnnxValue.CreateFromOnnxValue(outputNamesArray[i], outputValueArray[i]));
                }

                return result;
            }
            catch (OnnxRuntimeException e)
            {
                //clean up the individual output tensors if it is not null;
                for (uint i = 0; i < outputValueArray.Length; i++)
                {
                    if (outputValueArray[i] != IntPtr.Zero)
                    {
                        NativeMethods.OrtReleaseValue(outputValueArray[i]);
                    }
                }
                throw e;
            }
            finally
            {
                // always unpin the input buffers, and delete the native Onnx value objects
                for (int i = 0; i < inputs.Count; i++)
                {
                    NativeMethods.OrtReleaseValue(inputTensors[i]); // this should not release the buffer, but should delete the native tensor object
                    pinnedBufferHandles[i].Dispose();
                }
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
        private string GetOutputName(ulong index)
        {
            IntPtr nameHandle = IntPtr.Zero;
            string str = null;

            IntPtr  status = NativeMethods.OrtSessionGetOutputName(
                                                _nativeHandle,
                                                index,  
                                                NativeMemoryAllocator.DefaultInstance.Handle,
                                                out nameHandle);
            try
            {
                NativeApiStatus.VerifySuccess(status);
                str = Marshal.PtrToStringAnsi(nameHandle); //assumes charset = ANSI
            }
            finally 
            {
                if (nameHandle != IntPtr.Zero)
                {
                    NativeMemoryAllocator.DefaultInstance.FreeMemory(nameHandle);
                }
            }

            return str;
        }

        private string GetInputName(ulong index)
        {
            IntPtr nameHandle = IntPtr.Zero;
            string str = null;

            IntPtr status = NativeMethods.OrtSessionGetInputName(
                                                _nativeHandle,
                                                index,
                                                NativeMemoryAllocator.DefaultInstance.Handle,
                                                out nameHandle);
            try
            {
                NativeApiStatus.VerifySuccess(status);
                str = Marshal.PtrToStringAnsi(nameHandle); //assumes charset = ANSI
            }
            finally
            {
                if (nameHandle != IntPtr.Zero)
                {
                    NativeMemoryAllocator.DefaultInstance.FreeMemory(nameHandle);
                }
            }

            return str;
        }


        private NodeMetadata GetInputMetadata(ulong index)
        {
            IntPtr typeInfo = IntPtr.Zero;
            try
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetInputTypeInfo(_nativeHandle, index, out typeInfo));
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
                NativeApiStatus.VerifySuccess(NativeMethods.OrtSessionGetOutputTypeInfo(_nativeHandle, index, out typeInfo));
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
            IntPtr tensorInfo = NativeMethods.OrtCastTypeInfoToTensorInfo(typeInfo);
            // Convert the newly introduced OrtTypeInfo* to the older OrtTypeAndShapeInfo*

            if (tensorInfo == IntPtr.Zero)
                return null;

            TensorElementType type = NativeMethods.OrtGetTensorElementType(tensorInfo);
            Type dotnetType = null;
            int width = 0;
            TensorElementTypeConverter.GetTypeAndWidth(type, out dotnetType, out width); 
            ulong numDimensions = NativeMethods.OrtGetNumOfDimensions(tensorInfo);
            long[] dimensions = new long[(int)numDimensions];
            NativeMethods.OrtGetDimensions(tensorInfo, dimensions, numDimensions);
            int[] intDimensions = new int[(int)numDimensions];
            for (ulong i = 0; i < numDimensions; i++)
            {
                intDimensions[i] = (int)dimensions[i];
            }
            return new NodeMetadata(intDimensions, dotnetType);
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
        private int[] _dimensions;
        private Type _type;

        internal NodeMetadata(int[] dimensions, Type type)
        {
            _dimensions = dimensions;
            _type = type;
        }

        public int[] Dimensions
        {
            get
            {
                return _dimensions;
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

    /// Sets various runtime options. 
    /// TODO: currently uses Default options only. kept internal until fully implemented
    internal class RunOptions
    {
        protected static readonly Lazy<RunOptions> _default = new Lazy<RunOptions>(() => new RunOptions());

        public static RunOptions Default
        {
            get
            {
                return _default.Value;
            }
        }

        private void RuntOptions()
        {

        }
    }

}
