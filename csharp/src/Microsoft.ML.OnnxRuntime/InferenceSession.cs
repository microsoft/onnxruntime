// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.IO;
using System.Linq;


namespace Microsoft.ML.OnnxRuntime
{

    public struct RunOptions
    {
        // placeholder for RunOptions
    }

    /// <summary>
    /// Represents an Inference Session against an ONNX Model
    /// </summary>
    public class InferenceSession: IDisposable
    {
        protected IntPtr _nativeHandle;
        protected Dictionary<string, NodeMetadata> _inputMetadata, _outputMetadata;


        internal InferenceSession(IntPtr nativeHandle)
        {
            _nativeHandle = nativeHandle;
        }

        #region Public API
        public InferenceSession(string modelPath)
            : this(modelPath, SessionOptions.Default)
        {
        }

        public InferenceSession(string modelPath, SessionOptions options)
        {
            var envHandle = OnnxRuntime.Handle;

            _nativeHandle = IntPtr.Zero;
            try
            {
                NativeApiStatus.VerifySuccess(NativeMethods.ONNXRuntimeCreateInferenceSession(envHandle, modelPath, options.NativeHandle, out _nativeHandle));
            
                // Initialize input/output metadata
                _inputMetadata = new Dictionary<string, NodeMetadata>();
                _outputMetadata = new Dictionary<string, NodeMetadata>();

                // get input count
                ulong inputCount = 0;
                NativeApiStatus.VerifySuccess(NativeMethods.ONNXRuntimeInferenceSessionGetInputCount(_nativeHandle, out inputCount));

                // get all the output names
                for (ulong i = 0; i < inputCount; i++)
                {
                    _inputMetadata[GetInputName(i)] = new NodeMetadata();  //TODO: fill the shape/type when C-api available
                }

                // get output count
                ulong outputCount = 0;
                NativeApiStatus.VerifySuccess(NativeMethods.ONNXRuntimeInferenceSessionGetOutputCount(_nativeHandle, out outputCount));

                // get all the output names
                for (ulong i = 0; i < outputCount; i++)
                {
                    _outputMetadata[GetOutputName(i)] = new NodeMetadata();  //TODO: fill the shape/type when C-api available
                }
            }
            catch (OnnxRuntimeException e)
            {
                if (_nativeHandle != IntPtr.Zero)
                {
                    NativeMethods.ReleaseONNXSession(_nativeHandle);
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

        public ModelMetadata ModelMetadata
        {
            get
            {
                return new ModelMetadata(); //TODO: implement
            }
        }

        public IReadOnlyCollection<NamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs, RunOptions options = new RunOptions())
        {
            string[] outputNames = new string[_outputMetadata.Count];
            _outputMetadata.Keys.CopyTo(outputNames, 0);
            return Run(inputs, outputNames, options);
        }

        /// <summary>
        /// Runs the loaded model for the given inputs, and fetches the specified outputs in <paramref name="outputNames"/>.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="outputNames"></param>
        /// <param name="options"></param>
        /// <returns>Output Tensors in a Dictionary</returns>
        public IReadOnlyCollection<NamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs, IReadOnlyCollection<string> outputNames, RunOptions options = new RunOptions())
        {
            var inputNames = new string[inputs.Count];
            var inputTensors = new IntPtr[inputs.Count];
            var pinnedBufferHandles = new System.Buffers.MemoryHandle[inputs.Count];

            int offset = 0;
            foreach (var input in inputs)
            {
                inputNames[offset] = input.Name;

                // create Tensor fromt the input if feasible, else throw notsupported exception for now
                input.ToNativeOnnxValue(out inputTensors[offset], out pinnedBufferHandles[offset]);

                offset++;
            }

            string[] outputNamesArray = outputNames.ToArray();
            IntPtr[] outputValueArray = new IntPtr[outputNames.Count];

            IntPtr status = NativeMethods.ONNXRuntimeRunInference(
                                                this._nativeHandle,
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
                var result = new List<NamedOnnxValue>();
                for (uint i = 0; i < outputValueArray.Length; i++)
                {
                    result.Add(NamedOnnxValue.CreateFromOnnxValue(outputNamesArray[i], outputValueArray[i]));  
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
                        NativeMethods.ReleaseONNXValue(outputValueArray[i]);
                    }
                }
                throw e;
            }
            finally
            {
                // always unpin the input buffers, and delete the native Onnx value objects
                for (int i = 0; i < inputs.Count; i++)
                {
                    NativeMethods.ReleaseONNXValue(inputTensors[i]); // this should not release the buffer, but should delete the native tensor object
                    pinnedBufferHandles[i].Dispose();
                }
            }
            
        }


        #endregion

        #region private methods
        private string GetOutputName(ulong index)
        {
            IntPtr nameHandle = IntPtr.Zero;
            string str = null;

            IntPtr  status = NativeMethods.ONNXRuntimeInferenceSessionGetOutputName(
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

            IntPtr status = NativeMethods.ONNXRuntimeInferenceSessionGetInputName(
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
                NativeMethods.ReleaseONNXSession(_nativeHandle);
            }
        }

        #endregion

    }

    public struct NodeMetadata
    {
        //TODO: currently shape and type is not available in C-api, so this struct may change based on implementation
        public uint[] Shape
        {
            get; internal set;
        }
        public System.Type Type
        {
            get; internal set;
        }
    }


    public struct ModelMetadata
    {
        //TODO: placeholder for Model metadata. Currently C-API does not expose this
    }

}
