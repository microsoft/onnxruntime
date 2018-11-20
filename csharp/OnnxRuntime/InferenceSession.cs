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
            var envHandle = OnnxRuntime.Instance.NativeHandle;
            IntPtr outputHandle;

            IntPtr status = NativeMethods.ONNXRuntimeCreateInferenceSession(envHandle, modelPath, options.NativeHandle, out outputHandle);

            _nativeHandle = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(status);
            _nativeHandle = outputHandle;
        }

        public IReadOnlyDictionary<string, NodeMetadata> InputMetadata
        {
            get
            {
                return null;  // TODO: implement
            }
        }

        public IReadOnlyDictionary<string, NodeMetadata> OutputMetadata
        {
            get
            {
                return null; // TODO: implement
            }
        }

        public ModelMetadata ModelMetadata
        {
            get
            {
                return new ModelMetadata(); //TODO: implement
            }
        }

        public IReadOnlyList<NamedOnnxValue> Run(IReadOnlyList<NamedOnnxValue> inputs, RunOptions options = new RunOptions())
        {
            var inputNames = new string[inputs.Count];
            var inputTensors = new IntPtr[inputs.Count];
            var pinnedBufferHandles = new System.Buffers.MemoryHandle[inputs.Count];

            for (int i = 0; i < inputs.Count; i++)
            {
                inputNames[i] = inputs[i].Name;

                // create Tensor fromt the inputs[i] if feasible, else throw notsupported exception for now
                inputs[i].ToNativeOnnxValue(out inputTensors[i], out pinnedBufferHandles[i]);
            }

            IntPtr outputValueList = IntPtr.Zero;
            ulong outputLength = 0;
            IntPtr status = NativeMethods.ONNXRuntimeRunInferenceAndFetchAll(
                                this._nativeHandle,
                                inputNames,
                                inputTensors,
                                (uint)(inputTensors.Length),
                                out outputValueList,
                                out outputLength
                            );  //Note: the inputTensors and pinnedBufferHandles must be alive for the duration of the call

            try
            {
                NativeApiStatus.VerifySuccess(status);
                var result = new List<NamedOnnxValue>();
                for (uint i = 0; i < outputLength; i++)
                {
                    IntPtr tensorValue = NativeMethods.ONNXRuntimeONNXValueListGetNthValue(outputValueList, i);
                    result.Add(NamedOnnxValue.CreateFromOnnxValue(Convert.ToString(i), tensorValue));  // TODO: currently Convert.ToString(i) is used instead of the output name, for the absense of C-api.
                                                                                                       // Will be fixed as soon as the C-api for output name is available
                }

                return result;
            }
            catch (OnnxRuntimeException e)
            {
                //clean up the individual output tensors if it is not null;
                if (outputValueList != IntPtr.Zero)
                {
                    for (uint i = 0; i < outputLength; i++)
                    {
                        IntPtr tensorValue = NativeMethods.ONNXRuntimeONNXValueListGetNthValue(outputValueList, i);
                        NativeMethods.ReleaseONNXValue(tensorValue);
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

                // always release the output value list, because the individual tensor pointers are already obtained.
                if (outputValueList != IntPtr.Zero)
                {
                    NativeMethods.ReleaseONNXValueList(outputValueList);
                }
            }
        }



        /// <summary>
        /// Runs the loaded model for the given inputs, and fetches the specified outputs in <paramref name="outputNames"/>.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="outputNames"></param>
        /// <param name="options"></param>
        /// <returns>Output Tensors in a Dictionary</returns>
        public IReadOnlyList<NamedOnnxValue> Run(IReadOnlyList<NamedOnnxValue> inputs, ICollection<string> outputNames, RunOptions options = new RunOptions())
        {
            //TODO: implement
            return null;
        }


        #endregion

        #region private methods


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
        //placeholder for Model metadata. Python API has this
    }

}
