// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// This class enables binding of inputs and/or outputs to pre-allocated
    /// memory. This enables interesting scenarios. For example, if your input
    /// already resides in some pre-allocated memory like GPU, you can bind
    /// that piece of memory to an input name and shape and onnxruntime will use that as input.
    /// Other traditional inputs can also be bound that already exists as Tensors.
    ///
    /// Note, that this arrangement is designed to minimize data copies and to that effect
    /// your memory allocations must match what is expected by the model, whether you run on
    /// CPU or GPU. Data copy will still be made, if your pre-allocated memory location does not
    /// match the one expected by the model. However, copies with OrtIoBindings are only done once,
    /// at the time of the binding, not at run time. This means, that if your input data required a copy,
    /// your further input modifications would not be seen by onnxruntime unless you rebind it, even if it is
    /// the same buffer. If you require the scenario where data is copied, OrtIOBinding may not be the best match
    /// for your use case. The fact that data copy is not made during runtime also has performance implications.
    /// 
    /// Making OrtValue first class citizen in ORT C# API practically obsoletes all of the existing overloads
    /// because OrtValue can be created on top of the all other types of memory. No need to designate it as external
    /// or Ort allocation or wrap it in FixedBufferOnnxValue. The latter does not support rebinding or memory other than
    /// CPU anyway.
    /// 
    /// In fact, one can now create OrtValues over arbitrary pieces of memory, managed, native, stack and device(gpu)
    /// and feed them to the model and achieve the same effect without using IOBinding class.
    /// </summary>
    public class OrtIoBinding : SafeHandle
    {
        /// <summary>
        /// Use InferenceSession.CreateIoBinding()
        /// </summary>
        /// <param name="session"></param>
        internal OrtIoBinding(InferenceSession session)
            : base(IntPtr.Zero, true)
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateIoBinding(session.Handle, out handle));
        }

        internal IntPtr Handle
        {
            get
            {
                return handle;
            }
        }

        /// <summary>
        /// Overrides SafeHandle.IsInvalid
        /// </summary>
        /// <value>returns true if handle is equal to Zero</value>
        public override bool IsInvalid { get { return handle == IntPtr.Zero; } }


        /// <summary>
        /// This is the preferable and universal way to bind input to OrtValue.
        /// This way you retain control over the original value, can modify the data
        /// using OrtValue interfaces between the runs.
        /// 
        /// You can also create OrtValue on all kinds of memory, managed, native, stack and device(gpu).
        /// </summary>
        /// <param name="name">input name</param>
        /// <param name="ortValue"></param>
        public void BindInput(string name, OrtValue ortValue)
        {
            BindInputOrOutput(name, ortValue.Handle, true);
        }

        /// <summary>
        /// Bind a piece of pre-allocated native memory as a OrtValue Tensor with a given shape
        /// to an input with a given name. The model will read the specified input from that memory
        /// possibly avoiding the need to copy between devices. OrtMemoryAllocation continues to own
        /// the chunk of native memory, and the allocation should be alive until the end of execution.
        /// </summary>
        /// <param name="name">of the input</param>
        /// <param name="elementType">Tensor element type</param>
        /// <param name="shape"></param>
        /// <param name="allocation">native memory allocation</param>
        public void BindInput(string name, Tensors.TensorElementType elementType, long[] shape, OrtMemoryAllocation allocation)
        {
            BindOrtAllocation(name, elementType, shape, allocation, true);
        }

        /// <summary>
        /// Bind externally (not from OrtAllocator) allocated memory as input.
        /// The model will read the specified input from that memory
        /// possibly avoiding the need to copy between devices. The user code continues to own
        /// the chunk of externally allocated memory, and the allocation should be alive until the end of execution.
        /// </summary>
        /// <param name="name">name</param>
        /// <param name="allocation">non ort allocated memory</param>
        [Obsolete("This BindInput overload is deprecated. Create OrtValue over an arbitrary piece of memory.")]
        public void BindInput(string name, OrtExternalAllocation allocation)
        {
            BindExternalAllocation(name, allocation, true);
        }

        /// <summary>
        /// Bind the input with the given name as an OrtValue Tensor allocated in pinned managed memory.
        /// Instance of FixedBufferOnnxValue owns the memory and should be alive until the end of execution.
        /// </summary>
        /// <param name="name">name of input</param>
        /// <param name="fixedValue"></param>
        [Obsolete("This BindInput overload is deprecated. Use of OrtValue based overload is recommended.")]
        public void BindInput(string name, FixedBufferOnnxValue fixedValue)
        {
            if (fixedValue.OnnxValueType != OnnxValueType.ONNX_TYPE_TENSOR)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument, "Binding works only with Tensors");
            }
            BindInputOrOutput(name, fixedValue.Value.Handle, true);
        }

        /// <summary>
        /// Blocks until device completes all preceding requested tasks.
        /// Useful for memory synchronization.
        /// </summary>
        public void SynchronizeBoundInputs()
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSynchronizeBoundInputs(handle));
        }

        /// <summary>
        /// This is the preferable and universal way to bind output via OrtValue.
        /// This way you retain control over the original value, can modify the data
        /// using OrtValue interfaces between the runs, rebind output to input if you
        /// are feeding data circular.
        /// </summary>
        /// <param name="name">output name</param>
        /// <param name="ortValue">OrtValue to bind</param>
        public void BindOutput(string name, OrtValue ortValue)
        {
            BindInputOrOutput(name, ortValue.Handle, false);
        }

        /// <summary>
        /// Bind model output to an OrtValue as Tensor with a given type and shape. An instance of OrtMemoryAllocaiton
        /// owns the memory and should be alive for the time of execution.
        /// </summary>
        /// <param name="name">of the output</param>
        /// <param name="elementType">tensor element type</param>
        /// <param name="shape">tensor shape</param>
        /// <param name="allocation">allocated memory</param>
        public void BindOutput(string name, Tensors.TensorElementType elementType, long[] shape, OrtMemoryAllocation allocation)
        {
            BindOrtAllocation(name, elementType, shape, allocation, false);
        }

        /// <summary>
        /// Bind externally (not from OrtAllocator) allocated memory as output.
        /// The model will read the specified input from that memory
        /// possibly avoiding the need to copy between devices. The user code continues to own
        /// the chunk of externally allocated memory, and the allocation should be alive until the end of execution.
        /// </summary>
        /// <param name="name">name</param>
        /// <param name="allocation">non ort allocated memory</param>
        [Obsolete("This BindOutput overload is deprecated. Create OrtValue over an arbitrary piece of memory.")]
        public void BindOutput(string name, OrtExternalAllocation allocation)
        {
            BindExternalAllocation(name, allocation, false);
        }

        /// <summary>
        /// Bind model output to a given instance of FixedBufferOnnxValue which owns the underlying
        /// pinned managed memory and should be alive for the time of execution.
        /// </summary>
        /// <param name="name">of the output</param>
        /// <param name="fixedValue"></param>
        [Obsolete("This BindOutput overload is deprecated. Use of OrtValue based overload is recommended.")]
        public void BindOutput(string name, FixedBufferOnnxValue fixedValue)
        {
            if (fixedValue.OnnxValueType != OnnxValueType.ONNX_TYPE_TENSOR)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument, "Binding works only with Tensors");
            }
            BindInputOrOutput(name, fixedValue.Value.Handle, false);
        }

        /// <summary>
        /// This function will bind model output with the given name to a device
        /// specified by the memInfo.
        /// </summary>
        /// <param name="name">output name</param>
        /// <param name="memInfo">instance of memory info</param>
        public void BindOutputToDevice(string name, OrtMemoryInfo memInfo)
        {
            var utf8 = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(name);
            NativeApiStatus.VerifySuccess(NativeMethods.OrtBindOutputToDevice(handle, utf8, memInfo.Pointer));
        }

        /// <summary>
        /// Blocks until device completes all preceding requested tasks.
        /// Useful for memory synchronization.
        /// </summary>
        public void SynchronizeBoundOutputs()
        {
            NativeApiStatus.VerifySuccess(NativeMethods.OrtSynchronizeBoundOutputs(handle));
        }

        /// <summary>
        /// Bind allocation obtained from an Ort allocator
        /// </summary>
        /// <param name="name">name </param>
        /// <param name="elementType">data type</param>
        /// <param name="shape">tensor shape</param>
        /// <param name="allocation">ort allocation</param>
        /// <param name="isInput">whether this is input or output</param>
        private void BindOrtAllocation(string name, Tensors.TensorElementType elementType, long[] shape,
            OrtMemoryAllocation allocation, bool isInput)
        {
            using (var ortValue = OrtValue.CreateTensorValueWithData(allocation.Info,
                                                                    elementType,
                                                                    shape,
                                                                    allocation.Pointer, allocation.Size))
                BindInputOrOutput(name, ortValue.Handle, isInput);
        }


        /// <summary>
        /// Bind external allocation as input or output.
        /// The allocation is owned by the user code.
        /// </summary>
        /// <param name="name">name </param>
        /// <param name="allocation">non ort allocated memory</param>
        /// <param name="isInput">whether this is an input or output</param>

        // Disable obsolete warning for this method until we remove the code
#pragma warning disable 0618
        private void BindExternalAllocation(string name, OrtExternalAllocation allocation, bool isInput)
        {
            using (var ortValue = OrtValue.CreateTensorValueWithData(allocation.Info,
                                                        allocation.ElementType,
                                                        allocation.Shape,
                                                        allocation.Pointer,
                                                        allocation.Size))
                BindInputOrOutput(name, ortValue.Handle, isInput);
        }
#pragma warning restore 0618

        /// <summary>
        /// Internal helper
        /// </summary>
        /// <param name="name"></param>
        /// <param name="ortValue"></param>
        /// <param name="isInput"></param>
        private void BindInputOrOutput(string name, IntPtr ortValue, bool isInput)
        {
            var utf8 = NativeOnnxValueHelper.StringToZeroTerminatedUtf8(name);
            if (isInput)
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtBindInput(handle, utf8, ortValue));
            }
            else
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtBindOutput(handle, utf8, ortValue));
            }
        }

        /// <summary>
        /// Returns an array of output names in the same order they were bound
        /// </summary>
        /// <returns>array of output names</returns>
        public string[] GetOutputNames()
        {
            var allocator = OrtAllocator.DefaultInstance;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetBoundOutputNames(handle,
                allocator.Pointer, out IntPtr buffer, out IntPtr lengths, out UIntPtr count));

            if ((ulong)count == 0)
            {
                return Array.Empty<string>();
            }

            int outputCount = (int)count;
            Span<IntPtr> lenSpan;
            unsafe
            {
                lenSpan = new Span<IntPtr>(lengths.ToPointer(), outputCount);
            }

            try
            {
                var result = new string[outputCount];

                int readOffset = 0;
                for (int i = 0; i < outputCount; ++i)
                {
                    var strLen = (int)lenSpan[i];
                    unsafe
                    {
                        var strStart = new IntPtr(buffer.ToInt64() + readOffset);
                        result[i] = Encoding.UTF8.GetString((byte*)strStart.ToPointer(), strLen);
                    }
                    readOffset += strLen;
                }

                return result;
            }
            finally
            {
                allocator.FreeMemory(lengths);
                allocator.FreeMemory(buffer);
            }
        }

        /// <summary>
        /// This fetches bound outputs after running the model with RunWithBinding()
        /// </summary>
        /// <returns>IDisposableReadOnlyCollection<OrtValue></returns>
        public IDisposableReadOnlyCollection<OrtValue> GetOutputValues()
        {
            var ortValues = GetOutputOrtValues();
            return new DisposableList<OrtValue>(ortValues);
        }

        internal OrtValue[] GetOutputOrtValues()
        {
            var allocator = OrtAllocator.DefaultInstance;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetBoundOutputValues(handle, allocator.Pointer,
                out IntPtr ortValues, out UIntPtr count));

            if ((ulong)count == 0)
            {
                return Array.Empty<OrtValue>();
            }

            int outputCount = (int)count;
            Span<IntPtr> srcSpan;
            unsafe
            {
                srcSpan = new Span<IntPtr>(ortValues.ToPointer(), outputCount);
            }

            try
            {
                OrtValue[] result = new OrtValue[outputCount];

                for (int i = 0; i < outputCount; ++i)
                {
                    result[i] = new OrtValue(srcSpan[i]);
                }

                return result;
            }
            catch (Exception)
            {
                // There is a very little chance that we throw
                for (int i = 0; i < srcSpan.Length; ++i)
                {
                    NativeMethods.OrtReleaseValue(srcSpan[i]);
                }
                throw;
            }
            finally
            {
                allocator.FreeMemory(ortValues);
            }
        }

        /// <summary>
        /// Clear all bound inputs and start anew
        /// </summary>
        public void ClearBoundInputs()
        {
            NativeMethods.OrtClearBoundInputs(handle);
        }

        /// <summary>
        /// Clear all bound outputs
        /// </summary>
        public void ClearBoundOutputs()
        {
            NativeMethods.OrtClearBoundOutputs(handle);
        }

        #region SafeHandle
        /// <summary>
        /// Overrides SafeHandle.ReleaseHandle() to properly dispose of
        /// the native instance of OrtIoBidning
        /// </summary>
        /// <returns>always returns true</returns>
        protected override bool ReleaseHandle()
        {
            NativeMethods.OrtReleaseIoBinding(handle);
            handle = IntPtr.Zero;
            return true;
        }

        #endregion
    }
}
