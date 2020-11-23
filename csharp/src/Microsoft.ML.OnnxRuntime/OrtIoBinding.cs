// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// This class enable to bind inputs and outputs to pre-allocated
    /// memory. This enables interesting scenarios. For example, if your input
    /// already resides in some pre-allocated memory even if on a device you bind
    /// that piece of memory to an input name and shape and onnxruntime will use that as input.
    /// Other traditional inputs can also be bound that already exists as Tensors
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
        /// Bind a piece of pre-allocated native memory as a OrtValue Tensor with a given shape
        /// to an input with a given name. The model will read the specified input from that memory
        /// possibly avoiding the need to copy between devices. OrtMemoryAllocation continues to own
        /// the chunk of native memory and should be alive until the end of execution.
        /// The size of the allocation can not be less than required.
        /// by the Tensor of the given size.
        /// </summary>
        /// <param name="name">of the input</param>
        /// <param name="elementType">Tensor element type</param>
        /// <param name="shape"></param>
        /// <param name="allocation">native memory allocation</param>
        public void BindInput(string name, Tensors.TensorElementType elementType, long[] shape, OrtMemoryAllocation allocation)
        {
            using (var ortValue = OrtValue.CreateTensorValueWithData(allocation.Info,
                                                                    elementType,
                                                                    shape,
                                                                    allocation.Pointer, allocation.Size))
                BindInputOrOutput(name, ortValue.Handle, true);
        }

        /// <summary>
        /// Bind the input with the given name as an OrtValue Tensor allocated in pinned managed memory.
        /// Instance of FixedBufferOnnxValue owns the memory and should be alive until the end of execution.
        /// </summary>
        /// <param name="name">name of input</param>
        /// <param name="fixedValue"></param>
        public void BindInput(string name, FixedBufferOnnxValue fixedValue)
        {
            if(fixedValue.OnnxValueType != OnnxValueType.ONNX_TYPE_TENSOR)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument, "Binding works only with Tensors");
            }
            BindInputOrOutput(name, fixedValue.Value.Handle, true);
        }

        /// <summary>
        /// Bind model output to an OrtValue as Tensor with a given type and shape. An instance of OrtMemoryAllocaiton
        /// owns the memory and should be alive for the time of execution.The size of the allocation can not be less than required
        /// by the Tensor of the given size.
        /// </summary>
        /// <param name="name">of the output</param>
        /// <param name="elementType">tensor element type</param>
        /// <param name="shape">tensor shape</param>
        /// <param name="allocation">allocated memory</param>
        public void BindOutput(string name, Tensors.TensorElementType elementType, long[] shape, OrtMemoryAllocation allocation)
        {
            using (var ortValue = OrtValue.CreateTensorValueWithData(allocation.Info,
                                                                    elementType,
                                                                    shape,
                                                                    allocation.Pointer, allocation.Size))
                BindInputOrOutput(name, ortValue.Handle, false);
        }

        /// <summary>
        /// Bind model output to a given instance of FixedBufferOnnxValue which owns the underlying
        /// pinned managed memory and should be alive for the time of execution.
        /// </summary>
        /// <param name="name">of the output</param>
        /// <param name="fixedValue"></param>
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
            var utf8NamePinned = GCHandle.Alloc(NativeOnnxValueHelper.StringToZeroTerminatedUtf8(name), GCHandleType.Pinned);
            using (var pinnedName = new PinnedGCHandle(utf8NamePinned))
            NativeApiStatus.VerifySuccess(NativeMethods.OrtBindOutputToDevice(handle, pinnedName.Pointer, memInfo.Pointer));
        }

        /// <summary>
        /// Internal helper
        /// </summary>
        /// <param name="name"></param>
        /// <param name="ortValue"></param>
        /// <param name="isInput"></param>
        private void BindInputOrOutput(string name, IntPtr ortValue, bool isInput)
        {
            var utf8NamePinned = GCHandle.Alloc(NativeOnnxValueHelper.StringToZeroTerminatedUtf8(name), GCHandleType.Pinned);
            using (var pinnedName = new PinnedGCHandle(utf8NamePinned))
            {
                if (isInput)
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtBindInput(handle, pinnedName.Pointer, ortValue));
                }
                else
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtBindOutput(handle, pinnedName.Pointer, ortValue));
                }
            }
        }

        /// <summary>
        /// Returns an array of output names in the same order they were bound
        /// </summary>
        /// <returns>array of output names</returns>
        public string[] GetOutputNames()
        {
            IntPtr buffer = IntPtr.Zero;
            IntPtr lengths = IntPtr.Zero;
            UIntPtr count = UIntPtr.Zero;
            var allocator = OrtAllocator.DefaultInstance;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetBoundOutputNames(handle, allocator.Pointer, out buffer, out lengths, out count));

            if(count.Equals(UIntPtr.Zero))
            {
                return new string[0];
            }

            using (var bufferAllocation = new OrtMemoryAllocation(allocator, buffer, 0))
            using (var lengthsAllocation = new OrtMemoryAllocation(allocator, lengths, 0))
            {
                int outputCount = (int)count;
                var lens = new int[outputCount];
                int totalLength = 0;
                for(int i = 0; i < outputCount; ++i)
                {
                    var len =(int)Marshal.ReadIntPtr(lengths, IntPtr.Size * i);
                    lens[i] = len;
                    totalLength += len;
                }

                var stringData = new byte[totalLength];
                Marshal.Copy(buffer, stringData, 0, stringData.Length);

                string[] result = new string[outputCount];
                int readOffset = 0;
                for(int i = 0; i < outputCount; ++i)
                {
                    var strLen = lens[i];
                    result[i] = Encoding.UTF8.GetString(stringData, readOffset, strLen);
                    readOffset += strLen;
                }
                return result;
            }
        }

        /// <summary>
        /// This fetches bound outputs after running the model with RunWithBinding()
        /// </summary>
        /// <returns>IDisposableReadOnlyCollection<OrtValue></returns>
        public IDisposableReadOnlyCollection<OrtValue> GetOutputValues()
        {
            IntPtr ortValues = IntPtr.Zero;
            UIntPtr count = UIntPtr.Zero;
            var allocator = OrtAllocator.DefaultInstance;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetBoundOutputValues(handle, allocator.Pointer, out ortValues, out count));

            if(count.Equals(UIntPtr.Zero))
            {
                return new DisposableList<OrtValue>();
            }

            using(var ortValuesAllocation = new OrtMemoryAllocation(allocator, ortValues, 0))
            {
                int outputCount = (int)count;
                var ortList = new DisposableList<OrtValue>(outputCount);
                try
                {
                    for(int i = 0; i < outputCount; ++i)
                    {
                        IntPtr ortValue = Marshal.ReadIntPtr(ortValues, IntPtr.Size * i);
                        ortList.Add(new OrtValue(ortValue));
                    }
                } catch(Exception e)
                {
                    ortList.Dispose();
                    throw e;
                }
                return ortList;
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
