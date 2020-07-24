using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Represents a disposable OrtValue
    /// </summary>
    public class OrtValue : IDisposable
    {
        /// <summary>
        /// Use factory methods to instantiate
        /// </summary>
        /// <param name="handle"></param>
        internal OrtValue(IntPtr handle)
        {
            Handle = handle;
        }

        internal IntPtr Handle { get; private set; }

        /// <summary>
        /// This internal interface is used to transfer ownership to
        /// DisposableNamdOnnxValue class.
        /// </summary>
        /// <returns></returns>
        internal IntPtr TakeOwnership()
        {
            var handle = Handle;
            Handle = IntPtr.Zero;
            return handle;
        }

        /// <summary>
        /// Factory method to construct an OrtValue of Tensor type on top of pre-allocated memory.
        /// This can be a piece of native memory allocated by OrtAllocator (possibly on a device)
        /// or a piece of pinned managed memory.
        /// 
        /// The resulting OrtValue does not own the underlying memory buffer and will not attempt to
        /// deallocated it.
        /// </summary>
        /// <param name="memInfo">Memory Info. For managed memory it is a default cpu.
        ///                       For Native memory must be obtained from the allocator or OrtMemoryAllocation instance</param>
        /// <param name="elementType">DataType for the Tensor</param>
        /// <param name="shape">Tensor shape</param>
        /// <param name="dataBuffer">Pointer to a raw memory buffer</param>
        /// <param name="bufferLength">Buffer length in bytes</param>
        /// <returns>A disposable instance of OrtValue</returns>
        public static OrtValue CreateTensorValueWithData(OrtMemoryInfo memInfo, TensorElementType elementType,
                                                         long[] shape,
                                                         IntPtr dataBuffer,
                                                         uint bufferLength)
        {
            var shapeSize = ArrayUtilities.GetSizeForShape(shape);
            if(shapeSize > bufferLength)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument, "Can not bind the shape to smaller buffer");
            }

            IntPtr ortValueHandle = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateTensorWithDataAsOrtValue(
                                    memInfo.Pointer,
                                    dataBuffer,
                                    (UIntPtr)bufferLength,
                                    shape,
                                    (UIntPtr)shape.Length,
                                    elementType,
                                    out ortValueHandle
                                ));
            return new OrtValue(ortValueHandle);
        }

        #region Disposable Support
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                // We have to surrender ownership to some legacy classes
                if (Handle != IntPtr.Zero)
                {
                    NativeMethods.OrtReleaseValue(Handle);
                    // Prevent use after disposal
                    Handle = IntPtr.Zero;
                }
            }
        }
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        // No need for the finalizer
        #endregion
    }
}
