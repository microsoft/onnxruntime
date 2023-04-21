// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Buffers;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// Provides access from the underlying object that owns disposable OrtValue
    /// The returned value does not own the actual memory and does nothing on Dispose()
    /// </summary>
    internal interface IOrtValueOwner : IDisposable
    {
        OrtValue Value { get; }
    }

    /// <summary>
    /// This class is used in conjunction with DisposableNamedOnnxValue to 
    /// own native collection OrtValue and dispose of it along with any DisposableNamedOnnxValues
    /// </summary>
    internal class NativeOrtValueCollectionOwner<T> : IOrtValueOwner, IDisposable
        where T : IDisposable
    {
        private OrtValue _ortValue;
        private DisposableList<T> _disposables;
        bool _disposed = false;

        internal NativeOrtValueCollectionOwner(OrtValue ortValue, DisposableList<T> disposables)
        {
            Debug.Assert(ortValue.IsOwned);
            _ortValue = new OrtValue(ortValue.Disown());
            _disposables = disposables;
        }

        #region IOrtValueOwner
        /// <summary>
        /// Returns a non-owning ortValue
        /// </summary>
        public OrtValue Value { get { return new OrtValue(_ortValue.Handle, false); } }
        #endregion IOrtValueOwner

        #region Disposable
        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
            {
                return;
            }

            // dispose managed state (managed objects).
            if (disposing)
            {
                if (_disposables != null)
                {
                    _disposables.Dispose();
                    _disposables = null;
                }
                // _ortValueHolder can be null when no native memory is involved
                if (_ortValue != null)
                {
                    _ortValue.Dispose();
                    _ortValue = null;
                }
                _disposed = true;
            }
        }
        public void Dispose()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        #endregion Disposable
    }

    /// <summary>
    /// This helper class owns the underlying OrtValue that is assumed to be a Tensor,
    /// it does not support any other ortValues and caches Tensor properties.
    /// 
    /// It is easy to expose as a Tensor<T> as DenseTensor can take Memory Mapping from
    /// this.
    /// 
    /// This class is disposable because of the MemoryManager inheritance
    /// </summary>
    /// <typeparam name="T"></typeparam>
    internal class OrtValueTensor<T> : MemoryManager<T>, IOrtValueOwner
    {
        private OrtValue _ortValue; // Disposable
        private readonly IntPtr _dataBufferPointer;    // pointer to mutable tensor data in native memory
        private readonly string[] _dataBufferAsString; // string tensor values copied into managed memory

        /// <summary>
        /// Constructs an instance and takes ownership of ortValue on success
        /// </summary>
        /// <param name="ortValue">ortValue that is a Tensor</param>
        public OrtValueTensor(OrtValue ortValue)
        {
            Type type = null;
            int width = 0;
            IntPtr typeAndShape = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorTypeAndShape(ortValue.Handle, out typeAndShape));
            try
            {
                TensorElementType elemType;
                {
                    IntPtr el_type;
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorElementType(typeAndShape, out el_type));
                    elemType = (TensorElementType)el_type;
                }

                if (!TensorElementTypeConverter.GetTypeAndWidth(elemType, out type, out width))
                {
                    throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                        "Unable to query type information for data type: " + elemType.ToString());
                }

                if (typeof(T) != type)
                {
                    var message = String.Format("The OrtValueTensor<T> type being instantiated for T = : {0} while supplied OrtValue contains T = {1}",
                        typeof(T), type);
                    throw new OnnxRuntimeException(ErrorCode.InvalidArgument, message);
                }

                ElementType = elemType;
                ElementWidth = width;
                UIntPtr dimension;
                long count;
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetDimensionsCount(typeAndShape, out dimension));
                {
                    IntPtr el_count;
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorShapeElementCount(typeAndShape, out el_count));  // count can be negative. 
                    count = (long)el_count;
                }
                if (count < 0)
                {
                    throw new NotSupportedException("Symbolic dimensions in the tensor is not supported");
                }

                long[] shape = new long[dimension.ToUInt64()];
                NativeApiStatus.VerifySuccess(NativeMethods.OrtGetDimensions(typeAndShape, shape, dimension)); //Note: shape must be alive during the call

                Count = (int)count;
                Dimensions = new int[dimension.ToUInt64()];
                for (ulong i = 0; i < dimension.ToUInt64(); i++)
                {
                    Dimensions[i] = (int)shape[i];
                }

                if (elemType != TensorElementType.String)
                {
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorMutableData(ortValue.Handle, out _dataBufferPointer));
                }
                else
                {
                    var offsets = new UIntPtr[Count];
                    NativeApiStatus.VerifySuccess(NativeMethods.OrtGetStringTensorDataLength(ortValue.Handle, out UIntPtr strLen));
                    var dataBuffer = new byte[strLen.ToUInt64()];

                    NativeApiStatus.VerifySuccess(
                        NativeMethods.OrtGetStringTensorContent(
                        ortValue.Handle, dataBuffer, strLen,
                        offsets,
                        (UIntPtr)Count));

                    _dataBufferAsString = new string[Count];

                    for (var i = 0; i < offsets.Length; i++)
                    {
                        var length = (i == offsets.Length - 1)
                            ? strLen.ToUInt64() - offsets[i].ToUInt64()
                            : offsets[i + 1].ToUInt64() - offsets[i].ToUInt64();
                        // ORT API specifies strings always in UTF-8, no trailing null, no leading BOM
                        _dataBufferAsString[i] = Encoding.UTF8.GetString(dataBuffer, (int)offsets[i], (int)length);
                    }
                }
                // Transfer ownership
                _ortValue = new OrtValue(ortValue.Disown());
            }
            finally
            {
                NativeMethods.OrtReleaseTensorTypeAndShapeInfo(typeAndShape);
            }
        }

        /// <summary>
        /// Returns a non-owning copy of OrtValue so the
        /// result can not release native memory
        /// </summary>
        public OrtValue Value { get { return new OrtValue(_ortValue.Handle, false); } }

        public bool IsDisposed { get; private set; } = false;

        public int[] Dimensions { get; }

        public int Rank => Dimensions.Length;

        public int Count { get; }

        public int ElementWidth { get; }

        public Tensors.TensorElementType ElementType { get; }

        /// <summary>
        /// Used by MemoryManager to produce Memory Property
        /// </summary>
        /// <returns>SpanT</returns>
        public override Span<T> GetSpan()
        {
            if (IsDisposed)
                throw new ObjectDisposedException(nameof(OrtValueTensor<T>));
            Span<T> span = null;
            unsafe
            {
                span = new Span<T>((void*)_dataBufferPointer, Count);
            }

            return span;
        }
        public Memory<String> GetBytesAsStringMemory()
        {
            if (IsDisposed)
                throw new ObjectDisposedException(nameof(OrtValueTensor<T>));

            if (typeof(T) != typeof(string))
                throw new NotSupportedException(nameof(OrtValueTensor<T>.GetBytesAsStringMemory) + ": T must be byte");

            return (_dataBufferAsString == null) ? new Memory<string>() : new Memory<string>(_dataBufferAsString);
        }

        /// <summary>
        /// Satisfy MemoryManager abstract implementation
        /// </summary>
        /// <param name="elementIndex"></param>
        /// <returns></returns>
        public override MemoryHandle Pin(int elementIndex = 0)
        {
            //Note: always pin the full buffer and return
            unsafe
            {
                if (elementIndex >= Count)
                {
                    throw new ArgumentOutOfRangeException(nameof(elementIndex));
                }
                return new MemoryHandle(new IntPtr(_dataBufferPointer.ToInt64() + (long)elementIndex * ElementWidth).ToPointer());
            }
        }

        // MemoryHandle returned above by Pin() should be disposed.
        // Unpin() is purely to satisfy the interface.
        public override void Unpin() { }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected override void Dispose(bool disposing)
        {
            if (IsDisposed)
            {
                return;
            }

            if (_ortValue != null)
            {
                _ortValue.Dispose();
                _ortValue = null;
            }
            IsDisposed = true;
        }
    }
}
