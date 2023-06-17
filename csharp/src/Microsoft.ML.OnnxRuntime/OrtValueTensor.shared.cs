// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Buffers;

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
    internal class NativeOrtValueCollectionOwner : IOrtValueOwner, IDisposable
    {
        private OrtValue _ortValue;
        private IDisposable _disposables;
        bool _disposed = false;

        /// <summary>
        /// _Ctor. Takes ownership of ortValue and sets it to null on success.
        /// </summary>
        /// <param name="ortValue">becomes null on success</param>
        /// <param name="disposables">A collection of disposables that support composed types.
        /// We stick them here and dispose them when this instance is disposed.
        /// </param>
        internal NativeOrtValueCollectionOwner(ref OrtValue ortValue, IDisposable disposables)
        {
            _ortValue = ortValue;
            ortValue = null;
            _disposables = disposables;
        }

        #region IOrtValueOwner
        /// <summary>
        /// Returns OrtValue that is owned by this instance
        /// </summary>
        public OrtValue Value { get { return _ortValue; } }
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
    /// This class is disposable because of the MemoryManager inheritance. Because this class
    /// always backs exactly only one DenseTensor<typeparamref name="T"/> instance, it does
    /// not implement ref-counting for Pin/Unpin.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    internal class OrtValueTensor<T> : MemoryManager<T>, IOrtValueOwner
    {
        private OrtValue _ortValue; // Disposable
        private readonly IntPtr _dataBufferPointer;    // pointer to mutable tensor data in native memory

        /// <summary>
        /// Constructs an instance and takes ownership of ortValue on success
        /// </summary>
        /// <param name="ortValue">ortValue that is a Tensor. It becomes null on successful return.</param>
        public OrtValueTensor(ref OrtValue ortValue)
        {
            var typeAndShapeInfo = ortValue.GetTensorTypeAndShape();
            TensorElementType elemType = typeAndShapeInfo.ElementDataType;

            var typeInfo = TensorBase.GetElementTypeInfo(elemType) ?? throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                    $"Unable to query type information for data type: {elemType}");

            if (typeof(T) != typeInfo.TensorType)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                    $"The OrtValueTensor<T> type being instantiated for T = [{typeof(T)}] while supplied OrtValue contains T = [{typeInfo.TensorType}]");
            }

            ElementType = elemType;
            ElementWidth = typeInfo.TypeSize;
            Count = (int)typeAndShapeInfo.ElementCount;

            Dimensions = Array.ConvertAll(typeAndShapeInfo.Shape, Convert.ToInt32);

            NativeApiStatus.VerifySuccess(NativeMethods.OrtGetTensorMutableData(ortValue.Handle, out _dataBufferPointer));

            // Transfer ownership
            _ortValue = ortValue;
            ortValue = null;
        }

        /// <summary>
        /// Returns OrtValue that is owned by this instance
        /// </summary>
        public OrtValue Value { get { return _ortValue; } }

        public bool IsDisposed { get; private set; } = false;

        public int[] Dimensions { get; }

        public int Rank => Dimensions.Length;

        public int Count { get; }

        public int ElementWidth { get; }

        public Tensors.TensorElementType ElementType { get; }

        /// <summary>
        /// Returns Span that is a view into the underlying native Tensor memory
        /// </summary>
        /// <returns>SpanT</returns>
        public override Span<T> GetSpan()
        {
            Span<T> span = null;
            unsafe
            {
                span = new Span<T>((void*)_dataBufferPointer, Count);
            }

            return span;
        }

        /// <summary>
        /// Satisfy MemoryManager abstract implementation.
        /// </summary>
        /// <param name="elementIndex">required for override</param>
        /// <returns></returns>
        public override MemoryHandle Pin(int elementIndex = 0)
        {
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
