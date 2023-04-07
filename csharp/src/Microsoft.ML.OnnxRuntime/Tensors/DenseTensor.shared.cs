// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is copied and adapted from the following git repository -
// https://github.com/dotnet/corefx
// Commit ID: bdd0814360d4c3a58860919f292a306242f27da1
// Path: /src/System.Numerics.Tensors/src/System/Numerics/Tensors/DenseTensor.cs
// Original license statement below -

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Runtime.InteropServices;
using System;

namespace Microsoft.ML.OnnxRuntime.Tensors
{
    /// <summary>
    /// Represents a multi-dimensional collection of objects of type T that can be accessed by indices.  
    /// DenseTensor stores values in a contiguous sequential block of memory where all values are represented.
    /// </summary>
    /// <typeparam name="T">
    /// Type contained within the Tensor. Typically a value type such as int, double, float, etc.
    /// </typeparam>
    public class DenseTensor<T> : Tensor<T>
    {
        private readonly Memory<T> memory;

        internal DenseTensor(Array fromArray, bool reverseStride = false) : base(fromArray, reverseStride)
        {
            // copy initial array
            var backingArray = new T[fromArray.Length];

            int index = 0;
            if (reverseStride)
            {
                // Array is always row-major
                var sourceStrides = ArrayUtilities.GetStrides(dimensions);

                foreach (var item in fromArray)
                {
                    var destIndex = ArrayUtilities.TransformIndexByStrides(index++, sourceStrides, false, strides);
                    backingArray[destIndex] = (T)item;
                }
            }
            else
            {
                foreach (var item in fromArray)
                {
                    backingArray[index++] = (T)item;
                }
            }

            memory = backingArray;
        }

        /// <summary>
        /// Initializes a rank-1 Tensor using the specified <paramref name="length"/>.
        /// </summary>
        /// <param name="length">Size of the 1-dimensional tensor</param>
        public DenseTensor(int length) : base(length)
        {
            memory = new T[length];
        }

        /// <summary>
        /// Initializes a rank-n Tensor using the dimensions specified in <paramref name="dimensions"/>.
        /// </summary>
        /// <param name="dimensions">
        /// An span of integers that represent the size of each dimension of the DenseTensor to create.
        /// </param>
        /// <param name="reverseStride">
        /// False (default) to indicate that the first dimension is most major (farthest apart) and the last dimension 
        /// is most minor (closest together): akin to row-major in a rank-2 tensor.  
        /// True to indicate that the last dimension is most major (farthest apart) and the first dimension is most 
        /// minor (closest together): akin to column-major in a rank-2 tensor.
        /// </param>
        public DenseTensor(ReadOnlySpan<int> dimensions, bool reverseStride = false) : base(dimensions, reverseStride)
        {
            memory = new T[Length];
        }

        /// <summary>
        /// Constructs a new DenseTensor of the specified dimensions, wrapping existing backing memory for the contents.
        /// </summary>
        /// <param name="memory"></param>
        /// <param name="dimensions">
        /// An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
        /// <param name="reverseStride">
        /// False (default) to indicate that the first dimension is most major (farthest apart) and the last dimension 
        /// is most minor (closest together): akin to row-major in a rank-2 tensor.  
        /// True to indicate that the last dimension is most major (farthest apart) and the first dimension is most 
        /// minor (closest together): akin to column-major in a rank-2 tensor.
        /// </param>
        public DenseTensor(Memory<T> memory, ReadOnlySpan<int> dimensions, bool reverseStride = false) 
            : base(dimensions, reverseStride)
        {
            this.memory = memory;

            if (Length != memory.Length)
            {
                throw new ArgumentException(
                    $"Length of {nameof(memory)} ({memory.Length}) must match product of " +
                    $"{nameof(dimensions)} ({Length}).");
            }
        }

        /// <summary>
        /// Memory storing backing values of this tensor.
        /// </summary>
        public Memory<T> Buffer => memory;

        /// <summary>
        /// Gets the value at the specified index, where index is a linearized version of n-dimension indices 
        /// using strides. For a scalar, use index = 0
        /// </summary>
        /// <param name="index">An integer index computed as a dot-product of indices.</param>
        /// <returns>The value at the specified position in this Tensor.</returns>
        public override T GetValue(int index)
        {
            return Buffer.Span[index];
        }

        /// <summary>
        /// Sets the value at the specified index, where index is a linearized version of n-dimension indices 
        /// using strides. For a scalar, use index = 0
        /// </summary>
        /// <param name="index">An integer index computed as a dot-product of indices.</param>
        /// <param name="value">The new value to set at the specified position in this Tensor.</param>
        public override void SetValue(int index, T value)
        {
            Buffer.Span[index] = value;
        }

        /// <summary>
        /// Overrides Tensor.CopyTo(). Copies the content of the Tensor
        /// to the specified array starting with arrayIndex
        /// </summary>
        /// <param name="array">destination array</param>
        /// <param name="arrayIndex">start index</param>
        protected override void CopyTo(T[] array, int arrayIndex)
        {
            if (array == null)
            {
                throw new ArgumentNullException(nameof(array));
            }
            if (array.Length < arrayIndex + Length)
            {
                throw new ArgumentException(
                    "The number of elements in the Tensor is greater than the available space from index to " + 
                    "the end of the destination array.", nameof(array));
            }

            Buffer.Span.CopyTo(array.AsSpan(arrayIndex));
        }

        /// <summary>
        /// Determines the index of a specific item in the Tensor&lt;T&gt;.
        /// </summary>
        /// <param name="item">Object to locate</param>
        /// <returns>The index of item if found in the tensor; otherwise, -1</returns>
        protected override int IndexOf(T item)
        {
            // TODO: use Span.IndexOf when/if it removes the IEquatable type constraint
            if (MemoryMarshal.TryGetArray<T>(Buffer, out var arraySegment))
            {
                var result = Array.IndexOf(arraySegment.Array, item, arraySegment.Offset, arraySegment.Count);
                if (result != -1)
                {
                    result -= arraySegment.Offset;
                }
                return result;
            }
            else
            {
                return base.IndexOf(item);
            }
        }

        /// <summary>
        /// Creates a shallow copy of this tensor, with new backing storage.
        /// </summary>
        /// <returns>A shallow copy of this tensor.</returns>
        public override Tensor<T> Clone()
        {
            // create copy
            return new DenseTensor<T>(new Memory<T>(memory.ToArray()), dimensions, IsReversedStride);
        }

        /// <summary>
        /// Creates a new Tensor of a different type with the specified dimensions and the same layout as this tensor 
        /// with elements initialized to their default value.
        /// </summary>
        /// <typeparam name="TResult">Type contained in the returned Tensor.</typeparam>
        /// <param name="dimensions">
        /// An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
        /// <returns>A new tensor with the same layout as this tensor but different type and dimensions.</returns>
        public override Tensor<TResult> CloneEmpty<TResult>(ReadOnlySpan<int> dimensions)
        {
            return new DenseTensor<TResult>(dimensions, IsReversedStride);
        }

        /// <summary>
        /// Reshapes the current tensor to new dimensions, using the same backing storage.
        /// </summary>
        /// <param name="dimensions">
        /// An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
        /// <returns>A new tensor that reinterprets backing Buffer of this tensor with different dimensions.</returns>
        public override Tensor<T> Reshape(ReadOnlySpan<int> dimensions)
        {
 
            var newSize = ArrayUtilities.GetProduct(dimensions);

            if (newSize != Length)
            {
                throw new ArgumentException($"Cannot reshape array due to mismatch in lengths, " +
                    "currently {Length} would become {newSize}.", nameof(dimensions));
            }

            return new DenseTensor<T>(Buffer, dimensions, IsReversedStride);
        }
    }
}
