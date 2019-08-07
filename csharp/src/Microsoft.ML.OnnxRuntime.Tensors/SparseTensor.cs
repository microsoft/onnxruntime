// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System;

namespace Microsoft.ML.OnnxRuntime.Tensors
{
    /// <summary>
    /// Represents a multi-dimensional collection of objects of type T that can be accessed by indices.  Unlike other Tensor&lt;T&gt; implementations SparseTensor&lt;T&gt; does not expose its backing storage.  It is meant as an intermediate to be used to build other Tensors, such as CompressedSparseTensor.  Unlike CompressedSparseTensor where insertions are O(n), insertions to SparseTensor&lt;T&gt; are nominally O(1).
    /// </summary>
    /// <typeparam name="T">type contained within the Tensor.  Typically a value type such as int, double, float, etc.</typeparam>
    internal class SparseTensor<T> : Tensor<T>
    {
        private readonly Dictionary<int, T> values;
        /// <summary>
        /// Constructs a new SparseTensor of the specifed dimensions, initial capacity, and stride ordering.
        /// </summary>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the SparseTensor to create.</param>
        /// <param name="reverseStride">False (default) to indicate that the first dimension is most major (farthest apart) and the last dimension is most minor (closest together): akin to row-major in a rank-2 tensor.  True to indicate that the last dimension is most major (farthest apart) and the first dimension is most minor (closest together): akin to column-major in a rank-2 tensor.</param>
        /// <param name="capacity">The number of non-zero values this tensor can store without resizing.</param>
        public SparseTensor(ReadOnlySpan<int> dimensions, bool reverseStride = false, int capacity = 0) : base(dimensions, reverseStride)
        {
            values = new Dictionary<int, T>(capacity);
        }

        internal SparseTensor(Dictionary<int, T> values, ReadOnlySpan<int> dimensions, bool reverseStride = false) : base(dimensions, reverseStride)
        {
            this.values = values;
        }

        internal SparseTensor(Array fromArray, bool reverseStride = false) : base(fromArray, reverseStride)
        {
            values = new Dictionary<int, T>(fromArray.Length);

            int index = 0;
            if (reverseStride)
            {
                // Array is always row-major
                var sourceStrides = ArrayUtilities.GetStrides(dimensions);

                foreach (T item in fromArray)
                {
                    if (!item.Equals(Zero))
                    {
                        var destIndex = ArrayUtilities.TransformIndexByStrides(index, sourceStrides, false, strides);
                        values[destIndex] = item;
                    }

                    index++;
                }
            }
            else
            {
                foreach (T item in fromArray)
                {
                    if (!item.Equals(Zero))
                    {
                        values[index] = item;
                    }

                    index++;
                }
            }
        }

        /// <summary>
        /// Gets the value at the specied index, where index is a linearized version of n-dimension indices using strides.
        /// </summary>
        /// <param name="index">An integer index computed as a dot-product of indices.</param>
        /// <returns>The value at the specified position in this Tensor.</returns>
        public override T GetValue(int index)
        {

            if (!values.TryGetValue(index, out T value))
            {
                value = Zero;
            }

            return value;
        }

        /// <summary>
        /// Sets the value at the specied index, where index is a linearized version of n-dimension indices using strides.
        /// </summary>
        /// <param name="index">An integer index computed as a dot-product of indices.</param>
        /// <param name="value">The new value to set at the specified position in this Tensor.</param>
        public override void SetValue(int index, T value)
        {
            if (value.Equals(Zero))
            {
                values.Remove(index);
            }
            else
            {
                values[index] = value;
            }
        }

        /// <summary>
        /// Get's the number on non-zero values currently being stored in this tensor.
        /// </summary>
        public int NonZeroCount => values.Count;

        /// <summary>
        /// Creates a shallow copy of this tensor, with new backing storage.
        /// </summary>
        /// <returns>A shallow copy of this tensor.</returns>
        public override Tensor<T> Clone()
        {
            var valueCopy = new Dictionary<int, T>(values);
            return new SparseTensor<T>(valueCopy, dimensions, IsReversedStride);
        }

        /// <summary>
        /// Creates a new Tensor of a different type with the specified dimensions and the same layout as this tensor with elements initialized to their default value.
        /// </summary>
        /// <typeparam name="TResult">Type contained in the returned Tensor.</typeparam>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the SparseTensor to create.</param>
        /// <returns>A new tensor with the same layout as this tensor but different type and dimensions.</returns>
        public override Tensor<TResult> CloneEmpty<TResult>(ReadOnlySpan<int> dimensions)
        {
            return new SparseTensor<TResult>(dimensions, IsReversedStride);
        }

        /// <summary>
        /// Reshapes the current tensor to new dimensions, using the same backing storage.
        /// </summary>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the SparseTensor to create.</param>
        /// <returns>A new tensor that reinterprets backing storage of this tensor with different dimensions.</returns>
        public override Tensor<T> Reshape(ReadOnlySpan<int> dimensions)
        {
            return new SparseTensor<T>(values, dimensions, IsReversedStride);
        }

        /// <summary>
        /// Creates a copy of this tensor as a DenseTensor&lt;T&gt;.  
        /// </summary>
        /// <returns>A copy of this tensor as a DenseTensor&lt;T&gt;</returns>
        public override DenseTensor<T> ToDenseTensor()
        {
            var denseTensor = new DenseTensor<T>(Dimensions, reverseStride: IsReversedStride);
            
            // only set non-zero values
            foreach (var pair in values)
            {
                denseTensor.SetValue(pair.Key, pair.Value);
            }

            return denseTensor;
        }

        /// <summary>
        /// Creates a copy of this tensor as a new SparseTensor&lt;T&gt; eliminating any unused space in the backing storage.
        /// </summary>
        /// <returns>A copy of this tensor as a SparseTensor&lt;T&gt; eliminated any usused space in the backing storage.</returns>
        internal override SparseTensor<T> ToSparseTensor()
        {
            var valueCopy = new Dictionary<int, T>(values);
            return new SparseTensor<T>(valueCopy, dimensions, IsReversedStride);
        }

        /// <summary>
        /// Creates a copy of this tensor as a CompressedSparseTensor&lt;T&gt;.
        /// </summary>
        /// <returns>A copy of this tensor as a CompressedSparseTensor&lt;T&gt;.</returns>
        internal override CompressedSparseTensor<T> ToCompressedSparseTensor()
        {
            var compressedSparseTensor = new CompressedSparseTensor<T>(dimensions, capacity: NonZeroCount, reverseStride: IsReversedStride);

            foreach (var pair in values)
            {
                compressedSparseTensor.SetValue(pair.Key, pair.Value);
            }
            return compressedSparseTensor;
        }
    }
}
