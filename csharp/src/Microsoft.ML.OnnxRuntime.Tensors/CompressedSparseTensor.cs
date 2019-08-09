// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is copied and adapted from the following git repository -
// https://github.com/dotnet/corefx
// Commit ID: bdd0814360d4c3a58860919f292a306242f27da1
// Path: /src/System.Numerics.Tensors/src/System/Numerics/Tensors/CompressedSparseTensor.cs
// Original license statement below -

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Diagnostics;
using System.Linq;
using System;

namespace Microsoft.ML.OnnxRuntime.Tensors
{
    /// <summary>
    /// Represents a tensor using compressed sparse format
    /// For a two dimensional tensor this is referred to as compressed sparse row (CSR, CRS, Yale), compressed sparse column (CSC, CCS)
    /// 
    /// In this format, data that is in the same value for the compressed dimension has locality
    /// 
    /// In standard layout of a dense tensor, data with the same value for first dimensions has locality.
    /// As such we'll use reverseStride = false (default) to mean that the first dimension is compressed (CSR)
    /// and reverseStride = true to mean that the last dimension is compressed (CSC)
    /// 
    /// </summary>
    /// <typeparam name="T"></typeparam>
    internal class CompressedSparseTensor<T> : Tensor<T>
    {
        private Memory<T> values;
        private Memory<int> compressedCounts;
        private Memory<int> indices;

        private int nonZeroCount;

        private readonly int[] nonCompressedStrides;
        private readonly int compressedDimension;

        private const int defaultCapacity = 64;

        /// <summary>
        /// Constructs a new CompressedSparseTensor of the specifed dimensions and stride ordering.
        /// </summary>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the CompressedSparseTensor to create.</param>
        /// <param name="reverseStride">False (default) to indicate that the first dimension is most major (farthest apart) and the last dimension is most minor (closest together): akin to row-major in a rank-2 tensor.  True to indicate that the last dimension is most major (farthest apart) and the first dimension is most minor (closest together): akin to column-major in a rank-2 tensor.</param>
        public CompressedSparseTensor(ReadOnlySpan<int> dimensions, bool reverseStride = false) : this(dimensions, defaultCapacity, reverseStride)
        { }

        /// <summary>
        /// Constructs a new CompressedSparseTensor of the specifed dimensions, initial capacity, and stride ordering.
        /// </summary>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the CompressedSparseTensor to create.</param>
        /// <param name="capacity">The number of non-zero values this tensor can store without resizing.</param>
        /// <param name="reverseStride">False (default) to indicate that the first dimension is most major (farthest apart) and the last dimension is most minor (closest together): akin to row-major in a rank-2 tensor.  True to indicate that the last dimension is most major (farthest apart) and the first dimension is most minor (closest together): akin to column-major in a rank-2 tensor.</param>
        public CompressedSparseTensor(ReadOnlySpan<int> dimensions, int capacity, bool reverseStride = false) : base(dimensions, reverseStride)
        {
            nonZeroCount = 0;
            compressedDimension = reverseStride ? Rank - 1 : 0;
            nonCompressedStrides = (int[])strides.Clone();
            nonCompressedStrides[compressedDimension] = 0;
            var compressedDimensionLength = dimensions[compressedDimension];
            compressedCounts = new int[compressedDimensionLength + 1];
            values = new T[capacity];
            indices = new int[capacity];
        }

        /// <summary>
        /// Constructs a new CompressedSparseTensor of the specifed dimensions, wrapping existing backing memory for the contents.
        /// Growing this CompressedSparseTensor will re-allocate the backing memory.
        /// </summary>
        /// <param name="values">Memory storing non-zero values to construct this tensor with.</param>
        /// <param name="compressedCounts">Memory storing the counts of non-zero elements at each index of the compressed dimension.</param>
        /// <param name="indices">Memory storing the linearized index (excluding the compressed dimension) of non-zero elements.</param>
        /// <param name="nonZeroCount">The number of valid entries (eg: non-zero values) in <paramref name="values"/> and <paramref name="indices"/>.</param>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the CompressedSparseTensor to create.</param>
        /// <param name="reverseStride">False (default) to indicate that the first dimension is most major (farthest apart) and the last dimension is most minor (closest together): akin to row-major in a rank-2 tensor.  True to indicate that the last dimension is most major (farthest apart) and the first dimension is most minor (closest together): akin to column-major in a rank-2 tensor.</param>
        public CompressedSparseTensor(Memory<T> values, Memory<int> compressedCounts, Memory<int> indices, int nonZeroCount, ReadOnlySpan<int> dimensions, bool reverseStride = false) : base(dimensions, reverseStride)
        {
            compressedDimension = reverseStride ? Rank - 1 : 0;
            nonCompressedStrides = (int[])strides.Clone();
            nonCompressedStrides[compressedDimension] = 0;
            this.values = values;
            this.compressedCounts = compressedCounts;
            this.indices = indices;
            this.nonZeroCount = nonZeroCount;
        }

        internal CompressedSparseTensor(Array fromArray, bool reverseStride = false) : base(fromArray, reverseStride)
        {
            nonZeroCount = 0;
            compressedDimension = reverseStride ? Rank - 1 : 0;
            nonCompressedStrides = (int[])strides.Clone();
            nonCompressedStrides[compressedDimension] = 0;
            var compressedDimensionLength = dimensions[compressedDimension];
            compressedCounts = new int[compressedDimensionLength + 1];

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
                        var compressedIndex = destIndex / strides[compressedDimension];
                        var nonCompressedIndex = destIndex % strides[compressedDimension];

                        SetAt(item, compressedIndex, nonCompressedIndex);
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
                        var compressedIndex = index / strides[compressedDimension];
                        var nonCompressedIndex = index % strides[compressedDimension];

                        SetAt(item, compressedIndex, nonCompressedIndex);
                    }

                    index++;
                }
            }
        }

        /// <summary>
        /// Obtains the value at the specified indices
        /// </summary>
        /// <param name="indices">A span of integers that represent the indices specifying the position of the element to get.</param>
        /// <returns>The value at the specified position in this Tensor.</returns>
        public override T this[ReadOnlySpan<int> indices]
        {
            get
            {
                var compressedIndex = indices[compressedDimension];
                var nonCompressedIndex = ArrayUtilities.GetIndex(nonCompressedStrides, indices);


                if (TryFindIndex(compressedIndex, nonCompressedIndex, out int valueIndex))
                {
                    return values.Span[valueIndex];
                }

                return Zero;
            }

            set
            {
                var compressedIndex = indices[compressedDimension];
                var nonCompressedIndex = ArrayUtilities.GetIndex(nonCompressedStrides, indices);

                SetAt(value, compressedIndex, nonCompressedIndex);
            }
        }

        /// <summary>
        /// Gets the value at the specied index, where index is lineraized as a dot product between indices and strides.
        /// </summary>
        /// <param name="index">An integer index computed as a dot-product of indices.</param>
        /// <returns>The value at the specified position in this Tensor.</returns>
        public override T GetValue(int index)
        {
            var compressedDimensionStride = strides[compressedDimension];
            Debug.Assert(compressedDimensionStride == strides.Max());

            var compressedIndex = index / compressedDimensionStride;
            var nonCompressedIndex = index % compressedDimensionStride;


            if (TryFindIndex(compressedIndex, nonCompressedIndex, out int valueIndex))
            {
                return values.Span[valueIndex];
            }

            return Zero;
        }

        /// <summary>
        /// Sets the value at the specied index, where index is a linearized version of n-dimension indices using strides.
        /// </summary>
        /// <param name="index">An integer index computed as a dot-product of indices.</param>
        /// <param name="value">The new value to set at the specified position in this Tensor.</param>
        public override void SetValue(int index, T value)
        {
            var compressedDimensionStride = strides[compressedDimension];
            Debug.Assert(compressedDimensionStride == strides.Max());

            var compressedIndex = index / compressedDimensionStride;
            var nonCompressedIndex = index % compressedDimensionStride;

            SetAt(value, compressedIndex, nonCompressedIndex);

        }

        /// <summary>
        /// Gets the number of non-zero values this tensor can store without resizing.
        /// </summary>
        public int Capacity => values.Length;

        /// <summary>
        /// Get's the number on non-zero values currently being stored in this tensor.
        /// </summary>
        public int NonZeroCount => nonZeroCount;

        /// <summary>
        /// Memory storing non-zero values.
        /// </summary>
        public Memory<T> Values => values;

        /// <summary>
        /// Memory storing the counts of non-zero elements at each index of the compressed dimension.
        /// </summary>
        public Memory<int> CompressedCounts => compressedCounts;

        /// <summary>
        /// Memory storing the linearized index (excluding the compressed dimension) of non-zero elements.
        /// </summary>
        public Memory<int> Indices => indices;

        private void EnsureCapacity(int min, int allocateIndex = -1)
        {
            if (values.Length < min)
            {
                var newCapacity = values.Length == 0 ? defaultCapacity : values.Length * 2;

                if (newCapacity > Length)
                {
                    newCapacity = (int)Length;
                }

                if (newCapacity < min)
                {
                    newCapacity = min;
                }

                Memory<T> newValues = new T[newCapacity];
                Memory<int> newIndices = new int[newCapacity];

                if (nonZeroCount > 0)
                {
                    if (allocateIndex == -1)
                    {
                        var valuesSpan = values.Span.Slice(0, nonZeroCount);
                        var indicesSpan = indices.Span.Slice(0, nonZeroCount);

                        valuesSpan.CopyTo(newValues.Span);
                        indicesSpan.CopyTo(newIndices.Span);
                    }
                    else
                    {
                        Debug.Assert(allocateIndex <= nonZeroCount);
                        // leave a gap at allocateIndex

                        // copy range before allocateIndex
                        if (allocateIndex > 0)
                        {
                            var valuesSpan = values.Span.Slice(0, allocateIndex);
                            var indicesSpan = indices.Span.Slice(0, allocateIndex);

                            valuesSpan.CopyTo(newValues.Span);
                            indicesSpan.CopyTo(newIndices.Span);
                        }

                        if (allocateIndex < nonZeroCount)
                        {
                            var valuesSpan = values.Span.Slice(allocateIndex, nonZeroCount - allocateIndex);
                            var indicesSpan = indices.Span.Slice(allocateIndex, nonZeroCount - allocateIndex);

                            var newValuesSpan = newValues.Span.Slice(allocateIndex + 1, nonZeroCount - allocateIndex);
                            var newIndicesSpan = newIndices.Span.Slice(allocateIndex + 1, nonZeroCount - allocateIndex);

                            valuesSpan.CopyTo(newValuesSpan);
                            indicesSpan.CopyTo(newIndicesSpan);
                        }
                    }
                }

                values = newValues;
                indices = newIndices;
            }
        }

        private void InsertAt(int valueIndex, T value, int compressedIndex, int nonCompressedIndex)
        {
            Debug.Assert(valueIndex <= nonZeroCount);
            Debug.Assert(compressedIndex < compressedCounts.Length - 1);

            if (values.Length <= valueIndex)
            {
                // allocate a new array, leaving a gap
                EnsureCapacity(valueIndex + 1, valueIndex);
            }
            else if (nonZeroCount != valueIndex)
            {
                // shift values to make a gap
                values.Span.Slice(valueIndex, nonZeroCount - valueIndex).CopyTo(values.Span.Slice(valueIndex + 1));
                indices.Span.Slice(valueIndex, nonZeroCount - valueIndex).CopyTo(indices.Span.Slice(valueIndex + 1));
            }

            values.Span[valueIndex] = value;
            indices.Span[valueIndex] = nonCompressedIndex;

            var compressedCountsSpan = compressedCounts.Span.Slice(compressedIndex + 1);
            for (int i = 0; i < compressedCountsSpan.Length; i++)
            {
                compressedCountsSpan[i]++;
            }
            nonZeroCount++;
        }

        private void RemoveAt(int valueIndex, int compressedIndex)
        {
            Debug.Assert(valueIndex < nonZeroCount);
            Debug.Assert(compressedIndex < compressedCounts.Length - 1);

            // shift values to close the gap
            values.Span.Slice(valueIndex + 1, nonZeroCount - valueIndex - 1).CopyTo(values.Span.Slice(valueIndex));
            indices.Span.Slice(valueIndex + 1, nonZeroCount - valueIndex - 1).CopyTo(indices.Span.Slice(valueIndex));

            var compressedCountsSpan = compressedCounts.Span.Slice(compressedIndex + 1);
            for (int i = 0; i < compressedCountsSpan.Length; i++)
            {
                compressedCountsSpan[i]--;
            }
            nonZeroCount--;
        }

        private void SetAt(T value, int compressedIndex, int nonCompressedIndex)
        {
            bool isZero = value.Equals(Zero);

            if (TryFindIndex(compressedIndex, nonCompressedIndex, out int valueIndex))
            {
                if (isZero)
                {
                    RemoveAt(valueIndex, compressedIndex);
                }
                else
                {
                    values.Span[valueIndex] = value;
                    indices.Span[valueIndex] = nonCompressedIndex;
                }
            }
            else if (!isZero)
            {
                InsertAt(valueIndex, value, compressedIndex, nonCompressedIndex);
            }
        }

        /// <summary>
        /// Trys to find the place to store a value
        /// </summary>
        /// <param name="compressedIndex"></param>
        /// <param name="nonCompressedIndex"></param>
        /// <param name="valueIndex"></param>
        /// <returns>True if element is found at specific index, false if no specific index is found and insertion point is returned</returns>
        private bool TryFindIndex(int compressedIndex, int nonCompressedIndex, out int valueIndex)
        {
            if (nonZeroCount == 0)
            {
                valueIndex = 0;
                return false;
            }

            Debug.Assert(compressedIndex < compressedCounts.Length - 1);

            var compressedCountsSpan = compressedCounts.Span;
            var lowerValueIndex = compressedCountsSpan[compressedIndex];
            var upperValueIndex = compressedCountsSpan[compressedIndex + 1];
            var indicesSpan = indices.Span;

            // could be a faster search
            for (valueIndex = lowerValueIndex; valueIndex < upperValueIndex; valueIndex++)
            {
                if (indicesSpan[valueIndex] == nonCompressedIndex)
                {
                    return true;
                }
            }

            return false;
        }

        /// <summary>
        /// Creates a shallow copy of this tensor, with new backing storage.
        /// </summary>
        /// <returns>A shallow copy of this tensor.</returns>
        public override Tensor<T> Clone()
        {
            return new CompressedSparseTensor<T>(values.ToArray(), compressedCounts.ToArray(), indices.ToArray(), nonZeroCount, dimensions, IsReversedStride);
        }

        /// <summary>
        /// Creates a new Tensor of a different type with the specified dimensions and the same layout as this tensor with elements initialized to their default value.
        /// </summary>
        /// <typeparam name="TResult">Type contained in the returned Tensor.</typeparam>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the CompressedSparseTensor to create.</param>
        /// <returns>A new tensor with the same layout as this tensor but different type and dimensions.</returns>
        public override Tensor<TResult> CloneEmpty<TResult>(ReadOnlySpan<int> dimensions)
        {
            return new CompressedSparseTensor<TResult>(dimensions, IsReversedStride);
        }

        /// <summary>
        /// Reshapes the current tensor to new dimensions. Unlike other Tensor implementations, CompressedSparseTensor&lt;T&gt; must allocate new backing storage to represent a reshaped Tensor.
        /// </summary>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the CompressedSparseTensor to create.</param>
        /// <returns>A new tensor that reinterprets the content of this tensor to new dimensions (assuming the same linear index for each element).</returns>
        public override Tensor<T> Reshape(ReadOnlySpan<int> dimensions)
        {
            // reshape currently has shallow semantics which are not compatible with the backing storage for CompressedSparseTensor
            // which bakes in information about dimensions (compressedCounts and indices)

            var newCompressedDimension = IsReversedStride ? dimensions.Length - 1 : 0;
            var newCompressedDimensionLength = dimensions[newCompressedDimension];
            var newCompressedDimensionStride = (int)(Length / newCompressedDimensionLength);
            
            var newValues = (T[])values.ToArray();
            var newCompressedCounts = new int[newCompressedDimensionLength + 1];
            var newIndices = new int[indices.Length];

            var compressedIndex = 0;

            var compressedCountsSpan = compressedCounts.Span;
            var indicesSpan = indices.Span.Slice(0, nonZeroCount);
            for (int valueIndex = 0; valueIndex < indicesSpan.Length; valueIndex++)
            {
                while (valueIndex >= compressedCountsSpan[compressedIndex + 1])
                {
                    compressedIndex++;
                    Debug.Assert(compressedIndex < compressedCounts.Length);
                }

                var currentIndex = indicesSpan[valueIndex] + compressedIndex * strides[compressedDimension];

                newIndices[valueIndex] = currentIndex % newCompressedDimensionStride;

                var newCompressedIndex = currentIndex / newCompressedDimensionStride;
                newCompressedCounts[newCompressedIndex + 1] = valueIndex + 1;
            }

            return new CompressedSparseTensor<T>(newValues, newCompressedCounts, newIndices, nonZeroCount, dimensions, IsReversedStride);
        }

        /// <summary>
        /// Creates a copy of this tensor as a DenseTensor&lt;T&gt;.
        /// </summary>
        /// <returns>A copy of this tensor as a DenseTensor&lt;T&gt;</returns>
        public override DenseTensor<T> ToDenseTensor()
        {
            var denseTensor = new DenseTensor<T>(Dimensions, reverseStride: IsReversedStride);

            var compressedIndex = 0;

            var compressedCountsSpan = compressedCounts.Span;
            var indicesSpan = indices.Span.Slice(0, nonZeroCount);
            var valuesSpan = values.Span.Slice(0, nonZeroCount);
            for (int valueIndex = 0; valueIndex < valuesSpan.Length; valueIndex++)
            {
                while (valueIndex >= compressedCountsSpan[compressedIndex + 1])
                {
                    compressedIndex++;
                    Debug.Assert(compressedIndex < compressedCounts.Length);
                }

                var index = indicesSpan[valueIndex] + compressedIndex * strides[compressedDimension];

                denseTensor.SetValue(index, valuesSpan[valueIndex]);
            }

            return denseTensor;
        }

        /// <summary>
        /// Creates a copy of this tensor as a new CompressedSparseTensor&lt;T&gt; eliminating any unused space in the backing storage.
        /// </summary>
        /// <returns>A copy of this tensor as a CompressedSparseTensor&lt;T&gt;.</returns>
        internal override CompressedSparseTensor<T> ToCompressedSparseTensor()
        {
            // Create a copy of the backing storage, eliminating any unused space.
            var newValues = values.Slice(0, nonZeroCount).ToArray();
            var newIndicies = indices.Slice(0, nonZeroCount).ToArray();

            return new CompressedSparseTensor<T>(newValues, compressedCounts.ToArray(), newIndicies, nonZeroCount, dimensions, IsReversedStride);
        }

        /// <summary>
        /// Creates a copy of this tensor as a SparseTensor&lt;T&gt;. 
        /// </summary>
        /// <returns>A copy of this tensor as a SparseTensor&lt;T&gt;.</returns>
        internal override SparseTensor<T> ToSparseTensor()
        {
            var sparseTensor = new SparseTensor<T>(dimensions, capacity: NonZeroCount, reverseStride: IsReversedStride);

            var compressedIndex = 0;

            var compressedCountsSpan = compressedCounts.Span;
            var indicesSpan = indices.Span.Slice(0, nonZeroCount);
            var valuesSpan = values.Span.Slice(0, nonZeroCount);
            for (int valueIndex = 0; valueIndex < valuesSpan.Length; valueIndex++)
            {
                while (valueIndex >= compressedCountsSpan[compressedIndex + 1])
                {
                    compressedIndex++;
                    Debug.Assert(compressedIndex < compressedCounts.Length);
                }

                var index = indicesSpan[valueIndex] + compressedIndex * strides[compressedDimension];
                
                sparseTensor.SetValue(index, valuesSpan[valueIndex]);
            }

            return sparseTensor;
        }
    }
}
