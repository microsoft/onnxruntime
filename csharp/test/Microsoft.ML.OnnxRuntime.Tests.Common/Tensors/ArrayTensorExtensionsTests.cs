// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using Xunit;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.OnnxRuntime.Tests.ArrayTensorExtensions
{
    public class ArrayTensorExtensionsTests
    {
        static void CheckValues(IEnumerable<int> expected, DenseTensor<int> tensor) 
        {
            foreach (var pair in expected.Zip(tensor.Buffer.ToArray(), Tuple.Create))
            {
                Assert.Equal(pair.Item1, pair.Item2);
            }
        }

        [Fact]
        public void ConstructFrom1D()
        {
            var array = new int[] { 1, 2, 3, 4 };
            var tensor = array.ToTensor();

            var expectedDims = new int[] { 4 };
            Assert.Equal(tensor.Length, array.Length);
            Assert.Equal(expectedDims, tensor.Dimensions.ToArray());
            CheckValues(array.Cast<int>(), tensor);
        }

        [Fact]
        public void ConstructFrom2D()
        {
            var array = new int[,] { { 1, 2 } , { 3, 4 } };
            var tensor = array.ToTensor();

            var expectedDims = new int[] { 2, 2 };
            Assert.Equal(tensor.Length, array.Length);
            Assert.Equal(expectedDims, tensor.Dimensions.ToArray());
            CheckValues(array.Cast<int>(), tensor);
        }

        [Fact]
        public void ConstructFrom3D()
        {
            var array = new int[,,] { { { 1, 2 }, { 3, 4 } }, 
                                      { { 5, 6 }, { 7, 8 } } };
            var tensor = array.ToTensor();

            var expectedDims = new int[] { 2, 2, 2 };
            Assert.Equal(tensor.Length, array.Length);
            Assert.Equal(expectedDims, tensor.Dimensions.ToArray());
            CheckValues(array.Cast<int>(), tensor);
        }

        [Fact]
        public void ConstructFrom3DWithDim1()
        {
            var array = new int[,,] { { { 1, 2 } }, 
                                      { { 3, 4 } } };
            var tensor = array.ToTensor();

            var expectedDims = new int[] { 2, 1, 2 };
            Assert.Equal(tensor.Length, array.Length);
            Assert.Equal(expectedDims, tensor.Dimensions.ToArray());
            CheckValues(array.Cast<int>(), tensor);
        }

        [Fact]
        public void ConstructFrom4D()
        {
            var array = new int[,,,] {
                { { { 1, 2 }, { 3, 4 } },
                  { { 5, 6 }, { 7, 8 } } }
            };
            var tensor = array.ToTensor();

            var expectedDims = new int[] { 1, 2, 2, 2 };
            Assert.Equal(tensor.Length, array.Length);
            Assert.Equal(expectedDims, tensor.Dimensions.ToArray());
            CheckValues(array.Cast<int>(), tensor);
        }

        [Fact]
        public void ConstructFrom5D()
        {
            var array = new int[,,,,] {
                { { { { 1, 2 }, { 3, 4 } },
                    { { 5, 6 }, { 7, 8 } } } }
            };

            // 5D requires cast to Array
            Array a = (Array)array;
            var tensor = a.ToTensor<int>();

            var expectedDims = new int[] { 1, 1, 2, 2, 2 };
            Assert.Equal(tensor.Length, array.Length);
            Assert.Equal(expectedDims, tensor.Dimensions.ToArray());
            CheckValues(array.Cast<int>(), tensor);
        }

        [Fact]
        public void TestLongStrides()
        {
            long[] emptyStrides = ShapeUtils.GetStrides(Array.Empty<long>());
            Assert.Empty(emptyStrides);

            long[] negativeDims = { 2, -3, 4, 5 };
            Assert.Throws<ArgumentException>(() => ShapeUtils.GetStrides(negativeDims));

            ReadOnlySpan<long> goodDims = stackalloc long[] { 2, 3, 4, 5 };
            long[] expectedStrides = { 60, 20, 5, 1 };
            Assert.Equal(expectedStrides, ShapeUtils.GetStrides(goodDims));
        }

        [Fact]
        public void TestLongGetIndex()
        {
            ReadOnlySpan<long> dims = stackalloc long[] { 2, 3, 4, 5 };
            long size = ShapeUtils.GetSizeForShape(dims);
            Assert.Equal(120, size);

            ReadOnlySpan<long> strides = ShapeUtils.GetStrides(dims);

            static void IncDims(ReadOnlySpan<long> dims, Span<long> indices)
            {
                for (int i = dims.Length - 1; i >= 0; i--)
                {
                    indices[i]++;
                    if (indices[i] < dims[i])
                        break;
                    indices[i] = 0;
                }
            }

            Span<long> indices = stackalloc long[] { 0, 0, 0, 0 };
            for (long i = 0; i < size; i++)
            {
                long index = ShapeUtils.GetIndex(strides, indices);
                Assert.Equal(i, index);
                IncDims(dims, indices);
            }
        }
    }
}
