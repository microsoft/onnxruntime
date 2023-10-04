// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is copied and adapted from the following git repository -
// https://github.com/dotnet/corefx
// Commit ID: bdd0814360d4c3a58860919f292a306242f27da1
// Path: /src/System.Numerics.Tensors/tests/TensorTests.cs
// Original license statement below -

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace Microsoft.ML.OnnxRuntime.Tensors.Tests
{
    public class TensorTests : TensorTestsBase
    {
        [Theory(DisplayName = "ConstructTensorFromArrayRank1")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void ConstructTensorFromArrayRank1(TensorConstructor tensorConstructor)
        {
            var tensor = tensorConstructor.CreateFromArray<int>(new[] { 0, 1, 2 });

            Assert.Equal(tensorConstructor.IsReversedStride, tensor.IsReversedStride);
            Assert.Equal(0, tensor[0]);
            Assert.Equal(1, tensor[1]);
            Assert.Equal(2, tensor[2]);
        }

        [Theory(DisplayName = "ConstructTensorFromArrayRank2")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void ConstructTensorFromArrayRank2(TensorConstructor tensorConstructor)
        {
            var tensor = tensorConstructor.CreateFromArray<int>(new[,]
            {
                {0, 1, 2},
                {3, 4, 5}
            });

            Assert.Equal(tensorConstructor.IsReversedStride, tensor.IsReversedStride);
            Assert.Equal(0, tensor[0, 0]);
            Assert.Equal(1, tensor[0, 1]);
            Assert.Equal(2, tensor[0, 2]);
            Assert.Equal(3, tensor[1, 0]);
            Assert.Equal(4, tensor[1, 1]);
            Assert.Equal(5, tensor[1, 2]);
        }

        [Theory(DisplayName = "ConstructTensorFromArrayRank3")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void ConstructTensorFromArrayRank3(TensorConstructor tensorConstructor)
        {
            var tensor = tensorConstructor.CreateFromArray<int>(new[, ,]
            {
                {
                    {0, 1, 2},
                    {3, 4, 5}
                },
                {
                    {6, 7 ,8 },
                    {9, 10 ,11 },
                },
                {
                    {12, 13 ,14 },
                    {15, 16 ,17 },
                },
                {
                    {18, 19 ,20 },
                    {21, 22 ,23 },
                }
            });

            Assert.Equal(tensorConstructor.IsReversedStride, tensor.IsReversedStride);

            Assert.Equal(0, tensor[0, 0, 0]);
            Assert.Equal(1, tensor[0, 0, 1]);
            Assert.Equal(2, tensor[0, 0, 2]);
            Assert.Equal(3, tensor[0, 1, 0]);
            Assert.Equal(4, tensor[0, 1, 1]);
            Assert.Equal(5, tensor[0, 1, 2]);

            Assert.Equal(6, tensor[1, 0, 0]);
            Assert.Equal(7, tensor[1, 0, 1]);
            Assert.Equal(8, tensor[1, 0, 2]);
            Assert.Equal(9, tensor[1, 1, 0]);
            Assert.Equal(10, tensor[1, 1, 1]);
            Assert.Equal(11, tensor[1, 1, 2]);

            Assert.Equal(12, tensor[2, 0, 0]);
            Assert.Equal(13, tensor[2, 0, 1]);
            Assert.Equal(14, tensor[2, 0, 2]);
            Assert.Equal(15, tensor[2, 1, 0]);
            Assert.Equal(16, tensor[2, 1, 1]);
            Assert.Equal(17, tensor[2, 1, 2]);

            Assert.Equal(18, tensor[3, 0, 0]);
            Assert.Equal(19, tensor[3, 0, 1]);
            Assert.Equal(20, tensor[3, 0, 2]);
            Assert.Equal(21, tensor[3, 1, 0]);
            Assert.Equal(22, tensor[3, 1, 1]);
            Assert.Equal(23, tensor[3, 1, 2]);
        }

        [Fact(DisplayName = "ConstructDenseTensorFromPointer")]
        public void ConstructDenseTensorFromPointer()
        {
            using (var nativeMemory = NativeMemoryFromArray(Enumerable.Range(0, 24).ToArray()))
            {
                var dimensions = new[] { 4, 2, 3 };
                var tensor = new DenseTensor<int>(nativeMemory.Memory, dimensions, false);

                Assert.Equal(0, tensor[0, 0, 0]);
                Assert.Equal(1, tensor[0, 0, 1]);
                Assert.Equal(2, tensor[0, 0, 2]);
                Assert.Equal(3, tensor[0, 1, 0]);
                Assert.Equal(4, tensor[0, 1, 1]);
                Assert.Equal(5, tensor[0, 1, 2]);

                Assert.Equal(6, tensor[1, 0, 0]);
                Assert.Equal(7, tensor[1, 0, 1]);
                Assert.Equal(8, tensor[1, 0, 2]);
                Assert.Equal(9, tensor[1, 1, 0]);
                Assert.Equal(10, tensor[1, 1, 1]);
                Assert.Equal(11, tensor[1, 1, 2]);

                Assert.Equal(12, tensor[2, 0, 0]);
                Assert.Equal(13, tensor[2, 0, 1]);
                Assert.Equal(14, tensor[2, 0, 2]);
                Assert.Equal(15, tensor[2, 1, 0]);
                Assert.Equal(16, tensor[2, 1, 1]);
                Assert.Equal(17, tensor[2, 1, 2]);

                Assert.Equal(18, tensor[3, 0, 0]);
                Assert.Equal(19, tensor[3, 0, 1]);
                Assert.Equal(20, tensor[3, 0, 2]);
                Assert.Equal(21, tensor[3, 1, 0]);
                Assert.Equal(22, tensor[3, 1, 1]);
                Assert.Equal(23, tensor[3, 1, 2]);
            }
        }


        [Theory(DisplayName = "ConstructFromDimensions")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void ConstructFromDimensions(TensorConstructor tensorConstructor)
        {
            var tensor = tensorConstructor.CreateFromDimensions<int>(new[] { 2, 3, 4 });
            Assert.Equal(3, tensor.Rank);
            Assert.Equal(3, tensor.Dimensions.Length);
            Assert.Equal(2, tensor.Dimensions[0]);
            Assert.Equal(3, tensor.Dimensions[1]);
            Assert.Equal(4, tensor.Dimensions[2]);
            Assert.Equal(24, tensor.Length);
            Assert.Equal(tensorConstructor.IsReversedStride, tensor.IsReversedStride);

            // The null is converted to a 'null' ReadOnlySpan<T> which is a valid instance with length zero.
            // https://learn.microsoft.com/en-us/dotnet/api/system.readonlyspan-1.-ctor?view=netstandard-2.1#system-readonlyspan-1-ctor(-0())
            // Such a span is valid as dimensions as it represents a scalar. 
            // If we need to differentiate between the two we'd need to change the Tensor ctor to accept a
            // nullable span in order to tell the user that's invalid. 
            // Assert.Throws<ArgumentNullException>("dimensions", () => tensorConstructor.CreateFromDimensions<int>(dimensions: null));
            Assert.Throws<ArgumentOutOfRangeException>("dimensions", () => tensorConstructor.CreateFromDimensions<int>(dimensions: new[] { 1, -1 }));

            // ensure dimensions are immutable
            var dimensions = new[] { 1, 2, 3 };
            tensor = tensorConstructor.CreateFromDimensions<int>(dimensions: dimensions);
            dimensions[0] = dimensions[1] = dimensions[2] = 0;
            Assert.Equal(1, tensor.Dimensions[0]);
            Assert.Equal(2, tensor.Dimensions[1]);
            Assert.Equal(3, tensor.Dimensions[2]);
        }

        [Theory(DisplayName = "ConstructEmptyTensors")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void ConstructEmptyTensors(TensorConstructor tensorConstructor)
        {
            // tests associated with empty tensors (i.e.) tensors with empty content

            // test creation of empty tensors (from dimensions)
            var dimensions = new[] { 1, 0 };
            var emptyTensor1 = tensorConstructor.CreateFromDimensions<int>(dimensions: dimensions);
            dimensions[0] = dimensions[1] = 0;
            Assert.Equal(1, emptyTensor1.Dimensions[0]);
            Assert.Equal(0, emptyTensor1.Dimensions[1]);
            Assert.Equal(2, emptyTensor1.Rank);

            dimensions = new[] { 0, 2, 4};
            var emptyTensor2 = tensorConstructor.CreateFromDimensions<int>(dimensions: dimensions);
            dimensions[0] = dimensions[1] = dimensions[2] = 0;
            Assert.Equal(0, emptyTensor2.Dimensions[0]);
            Assert.Equal(2, emptyTensor2.Dimensions[1]);
            Assert.Equal(4, emptyTensor2.Dimensions[2]);
            Assert.Equal(3, emptyTensor2.Rank);

            // test creation of empty tensors (from an empty array)
            // by default it will create an empty tensor of shape: [0] (1D)
            var emptyTensor3 = tensorConstructor.CreateFromArray<int>(new int[] { });
            Assert.Equal(0, emptyTensor3.Dimensions[0]);
            Assert.Equal(1, emptyTensor3.Rank);
            
            // ensure the lengths of the empty tensors are 0
            Assert.Equal(0, emptyTensor1.Length);
            Assert.Equal(0, emptyTensor2.Length);
            Assert.Equal(0, emptyTensor3.Length);

            // equality comparison of tensors with a dimension value of '0' along different axes throws an ArgumentException
            Assert.Throws<ArgumentException>("other", () => StructuralComparisons.StructuralComparer.Compare(emptyTensor1, emptyTensor2));
            Assert.Throws<ArgumentException>("other", () => StructuralComparisons.StructuralComparer.Compare(emptyTensor2, emptyTensor1));
            Assert.Throws<ArgumentException>("other", () => StructuralComparisons.StructuralEqualityComparer.Equals(emptyTensor1, emptyTensor2));
            Assert.Throws<ArgumentException>("other", () => StructuralComparisons.StructuralEqualityComparer.Equals(emptyTensor2, emptyTensor1));

            // equality comparison of tensors with a dimension value of '0' along same axis is true
            // equality comparison of an empty tensor with itself
            Assert.Equal(0, StructuralComparisons.StructuralComparer.Compare(emptyTensor1, emptyTensor1));
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(emptyTensor1, emptyTensor1));
            // equality comparison of an empty tensor with another empty tensor with same dimensions as 'emptyTensor1'
            dimensions = new[] { 1, 0 };
            var emptyTensor4 = tensorConstructor.CreateFromDimensions<int>(dimensions: dimensions);
            Assert.Equal(0, StructuralComparisons.StructuralComparer.Compare(emptyTensor1, emptyTensor4));
            Assert.Equal(0, StructuralComparisons.StructuralComparer.Compare(emptyTensor4, emptyTensor1));
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(emptyTensor1, emptyTensor4));
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(emptyTensor4, emptyTensor1));

            // create an empty DenseTensor from dimensions
            dimensions = new[] { 0, 2, 1 };
            var emptyDenseTensor1 = new DenseTensor<int>(new Span<int>(dimensions));
            Assert.Equal(0, emptyDenseTensor1.Length);
            // accessing any index in the underlying buffer should result in an IndexOutOfRangeException
            Assert.Throws<IndexOutOfRangeException>(() => emptyDenseTensor1.GetValue(0));
            Assert.Throws<IndexOutOfRangeException>(() => emptyDenseTensor1.GetValue(5));

            // create an empty DenseTensor from memory
            var memory = new Memory<int>(new int[] { });
            var emptyDenseTensor2 = new DenseTensor<int>(memory, new int[] {2, 0, 2 });
            Assert.Equal(0, emptyDenseTensor2.Length);
            // accessing any index in the underlying buffer should result in an IndexOutOfRangeException
            Assert.Throws<IndexOutOfRangeException>(() => emptyDenseTensor2.GetValue(0));
            Assert.Throws<IndexOutOfRangeException>(() => emptyDenseTensor2.GetValue(5));

        }

        [Theory(DisplayName = "ConstructScalarTensors")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void ConstructScalarTensors(TensorConstructor tensorConstructor)
        {
            // tests associated with scalar tensors (i.e.) tensors with no dimensions

            // test creation of scalar tensors (from dimensions)
            var dimensions = new int[] { };
            var scalarTensor1 = tensorConstructor.CreateFromDimensions<int>(dimensions: dimensions);
            Assert.Equal(0, scalarTensor1.Dimensions.Length);
            Assert.Equal(0, scalarTensor1.Rank);

            dimensions = new int[0];
            var scalarTensor2 = tensorConstructor.CreateFromDimensions<int>(dimensions: dimensions);
            Assert.Equal(0, scalarTensor2.Dimensions.Length);
            Assert.Equal(0, scalarTensor2.Rank);

            // TODO: Create from Array ?

            // ensure the lengths of the scalar tensors is 1
            Assert.Equal(1, scalarTensor1.Length);
            Assert.Equal(1, scalarTensor2.Length);

            // equality comparison of scalar tensors with non-scalar tensors throws an ArgumentException
            dimensions = new int[] { 2, 2 };
            var nonScalarTensor1 = tensorConstructor.CreateFromDimensions<int>(dimensions: dimensions);

            dimensions = new int[] { 0, 2, 4 };
            var nonScalarTensor2 = tensorConstructor.CreateFromDimensions<int>(dimensions: dimensions);

            Assert.Throws<ArgumentException>("other", () => StructuralComparisons.StructuralComparer.Compare(nonScalarTensor1, scalarTensor1));
            Assert.Throws<ArgumentException>("other", () => StructuralComparisons.StructuralComparer.Compare(scalarTensor1, nonScalarTensor1));
            Assert.Throws<ArgumentException>("other", () => StructuralComparisons.StructuralComparer.Compare(nonScalarTensor2, scalarTensor1));
            Assert.Throws<ArgumentException>("other", () => StructuralComparisons.StructuralComparer.Compare(scalarTensor1, nonScalarTensor2));

            // equality comparison of scalar tensors is true
            // equality comparison of a scalar tensor with itself
            Assert.Equal(0, StructuralComparisons.StructuralComparer.Compare(scalarTensor1, scalarTensor1));
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(scalarTensor1, scalarTensor1));
            // equality comparison of a scalar tensor with another scalar tensor
            Assert.Equal(0, StructuralComparisons.StructuralComparer.Compare(scalarTensor1, scalarTensor2));
            Assert.Equal(0, StructuralComparisons.StructuralComparer.Compare(scalarTensor2, scalarTensor1));
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(scalarTensor1, scalarTensor2));
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(scalarTensor2, scalarTensor1));

            // create a scalar DenseTensor from dimensions
            dimensions = new int[] { };
            var scalarDenseTensor1 = new DenseTensor<int>(new Span<int>(dimensions));
            Assert.Equal(1, scalarDenseTensor1.Length);
            // set and get values in the scalar tensor
            scalarDenseTensor1.SetValue(0, 100);
            Assert.Equal(100, scalarDenseTensor1.GetValue(0));

            // setting a non-zero in the underlying buffer should result in an IndexOutOfRangeException
            Assert.Throws<IndexOutOfRangeException>(() => scalarDenseTensor1.SetValue(6, 100));

            // accessing a non-zero index in the underlying buffer should result in an IndexOutOfRangeException
            Assert.Throws<IndexOutOfRangeException>(() => scalarDenseTensor1.GetValue(5));

            // create a scalar DenseTensor from memory
            var memory = new Memory<int>(new int[] { 1 });
            var scalarDenseTensor2 = new DenseTensor<int>(memory, new int[] { });
            Assert.Equal(1, scalarDenseTensor2.Length);
            Assert.Equal(1, scalarDenseTensor2.GetValue(0));
        }

        [Theory(DisplayName = "ConstructTensorFromArrayRank3WithLowerBounds")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void ConstructTensorFromArrayRank3WithLowerBounds(TensorConstructor tensorConstructor)
        {
            var dimensions = new[] { 2, 3, 4 };
            var lowerBounds = new[] { 0, 5, 200 };
            var arrayWithLowerBounds = Array.CreateInstance(typeof(int), dimensions, lowerBounds);

            int value = 0;
            for (int x = lowerBounds[0]; x < lowerBounds[0] + dimensions[0]; x++)
            {
                for (int y = lowerBounds[1]; y < lowerBounds[1] + dimensions[1]; y++)
                {
                    for (int z = lowerBounds[2]; z < lowerBounds[2] + dimensions[2]; z++)
                    {
                        arrayWithLowerBounds.SetValue(value++, x, y, z);
                    }
                }
            }

            var tensor = tensorConstructor.CreateFromArray<int>(arrayWithLowerBounds);

            var expected = tensorConstructor.CreateFromArray<int>(new[, ,]
                    {
                        {
                            { 0, 1, 2, 3 },
                            { 4, 5, 6, 7 },
                            { 8, 9, 10, 11 }
                        },
                        {
                            { 12, 13, 14, 15 },
                            { 16, 17, 18, 19 },
                            { 20, 21, 22, 23 }
                        }
                    }
                );
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(expected, tensor));
            Assert.Equal(tensorConstructor.IsReversedStride, tensor.IsReversedStride);
        }

        [Theory(DisplayName = "StructurallyEqualTensor")]
        [MemberData(nameof(GetDualTensorConstructors))]
        public void StructurallyEqualTensor(TensorConstructor leftConstructor, TensorConstructor rightConstructor)
        {
            var arr = new[, ,]
            {
                {
                    {0, 1, 2},
                    {3, 4, 5}
                },
                {
                    {6, 7 ,8 },
                    {9, 10 ,11 },
                },
                {
                    {12, 13 ,14 },
                    {15, 16 ,17 },
                },
                {
                    {18, 19 ,20 },
                    {21, 22 ,23 },
                }
            };
            var tensor = leftConstructor.CreateFromArray<int>(arr);
            var tensor2 = rightConstructor.CreateFromArray<int>(arr);

            Assert.Equal(0, StructuralComparisons.StructuralComparer.Compare(tensor, tensor2));
            Assert.Equal(0, StructuralComparisons.StructuralComparer.Compare(tensor2, tensor));
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tensor, tensor2));
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tensor2, tensor));
            // Issue: should Tensors with different layout be structurally equal?
            if (leftConstructor.IsReversedStride == leftConstructor.IsReversedStride)
            {
                Assert.Equal(StructuralComparisons.StructuralEqualityComparer.GetHashCode(tensor), StructuralComparisons.StructuralEqualityComparer.GetHashCode(tensor2));
            }
        }

        [Theory(DisplayName = "StructurallyEqualArray")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void StructurallyEqualArray(TensorConstructor tensorConstructor)
        {
            var arr = new[, ,]
            {
                {
                    {0, 1, 2},
                    {3, 4, 5}
                },
                {
                    {6, 7 ,8 },
                    {9, 10 ,11 },
                },
                {
                    {12, 13 ,14 },
                    {15, 16 ,17 },
                },
                {
                    {18, 19 ,20 },
                    {21, 22 ,23 },
                }
            };
            var tensor = tensorConstructor.CreateFromArray<int>(arr);

            Assert.Equal(0, StructuralComparisons.StructuralComparer.Compare(tensor, arr));
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tensor, arr));

        }

        [Theory(DisplayName = "GetDiagonalSquare")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void GetDiagonalSquare(TensorConstructor tensorConstructor)
        {
            var arr = new[,]
            {
               { 1, 2, 4 },
               { 8, 3, 9 },
               { 1, 7, 5 },
            };

            var tensor = tensorConstructor.CreateFromArray<int>(arr);
            var diag = tensor.GetDiagonal();
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(diag, new[] { 1, 3, 5 }));
            diag = tensor.GetDiagonal(1);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(diag, new[] { 2, 9 }));
            diag = tensor.GetDiagonal(2);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(diag, new[] { 4 }));
            Assert.Throws<ArgumentException>("offset", () => tensor.GetDiagonal(3));

            diag = tensor.GetDiagonal(-1);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(diag, new[] { 8, 7 }));
            diag = tensor.GetDiagonal(-2);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(diag, new[] { 1 }));
            Assert.Throws<ArgumentException>("offset", () => tensor.GetDiagonal(-3));
        }

        [Theory(DisplayName = "GetDiagonalRectangle")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void GetDiagonalRectangle(TensorConstructor tensorConstructor)
        {
            var arr = new[,]
            {
               { 1, 2, 4, 3, 7 },
               { 8, 3, 9, 2, 6 },
               { 1, 7, 5, 2, 9 }
            };

            var tensor = tensorConstructor.CreateFromArray<int>(arr);
            var diag = tensor.GetDiagonal();
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(diag, new[] { 1, 3, 5 }));
            diag = tensor.GetDiagonal(1);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(diag, new[] { 2, 9, 2 }));
            diag = tensor.GetDiagonal(2);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(diag, new[] { 4, 2, 9 }));
            diag = tensor.GetDiagonal(3);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(diag, new[] { 3, 6 }));
            diag = tensor.GetDiagonal(4);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(diag, new[] { 7 }));
            Assert.Throws<ArgumentException>("offset", () => tensor.GetDiagonal(5));

            diag = tensor.GetDiagonal(-1);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(diag, new[] { 8, 7 }));
            diag = tensor.GetDiagonal(-2);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(diag, new[] { 1 }));
            Assert.Throws<ArgumentException>("offset", () => tensor.GetDiagonal(-3));
            Assert.Throws<ArgumentException>("offset", () => tensor.GetDiagonal(-4));
            Assert.Throws<ArgumentException>("offset", () => tensor.GetDiagonal(-5));
        }


        [Theory(DisplayName = "GetDiagonalCube")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void GetDiagonalCube(TensorConstructor tensorConstructor)
        {
            var arr = new[, ,]
            {
                {
                   { 1, 2, 4 },
                   { 8, 3, 9 },
                   { 1, 7, 5 },
                },
                {
                   { 4, 5, 7 },
                   { 1, 6, 2 },
                   { 3, 0, 8 },
                },
                {
                   { 5, 6, 1 },
                   { 2, 2, 3 },
                   { 4, 9, 4 },
                },

            };

            var tensor = tensorConstructor.CreateFromArray<int>(arr);
            var diag = tensor.GetDiagonal();
            var expected = new[,]
            {
                { 1, 2, 4 },
                { 1, 6, 2 },
                { 4, 9, 4 }
            };
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(diag, expected));
            Assert.Equal(tensorConstructor.IsReversedStride, diag.IsReversedStride);
        }

        [Theory(DisplayName = "GetTriangleSquare")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void GetTriangleSquare(TensorConstructor tensorConstructor)
        {
            var arr = new[,]
            {
               { 1, 2, 4 },
               { 8, 3, 9 },
               { 1, 7, 5 },
            };

            var tensor = tensorConstructor.CreateFromArray<int>(arr);
            var tri = tensor.GetTriangle(0);
            Assert.Equal(tensorConstructor.IsReversedStride, tri.IsReversedStride);

            var expected = tensorConstructor.CreateFromArray<int>(new[,]
            {
               { 1, 0, 0 },
               { 8, 3, 0 },
               { 1, 7, 5 },
            });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));
            tri = tensor.GetTriangle(1);
            expected = tensorConstructor.CreateFromArray<int>(new[,]
            {
               { 1, 2, 0 },
               { 8, 3, 9 },
               { 1, 7, 5 },
            });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));
            tri = tensor.GetTriangle(2);
            expected = tensorConstructor.CreateFromArray<int>(new[,]
            {
               { 1, 2, 4 },
               { 8, 3, 9 },
               { 1, 7, 5 },
            });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));

            tri = tensor.GetTriangle(3);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));

            tri = tensor.GetTriangle(200);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));

            tri = tensor.GetTriangle(-1);
            expected = tensorConstructor.CreateFromArray<int>(new[,]
            {
               { 0, 0, 0 },
               { 8, 0, 0 },
               { 1, 7, 0 },
            });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));
            tri = tensor.GetTriangle(-2);
            expected = tensorConstructor.CreateFromArray<int>(new[,]
            {
               { 0, 0, 0 },
               { 0, 0, 0 },
               { 1, 0, 0 },
            });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));


            expected = tensorConstructor.CreateFromArray<int>(new[,]
            {
               { 0, 0, 0 },
               { 0, 0, 0 },
               { 0, 0, 0 },
            });
            tri = tensor.GetTriangle(-3);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));

            // same as -3, should it be an exception?
            tri = tensor.GetTriangle(-4);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));
            tri = tensor.GetTriangle(-300);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));
        }

        [Theory(DisplayName = "GetTriangleRectangle")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void GetTriangleRectangle(TensorConstructor tensorConstructor)
        {
            var arr = new[,]
            {
               { 1, 2, 4, 3, 7 },
               { 8, 3, 9, 2, 6 },
               { 1, 7, 5, 2, 9 }
            };

            var tensor = tensorConstructor.CreateFromArray<int>(arr);
            var tri = tensor.GetTriangle(0);
            var expected = tensorConstructor.CreateFromArray<int>(new[,]
            {
               { 1, 0, 0, 0, 0 },
               { 8, 3, 0, 0, 0 },
               { 1, 7, 5, 0, 0 }
            });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));
            Assert.Equal(tensorConstructor.IsReversedStride, tri.IsReversedStride);

            tri = tensor.GetTriangle(1);
            expected = tensorConstructor.CreateFromArray<int>(new[,]
            {
               { 1, 2, 0, 0, 0 },
               { 8, 3, 9, 0, 0 },
               { 1, 7, 5, 2, 0 }
            });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));
            tri = tensor.GetTriangle(2);
            expected = tensorConstructor.CreateFromArray<int>(new[,]
            {
               { 1, 2, 4, 0, 0 },
               { 8, 3, 9, 2, 0 },
               { 1, 7, 5, 2, 9 }
            });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));
            tri = tensor.GetTriangle(3);
            expected = tensorConstructor.CreateFromArray<int>(new[,]
            {
               { 1, 2, 4, 3, 0 },
               { 8, 3, 9, 2, 6 },
               { 1, 7, 5, 2, 9 }
            });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));

            tri = tensor.GetTriangle(4);
            expected = tensorConstructor.CreateFromArray<int>(new[,]
            {
               { 1, 2, 4, 3, 7 },
               { 8, 3, 9, 2, 6 },
               { 1, 7, 5, 2, 9 }
            });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));

            // same as 4, should it be an exception?
            tri = tensor.GetTriangle(5);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));
            tri = tensor.GetTriangle(1000);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));

            tri = tensor.GetTriangle(-1);
            expected = tensorConstructor.CreateFromArray<int>(new[,]
            {
               { 0, 0, 0, 0, 0 },
               { 8, 0, 0, 0, 0 },
               { 1, 7, 0, 0, 0 }
            });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));

            expected = tensorConstructor.CreateFromArray<int>(new[,]
            {
               { 0, 0, 0, 0, 0 },
               { 0, 0, 0, 0, 0 },
               { 1, 0, 0, 0, 0 }
            });
            tri = tensor.GetTriangle(-2);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));

            expected = tensorConstructor.CreateFromArray<int>(new[,]
            {
               { 0, 0, 0, 0, 0 },
               { 0, 0, 0, 0, 0 },
               { 0, 0, 0, 0, 0 }
            });
            tri = tensor.GetTriangle(-3);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));

            tri = tensor.GetTriangle(-4);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));
            tri = tensor.GetTriangle(-5);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));
            tri = tensor.GetTriangle(-100);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));
        }

        [Theory(DisplayName = "GetTriangleCube")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void GetTriangleCube(TensorConstructor tensorConstructor)
        {
            var arr = new[, ,]
            {
                {
                   { 1, 2, 4 },
                   { 8, 3, 9 },
                   { 1, 7, 5 },
                },
                {
                   { 4, 5, 7 },
                   { 1, 6, 2 },
                   { 3, 0, 8 },
                },
                {
                   { 5, 6, 1 },
                   { 2, 2, 3 },
                   { 4, 9, 4 },
                },

            };

            var tensor = tensorConstructor.CreateFromArray<int>(arr);
            var tri = tensor.GetTriangle(0);
            var expected = tensorConstructor.CreateFromArray<int>(new[, ,]
            {
                {
                   { 1, 2, 4 },
                   { 0, 0, 0 },
                   { 0, 0, 0 },
                },
                {
                   { 4, 5, 7 },
                   { 1, 6, 2 },
                   { 0, 0, 0 },
                },
                {
                   { 5, 6, 1 },
                   { 2, 2, 3 },
                   { 4, 9, 4 },
                },

            });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));
            Assert.Equal(tensorConstructor.IsReversedStride, tri.IsReversedStride);
        }

        [Theory(DisplayName = "GetUpperTriangleSquare")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void GetUpperTriangleSquare(TensorConstructor tensorConstructor)
        {
            var arr = new[,]
            {
               { 1, 2, 4 },
               { 8, 3, 9 },
               { 1, 7, 5 },
            };

            var tensor = tensorConstructor.CreateFromArray<int>(arr);
            var tri = tensor.GetUpperTriangle(0);

            var expected = tensorConstructor.CreateFromArray<int>(new[,]
             {
               { 1, 2, 4 },
               { 0, 3, 9 },
               { 0, 0, 5 },
            });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));
            Assert.Equal(tensorConstructor.IsReversedStride, tri.IsReversedStride);

            tri = tensor.GetUpperTriangle(1);
            expected = tensorConstructor.CreateFromArray<int>(new[,]
            {
               { 0, 2, 4 },
               { 0, 0, 9 },
               { 0, 0, 0 },
            });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));
            tri = tensor.GetUpperTriangle(2);
            expected = tensorConstructor.CreateFromArray<int>(new[,]
            {
               { 0, 0, 4 },
               { 0, 0, 0 },
               { 0, 0, 0 },
            });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));

            tri = tensor.GetUpperTriangle(3);
            expected = tensorConstructor.CreateFromArray<int>(new[,]
            {
               { 0, 0, 0 },
               { 0, 0, 0 },
               { 0, 0, 0 },
            });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));

            tri = tensor.GetUpperTriangle(4);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));
            tri = tensor.GetUpperTriangle(42);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));

            tri = tensor.GetUpperTriangle(-1);
            expected = tensorConstructor.CreateFromArray<int>(new[,]
            {
               { 1, 2, 4 },
               { 8, 3, 9 },
               { 0, 7, 5 },
            });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));
            tri = tensor.GetUpperTriangle(-2);
            expected = tensorConstructor.CreateFromArray<int>(new[,]
            {
               { 1, 2, 4 },
               { 8, 3, 9 },
               { 1, 7, 5 },
            });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));

            tri = tensor.GetUpperTriangle(-3);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));
            tri = tensor.GetUpperTriangle(-300);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));
        }

        [Theory(DisplayName = "GetUpperTriangleRectangle")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void GetUpperTriangleRectangle(TensorConstructor tensorConstructor)
        {
            var arr = new[,]
            {
               { 1, 2, 4, 3, 7 },
               { 8, 3, 9, 2, 6 },
               { 1, 7, 5, 2, 9 }
            };

            var tensor = tensorConstructor.CreateFromArray<int>(arr);
            var tri = tensor.GetUpperTriangle(0);
            var expected = tensorConstructor.CreateFromArray<int>(new[,]
            {
               { 1, 2, 4, 3, 7 },
               { 0, 3, 9, 2, 6 },
               { 0, 0, 5, 2, 9 }
            });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));
            Assert.Equal(tensorConstructor.IsReversedStride, tri.IsReversedStride);
            tri = tensor.GetUpperTriangle(1);
            expected = tensorConstructor.CreateFromArray<int>(new[,]
            {
               { 0, 2, 4, 3, 7 },
               { 0, 0, 9, 2, 6 },
               { 0, 0, 0, 2, 9 }
            });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));
            tri = tensor.GetUpperTriangle(2);
            expected = tensorConstructor.CreateFromArray<int>(new[,]
            {
               { 0, 0, 4, 3, 7 },
               { 0, 0, 0, 2, 6 },
               { 0, 0, 0, 0, 9 }
            });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));
            tri = tensor.GetUpperTriangle(3);
            expected = tensorConstructor.CreateFromArray<int>(new[,]
            {
               { 0, 0, 0, 3, 7 },
               { 0, 0, 0, 0, 6 },
               { 0, 0, 0, 0, 0 }
            });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));

            tri = tensor.GetUpperTriangle(4);
            expected = tensorConstructor.CreateFromArray<int>(new[,]
            {
               { 0, 0, 0, 0, 7 },
               { 0, 0, 0, 0, 0 },
               { 0, 0, 0, 0, 0 }
            });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));

            expected = tensorConstructor.CreateFromArray<int>(new[,]
            {
               { 0, 0, 0, 0, 0 },
               { 0, 0, 0, 0, 0 },
               { 0, 0, 0, 0, 0 }
            });
            tri = tensor.GetUpperTriangle(5);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));
            tri = tensor.GetUpperTriangle(6);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));
            tri = tensor.GetUpperTriangle(1000);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));

            tri = tensor.GetUpperTriangle(-1);
            expected = tensorConstructor.CreateFromArray<int>(new[,]
            {
               { 1, 2, 4, 3, 7 },
               { 8, 3, 9, 2, 6 },
               { 0, 7, 5, 2, 9 }
            });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));

            expected = tensorConstructor.CreateFromArray<int>(new[,]
            {
               { 1, 2, 4, 3, 7 },
               { 8, 3, 9, 2, 6 },
               { 1, 7, 5, 2, 9 }
            });
            tri = tensor.GetUpperTriangle(-2);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));

            tri = tensor.GetUpperTriangle(-3);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));
            tri = tensor.GetUpperTriangle(-4);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));
            tri = tensor.GetUpperTriangle(-100);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));
        }

        [Theory(DisplayName = "GetUpperTriangleCube")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void GetUpperTriangleCube(TensorConstructor tensorConstructor)
        {
            var arr = new[, ,]
            {
                {
                   { 1, 2, 4 },
                   { 8, 3, 9 },
                   { 1, 7, 5 },
                },
                {
                   { 4, 5, 7 },
                   { 1, 6, 2 },
                   { 3, 0, 8 },
                },
                {
                   { 5, 6, 1 },
                   { 2, 2, 3 },
                   { 4, 9, 4 },
                },

            };

            var tensor = tensorConstructor.CreateFromArray<int>(arr);
            var tri = tensor.GetUpperTriangle(0);
            var expected = tensorConstructor.CreateFromArray<int>(new[, ,]
            {
                {
                   { 1, 2, 4 },
                   { 8, 3, 9 },
                   { 1, 7, 5 },
                },
                {
                   { 0, 0, 0 },
                   { 1, 6, 2 },
                   { 3, 0, 8 },
                },
                {
                   { 0, 0, 0 },
                   { 0, 0, 0 },
                   { 4, 9, 4 },
                },

            });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tri, expected));
            Assert.Equal(tensorConstructor.IsReversedStride, tri.IsReversedStride);
        }

        [Theory(DisplayName = "Reshape")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void Reshape(TensorConstructor tensorConstructor)
        {
            var arr = new[,]
            {
                { 1, 2, 3 },
                { 4, 5, 6 }
            };

            var tensor = tensorConstructor.CreateFromArray<int>(arr);
            var actual = tensor.Reshape(new[] { 3, 2 });

            var expected = tensorConstructor.IsReversedStride ?
                new[,]
                {
                    { 1, 5 },
                    { 4, 3 },
                    { 2, 6 }
                } :
                new[,]
                {
                    { 1, 2 },
                    { 3, 4 },
                    { 5, 6 }
                };
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
            Assert.Equal(tensorConstructor.IsReversedStride, actual.IsReversedStride);
        }

        [Fact(DisplayName = "Identity")]
        public void Identity()
        {
            var actual = Tensor.CreateIdentity<double>(3);

            var expected = new[,]
            {
                {1.0, 0, 0 },
                {0, 1.0, 0 },
                {0, 0, 1.0 }
            };

            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
        }

        [Theory(DisplayName = "CreateWithDiagonal")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void CreateWithDiagonal(TensorConstructor tensorConstructor)
        {
            var diagonal = tensorConstructor.CreateFromArray<int>(new[] { 1, 2, 3, 4, 5 });
            var actual = Tensor.CreateFromDiagonal(diagonal);

            var expected = new[,]
            {
                {1, 0, 0, 0, 0 },
                {0, 2, 0, 0, 0 },
                {0, 0, 3, 0, 0 },
                {0, 0, 0, 4, 0 },
                {0, 0, 0, 0, 5 }
            };

            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
        }

        [Theory(DisplayName = "CreateWithDiagonal3D")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void CreateWithDiagonal3D(TensorConstructor tensorConstructor)
        {
            var diagonal = tensorConstructor.CreateFromArray<int>(new[,]
            {
                { 1, 2, 3, 4, 5 },
                { 1, 2, 3, 4, 5 },
                { 1, 2, 3, 4, 5 }
            });
            var actual = Tensor.CreateFromDiagonal(diagonal);
            var expected = new[, ,]
            {
                {
                    {1, 2, 3, 4, 5 },
                    {0, 0, 0, 0, 0 },
                    {0, 0, 0, 0, 0 }
                },
                {
                    {0, 0, 0, 0, 0 },
                    {1, 2, 3, 4, 5 },
                    {0, 0, 0, 0, 0 }
                },
                {
                    {0, 0, 0, 0, 0 },
                    {0, 0, 0, 0, 0 },
                    {1, 2, 3, 4, 5 }
                }
            };

            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
        }

        [Theory(DisplayName = "CreateWithDiagonalAndOffset")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void CreateWithDiagonalAndOffset(TensorConstructor tensorConstructor)
        {
            var diagonal = tensorConstructor.CreateFromArray<int>(new[] { 1, 2, 3, 4 });
            var actual = Tensor.CreateFromDiagonal(diagonal, 1);

            var expected = new[,]
            {
                {0, 1, 0, 0, 0 },
                {0, 0, 2, 0, 0 },
                {0, 0, 0, 3, 0 },
                {0, 0, 0, 0, 4 },
                {0, 0, 0, 0, 0 }
            };

            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));

            diagonal = tensorConstructor.CreateFromArray<int>(new[] { 1, 2, 3, 4 });
            actual = Tensor.CreateFromDiagonal(diagonal, -1);

            expected = new[,]
            {
                {0, 0, 0, 0, 0 },
                {1, 0, 0, 0, 0 },
                {0, 2, 0, 0, 0 },
                {0, 0, 3, 0, 0 },
                {0, 0, 0, 4, 0 }
            };

            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));

            diagonal = tensorConstructor.CreateFromArray<int>(new[] { 1 });
            actual = Tensor.CreateFromDiagonal(diagonal, -4);
            expected = new[,]
            {
                {0, 0, 0, 0, 0 },
                {0, 0, 0, 0, 0 },
                {0, 0, 0, 0, 0 },
                {0, 0, 0, 0, 0 },
                {1, 0, 0, 0, 0 }
            };
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));

            diagonal = tensorConstructor.CreateFromArray<int>(new[] { 1 });
            actual = Tensor.CreateFromDiagonal(diagonal, 4);
            expected = new[,]
            {
                {0, 0, 0, 0, 1 },
                {0, 0, 0, 0, 0 },
                {0, 0, 0, 0, 0 },
                {0, 0, 0, 0, 0 },
                {0, 0, 0, 0, 0 }
            };
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
        }

        [Theory(DisplayName = "CreateWithDiagonalAndOffset3D")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void CreateWithDiagonalAndOffset3D(TensorConstructor tensorConstructor)
        {
            var diagonal = tensorConstructor.CreateFromArray<int>(new[,]
            {
                { 1, 2, 3 },
                { 1, 2, 3 },
                { 1, 2, 3 }
            });
            var actual = Tensor.CreateFromDiagonal(diagonal, 1);

            var expected = new[, ,]
            {
                {
                    { 0, 0, 0 },
                    { 1, 2, 3 },
                    { 0, 0, 0 },
                    { 0, 0, 0 }
                },
                {
                    { 0, 0, 0 },
                    { 0, 0, 0 },
                    { 1, 2, 3 },
                    { 0, 0, 0 }
                },
                {
                    { 0, 0, 0 },
                    { 0, 0, 0 },
                    { 0, 0, 0 },
                    { 1, 2, 3 }
                },
                {
                    { 0, 0, 0 },
                    { 0, 0, 0 },
                    { 0, 0, 0 },
                    { 0, 0, 0 }
                }
            };

            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));

            diagonal = tensorConstructor.CreateFromArray<int>(new[,]
            {
                { 1, 2, 3 },
                { 1, 2, 3 },
                { 1, 2, 3 }
            });
            actual = Tensor.CreateFromDiagonal(diagonal, -1);

            expected = new[, ,]
            {
                {
                    { 0, 0, 0 },
                    { 0, 0, 0 },
                    { 0, 0, 0 },
                    { 0, 0, 0 }
                },
                {
                    { 1, 2, 3 },
                    { 0, 0, 0 },
                    { 0, 0, 0 },
                    { 0, 0, 0 }
                },
                {
                    { 0, 0, 0 },
                    { 1, 2, 3 },
                    { 0, 0, 0 },
                    { 0, 0, 0 }
                },
                {
                    { 0, 0, 0 },
                    { 0, 0, 0 },
                    { 1, 2, 3 },
                    { 0, 0, 0 }
                }
            };

            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));

            diagonal = tensorConstructor.CreateFromArray<int>(new[,]
            {
                { 1, 2, 3 }
            });
            actual = Tensor.CreateFromDiagonal(diagonal, 3);

            expected = new[, ,]
            {
                {
                    { 0, 0, 0 },
                    { 0, 0, 0 },
                    { 0, 0, 0 },
                    { 1, 2, 3 },
                },
                {
                    { 0, 0, 0 },
                    { 0, 0, 0 },
                    { 0, 0, 0 },
                    { 0, 0, 0 }
                },
                {
                    { 0, 0, 0 },
                    { 0, 0, 0 },
                    { 0, 0, 0 },
                    { 0, 0, 0 }
                },
                {
                    { 0, 0, 0 },
                    { 0, 0, 0 },
                    { 0, 0, 0 },
                    { 0, 0, 0 }
                }
            };

            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));

            diagonal = tensorConstructor.CreateFromArray<int>(new[,]
            {
                { 1, 2, 3 }
            });
            actual = Tensor.CreateFromDiagonal(diagonal, -3);

            expected = new[, ,]
            {
                {
                    { 0, 0, 0 },
                    { 0, 0, 0 },
                    { 0, 0, 0 },
                    { 0, 0, 0 },
                },
                {
                    { 0, 0, 0 },
                    { 0, 0, 0 },
                    { 0, 0, 0 },
                    { 0, 0, 0 }
                },
                {
                    { 0, 0, 0 },
                    { 0, 0, 0 },
                    { 0, 0, 0 },
                    { 0, 0, 0 }
                },
                {
                    { 1, 2, 3 },
                    { 0, 0, 0 },
                    { 0, 0, 0 },
                    { 0, 0, 0 }
                }
            };

            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
        }

        [Theory(DisplayName = "Add")]
        [MemberData(nameof(GetDualTensorConstructors))]
        public void Add(TensorConstructor leftConstructor, TensorConstructor rightConstructor)
        {
            var left = leftConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 2},
                    {3, 4, 5}
                });
            var right = rightConstructor.CreateFromArray<int>(
                new[,]
                {
                    { 6, 7 ,8 },
                    { 9, 10 ,11 },
                });

            var expected = leftConstructor.CreateFromArray<int>(
                new[,]
                {
                    { 6, 8, 10 },
                    { 12, 14, 16 },
                });

            var actual = TensorOperations.Add(left, right);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
            Assert.Equal(leftConstructor.IsReversedStride, actual.IsReversedStride);

        }

        [Theory(DisplayName = "AddScalar")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void AddScalar(TensorConstructor tensorConstructor)
        {
            var tensor = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 2},
                    {3, 4, 5}
                });

            var expected = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    { 1, 2, 3 },
                    { 4, 5, 6 },
                });

            var actual = TensorOperations.Add(tensor, 1);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
            Assert.Equal(tensorConstructor.IsReversedStride, actual.IsReversedStride);

        }

        [Theory(DisplayName = "UnaryPlus")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void UnaryPlus(TensorConstructor tensorConstructor)
        {
            var tensor = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 2},
                    {3, 4, 5}
                });

            var expected = tensor;

            var actual = TensorOperations.UnaryPlus(tensor);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
            Assert.False(ReferenceEquals(actual, expected));
            Assert.Equal(tensorConstructor.IsReversedStride, actual.IsReversedStride);
        }

        [Theory(DisplayName = "Subtract")]
        [MemberData(nameof(GetDualTensorConstructors))]
        public void Subtract(TensorConstructor leftConstructor, TensorConstructor rightConstructor)
        {
            var left = leftConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 2},
                    {3, 4, 5}
                });
            var right = rightConstructor.CreateFromArray<int>(
                new[,]
                {
                    { 6, 7 ,8 },
                    { 9, 10 ,11 },
                });

            var expected = leftConstructor.CreateFromArray<int>(
                new[,]
                {
                    { -6, -6, -6 },
                    { -6, -6, -6},
                });

            var actual = TensorOperations.Subtract(left, right);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
            Assert.Equal(leftConstructor.IsReversedStride, actual.IsReversedStride);
        }

        [Theory(DisplayName = "SubtractScalar")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void SubtractScalar(TensorConstructor tensorConstructor)
        {
            var tensor = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 2},
                    {3, 4, 5}
                });
            var expected = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    { -1, 0, 1 },
                    { 2, 3, 4 },
                });

            var actual = TensorOperations.Subtract(tensor, 1);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
            Assert.Equal(tensorConstructor.IsReversedStride, actual.IsReversedStride);
        }

        [Theory(DisplayName = "UnaryMinus")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void UnaryMinus(TensorConstructor tensorConstructor)
        {
            var tensor = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 2},
                    {3, 4, 5}
                });

            var expected = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, -1, -2},
                    {-3, -4, -5}
                });

            var actual = TensorOperations.UnaryMinus(tensor);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
            Assert.False(ReferenceEquals(actual, expected));
            Assert.Equal(tensorConstructor.IsReversedStride, actual.IsReversedStride);
        }

        [Theory(DisplayName = "PrefixIncrement")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void PrefixIncrement(TensorConstructor tensorConstructor)
        {
            Tensor<int> tensor = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 2},
                    {3, 4, 5}
                });

            var expectedResult = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {1, 2, 3},
                    {4, 5, 6}
                });

            var expectedTensor = expectedResult;

            tensor = TensorOperations.Increment(tensor);
            var actual = tensor;
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expectedResult));
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tensor, expectedTensor));
            Assert.True(ReferenceEquals(tensor, actual));
            Assert.Equal(tensorConstructor.IsReversedStride, actual.IsReversedStride);
        }

        [Theory(DisplayName = "PostfixIncrement")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void PostfixIncrement(TensorConstructor tensorConstructor)
        {
            Tensor<int> tensor = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 2},
                    {3, 4, 5}
                });

            // returns original value
            var expectedResult = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 2},
                    {3, 4, 5}
                });

            // increments operand
            var expectedTensor = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {1, 2, 3},
                    {4, 5, 6}
                }); ;

            var actual = tensor;
            tensor = TensorOperations.Increment(tensor);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expectedResult));
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tensor, expectedTensor));
            Assert.False(ReferenceEquals(tensor, actual));
            Assert.Equal(tensorConstructor.IsReversedStride, actual.IsReversedStride);
        }

        [Theory(DisplayName = "PrefixDecrement")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void PrefixDecrement(TensorConstructor tensorConstructor)
        {
            Tensor<int> tensor = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 2},
                    {3, 4, 5}
                });

            var expectedResult = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {-1, 0, 1},
                    {2, 3, 4}
                });

            var expectedTensor = expectedResult;

            tensor = TensorOperations.Decrement(tensor);
            var actual = tensor;
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expectedResult));
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tensor, expectedTensor));
            Assert.True(ReferenceEquals(tensor, actual));
            Assert.Equal(tensorConstructor.IsReversedStride, actual.IsReversedStride);
        }

        [Theory(DisplayName = "PostfixDecrement")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void PostfixDecrement(TensorConstructor tensorConstructor)
        {
            Tensor<int> tensor = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 2},
                    {3, 4, 5}
                });

            // returns original value
            var expectedResult = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 2},
                    {3, 4, 5}
                });

            // decrements operand
            var expectedTensor = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {-1, 0, 1},
                    {2, 3, 4}
                }); ;

            var actual = tensor;
            tensor = TensorOperations.Decrement(tensor);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expectedResult));
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(tensor, expectedTensor));
            Assert.False(ReferenceEquals(tensor, actual));
            Assert.Equal(tensorConstructor.IsReversedStride, actual.IsReversedStride);
        }

        [Theory(DisplayName = "Multiply")]
        [MemberData(nameof(GetDualTensorConstructors))]
        public void Multiply(TensorConstructor leftConstructor, TensorConstructor rightConstructor)
        {
            var left = leftConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 2},
                    {3, 4, 5}
                });
            var right = rightConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 2},
                    {3, 4, 5}
                });

            var expected = leftConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 4},
                    {9, 16, 25}
                });

            var actual = TensorOperations.Multiply(left, right);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
            Assert.Equal(leftConstructor.IsReversedStride, actual.IsReversedStride);
        }

        [Theory(DisplayName = "MultiplyScalar")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void MultiplyScalar(TensorConstructor tensorConstructor)
        {
            var tensor = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 2},
                    {3, 4, 5}
                });

            var expected = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 2, 4},
                    {6, 8, 10}
                });

            var actual = TensorOperations.Multiply(tensor, 2);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
            Assert.Equal(tensorConstructor.IsReversedStride, actual.IsReversedStride);
        }

        [Theory(DisplayName = "Divide")]
        [MemberData(nameof(GetDualTensorConstructors))]
        public void Divide(TensorConstructor dividendConstructor, TensorConstructor divisorConstructor)
        {
            var dividend = dividendConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 4},
                    {9, 16, 25}
                });

            var divisor = divisorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {1, 1, 2},
                    {3, 4, 5}
                });

            var expected = divisorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 2},
                    {3, 4, 5}
                });

            var actual = TensorOperations.Divide(dividend, divisor);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
            Assert.Equal(dividendConstructor.IsReversedStride, actual.IsReversedStride);
        }

        [Theory(DisplayName = "DivideScalar")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void DivideScalar(TensorConstructor tensorConstructor)
        {
            var tensor = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 2, 4},
                    {6, 8, 10}
                });

            var expected = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 2},
                    {3, 4, 5}
                });

            var actual = TensorOperations.Divide(tensor, 2);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
            Assert.Equal(tensorConstructor.IsReversedStride, actual.IsReversedStride);
        }

        [Theory(DisplayName = "Modulo")]
        [MemberData(nameof(GetDualTensorConstructors))]
        public void Modulo(TensorConstructor dividendConstructor, TensorConstructor divisorConstructor)
        {
            var dividend = dividendConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 3, 8},
                    {11, 14, 17}
                });

            var divisor = divisorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {1, 2, 3},
                    {4, 5, 6}
                });

            var expected = dividendConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 2},
                    {3, 4, 5}
                });

            var actual = TensorOperations.Modulo(dividend, divisor);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
            Assert.Equal(dividendConstructor.IsReversedStride, actual.IsReversedStride);
        }

        [Theory(DisplayName = "ModuloScalar")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void ModuloScalar(TensorConstructor tensorConstructor)
        {
            var tensor = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 3, 4},
                    {7, 8, 9}
                });

            var expected = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 0},
                    {1, 0, 1}
                });

            var actual = TensorOperations.Modulo(tensor, 2);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
            Assert.Equal(tensorConstructor.IsReversedStride, actual.IsReversedStride);
        }

        [Theory(DisplayName = "And")]
        [MemberData(nameof(GetDualTensorConstructors))]
        public void And(TensorConstructor leftConstructor, TensorConstructor rightConstructor)
        {
            var left = leftConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 3},
                    {7, 15, 31}
                });

            var right = rightConstructor.CreateFromArray<int>(
                new[,]
                {
                    {1, 1, 3},
                    {2, 4, 8}
                });

            var expected = leftConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 3},
                    {2, 4, 8}
                });

            var actual = TensorOperations.And(left, right);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
            Assert.Equal(leftConstructor.IsReversedStride, actual.IsReversedStride);
        }

        [Theory(DisplayName = "AndScalar")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void AndScalar(TensorConstructor tensorConstructor)
        {
            var left = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 3},
                    {5, 15, 31}
                });

            var expected = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 0, 0},
                    {4, 4, 20}
                });

            var actual = TensorOperations.And(left, 20);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
            Assert.Equal(tensorConstructor.IsReversedStride, actual.IsReversedStride);
        }

        [Theory(DisplayName = "Or")]
        [MemberData(nameof(GetDualTensorConstructors))]
        public void Or(TensorConstructor leftConstructor, TensorConstructor rightConstructor)
        {
            var left = leftConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 3},
                    {7, 14, 31}
                });

            var right = rightConstructor.CreateFromArray<int>(
                new[,]
                {
                    {1, 2, 4},
                    {2, 4, 8}
                });

            var expected = leftConstructor.CreateFromArray<int>(
                new[,]
                {
                    {1, 3, 7},
                    {7, 14, 31}
                });

            var actual = TensorOperations.Or(left, right);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
            Assert.Equal(leftConstructor.IsReversedStride, actual.IsReversedStride);
        }

        [Theory(DisplayName = "OrScalar")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void OrScalar(TensorConstructor tensorConstructor)
        {
            var left = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 2},
                    {3, 4, 5}
                });

            var expected = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {1, 1, 3},
                    {3, 5, 5}
                });

            var actual = TensorOperations.Or(left, 1);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
            Assert.Equal(tensorConstructor.IsReversedStride, actual.IsReversedStride);
        }

        [Theory(DisplayName = "Xor")]
        [MemberData(nameof(GetDualTensorConstructors))]
        public void Xor(TensorConstructor leftConstructor, TensorConstructor rightConstructor)
        {
            var left = leftConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 3},
                    {7, 14, 31}
                });

            var right = rightConstructor.CreateFromArray<int>(
                new[,]
                {
                    {1, 2, 4},
                    {2, 4, 8}
                });

            var expected = leftConstructor.CreateFromArray<int>(
                new[,]
                {
                    {1, 3, 7},
                    {5, 10, 23}
                });

            var actual = TensorOperations.Xor(left, right);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
            Assert.Equal(leftConstructor.IsReversedStride, actual.IsReversedStride);
        }

        [Theory(DisplayName = "XorScalar")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void XorScalar(TensorConstructor tensorConstructor)
        {
            var left = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 2},
                    {3, 4, 5}
                });

            var expected = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {1, 0, 3},
                    {2, 5, 4}
                });

            var actual = TensorOperations.Xor(left, 1);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
            Assert.Equal(tensorConstructor.IsReversedStride, actual.IsReversedStride);
        }

        [Theory(DisplayName = "LeftShift")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void LeftShift(TensorConstructor tensorConstructor)
        {
            var left = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 2},
                    {3, 4, 5}
                });

            var expected = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 2, 4},
                    {6, 8, 10}
                });

            var actual = TensorOperations.LeftShift(left, 1);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
            Assert.Equal(tensorConstructor.IsReversedStride, actual.IsReversedStride);
        }

        [Theory(DisplayName = "RightShift")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void RightShift(TensorConstructor tensorConstructor)
        {
            var left = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 2},
                    {3, 4, 5}
                });

            var expected = tensorConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 0, 1},
                    {1, 2, 2}
                });

            var actual = TensorOperations.RightShift(left, 1);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
            Assert.Equal(tensorConstructor.IsReversedStride, actual.IsReversedStride);
        }

        [Theory(DisplayName = "ElementWiseEquals")]
        [MemberData(nameof(GetDualTensorConstructors))]
        public void ElementWiseEquals(TensorConstructor leftConstructor, TensorConstructor rightConstructor)
        {
            var left = leftConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 2},
                    {3, 4, 5}
                });
            var right = rightConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, -2},
                    {2, 3, 5}
                });

            var expected = new[,]
                {
                    {true, true, false },
                    {false, false, true}
                }.ToTensor();

            var actual = TensorOperations.Equals(left, right);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
            Assert.Equal(leftConstructor.IsReversedStride, actual.IsReversedStride);
        }

        [Theory(DisplayName = "ElementWiseNotEquals")]
        [MemberData(nameof(GetDualTensorConstructors))]
        public void ElementWiseNotEquals(TensorConstructor leftConstructor, TensorConstructor rightConstructor)
        {
            var left = leftConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 2},
                    {3, 4, 5}
                });
            var right = rightConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, -2},
                    {2, 3, 5}
                });

            var expected = new[,]
                {
                    {false, false, true},
                    {true, true, false}
                }.ToTensor();

            var actual = TensorOperations.NotEquals(left, right);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
            Assert.Equal(leftConstructor.IsReversedStride, actual.IsReversedStride);
        }

        [Theory(DisplayName = "MatrixMultiply")]
        [MemberData(nameof(GetDualTensorConstructors))]
        public void MatrixMultiply(TensorConstructor leftConstructor, TensorConstructor rightConstructor)
        {
            var left = leftConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 2},
                    {3, 4, 5}
                });

            var right = rightConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0, 1, 2, 3, 4},
                    {5, 6, 7, 8, 9},
                    {10, 11, 12, 13, 14}
                });

            var expected = leftConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0*0 + 1*5 + 2*10, 0*1 + 1*6 + 2*11, 0*2 + 1*7 + 2*12, 0*3 + 1*8 + 2*13, 0*4 + 1*9 + 2*14},
                    {3*0 + 4*5 + 5*10, 3*1 + 4*6 + 5*11, 3*2 + 4*7 + 5*12, 3*3 + 4*8 + 5*13, 3*4 + 4*9 + 5*14}
                });

            var actual = left.MatrixMultiply(right);
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
        }

        [Theory(DisplayName = "Contract")]
        [MemberData(nameof(GetDualTensorConstructors))]
        public void Contract(TensorConstructor leftConstructor, TensorConstructor rightConstructor)
        {
            var left = leftConstructor.CreateFromArray<int>(
                new[, ,]
                {
                    {
                        {0, 1},
                        {2, 3}
                    },
                    {
                        {4, 5},
                        {6, 7}
                    },
                    {
                        {8, 9},
                        {10, 11}
                    }
                });

            var right = rightConstructor.CreateFromArray<int>(
                new[, ,]
                {
                    {
                        {0, 1},
                        {2, 3},
                        {4, 5}
                    },
                    {
                        {6, 7},
                        {8, 9},
                        {10, 11}
                    },
                    {
                        {12, 13},
                        {14, 15},
                        {16, 17}
                    },
                    {
                        {18, 19},
                        {20, 21},
                        {22, 23}
                    }
                });

            // contract a 3*2*2 with a 4*3*2 tensor, summing on (3*2)*2 and 4*(3*2) to produce a 2*4 tensor
            var expected = leftConstructor.CreateFromArray<int>(
                new[,]
                {
                    {110, 290, 470, 650},
                    {125, 341, 557, 773},
                });
            var actual = TensorOperations.Contract(left, right, new[] { 0, 1 }, new[] { 1, 2 });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));

            // contract a 3*2*2 with a 4*3*2 tensor, summing on (3)*2*(2) and 4*(3*2) to produce a 2*4 tensor
            expected = leftConstructor.CreateFromArray<int>(
                new[,]
                {
                    {101, 263, 425, 587},
                    {131, 365, 599, 833},
                });
            actual = TensorOperations.Contract(left, right, new[] { 0, 2 }, new[] { 1, 2 });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
        }

        [Theory(DisplayName = "ContractWithSingleLengthDimension")]
        [MemberData(nameof(GetDualTensorConstructors))]
        public void ContractWithSingleLengthDimension(TensorConstructor leftConstructor, TensorConstructor rightConstructor)
        {
            var left = leftConstructor.CreateFromArray<int>(
                new[,]
                {
                    {1, 2, 3},
                    {4, 5, 6},
                });

            var right = rightConstructor.CreateFromArray<int>(
                new[,]
                {
                    { 1, 2 },
                    { 3, 4 },
                    { 5, 6 }
                });

            var expected = leftConstructor.CreateFromArray<int>(
                new[,]
                {
                    { 22, 28 },
                    { 49, 64 }
                });

            // contract a 2*3 with a 3*2 tensor, summing on 2*(3) and (3)*2 to produce a 2*2 tensor
            var actual = TensorOperations.Contract(left, right, new[] { 1 }, new[] { 0 });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));

            // contract a 1*2*3*1 with a 3*2 tensor, summing on 1*2*(3)*1 and (3)*2 to produce a 1*2*1*2 tensor
            var reshapedLeft = left.Reshape(new int[] { 1, 2, 3, 1 });
            var reshapedExpected = expected.Reshape(new int[] { 1, 2, 1, 2 });
            actual = TensorOperations.Contract(reshapedLeft, right, new[] { 2 }, new[] { 0 });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, reshapedExpected));

        }

        [Theory(DisplayName = "ContractMismatchedDimensions")]
        [MemberData(nameof(GetDualTensorConstructors))]
        public void ContractMismatchedDimensions(TensorConstructor leftConstructor, TensorConstructor rightConstructor)
        {
            var left = leftConstructor.CreateFromArray<int>(
                new[] { 0, 1, 2, 3 });

            var right = rightConstructor.CreateFromArray<int>(
                new[,]
                {
                    { 0 },
                    { 1 },
                    { 2 }
                });

            var expected = leftConstructor.CreateFromArray<int>(
                new[,]
                {
                    {0,0,0},
                    {0,1,2},
                    {0,2,4},
                    {0,3,6},
                });

            Assert.Throws<ArgumentException>(() => TensorOperations.Contract(left, right, new int[] { }, new[] { 1 }));

            // reshape to include dimension of length 1.
            var leftReshaped = left.Reshape(new[] { 1, (int)left.Length });

            var actual = TensorOperations.Contract(leftReshaped, right, new[] { 0 }, new[] { 1 });
            Assert.True(StructuralComparisons.StructuralEqualityComparer.Equals(actual, expected));
        }

        [Theory(DisplayName = "GetArrayString")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void GetArrayString(TensorConstructor constructor)
        {
            var tensor = constructor.CreateFromArray<int>(
                new[, ,]
                {
                    {
                        {0, 1},
                        {2, 3},
                        {4, 5}
                    },
                    {
                        {6, 7},
                        {8, 9},
                        {10, 11}
                    },
                    {
                        {12, 13},
                        {14, 15},
                        {16, 17}
                    },
                    {
                        {18, 19},
                        {20, 21},
                        {22, 23}
                    }
                });

            var expected =
@"{
    {
        {0,1},
        {2,3},
        {4,5}
    },
    {
        {6,7},
        {8,9},
        {10,11}
    },
    {
        {12,13},
        {14,15},
        {16,17}
    },
    {
        {18,19},
        {20,21},
        {22,23}
    }
}";

            Assert.Equal(expected, tensor.GetArrayString());

            var expectedNoSpace = expected.Replace(Environment.NewLine, "").Replace(" ", "");
            Assert.Equal(expectedNoSpace, tensor.GetArrayString(false));
        }

        [Theory(DisplayName = "TestICollectionMembers")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void TestICollectionMembers(TensorConstructor constructor)
        {
            var arr = new[,]
            {
                { 1, 2, 3 },
                { 4, 5, 6 }
            };

            var tensor = constructor.CreateFromArray<int>(arr);
            ICollection tensorCollection = tensor;

            Assert.Equal(6, tensorCollection.Count);

            Assert.False(tensorCollection.IsSynchronized);

            Assert.True(ReferenceEquals(tensorCollection, tensorCollection.SyncRoot));

            var actual = Array.CreateInstance(typeof(int), tensor.Length);
            tensorCollection.CopyTo(actual, 0);
            var expected = constructor.IsReversedStride ?
                new[] { 1, 4, 2, 5, 3, 6 } :
                new[] { 1, 2, 3, 4, 5, 6 };
            Assert.Equal(expected, actual);

            actual = Array.CreateInstance(typeof(int), tensor.Length + 2);
            tensorCollection.CopyTo(actual, 2);
            expected = constructor.IsReversedStride ?
                new[] { 0, 0, 1, 4, 2, 5, 3, 6 } :
                new[] { 0, 0, 1, 2, 3, 4, 5, 6 };
            Assert.Equal(expected, actual);

            Assert.Throws<ArgumentNullException>(() => tensorCollection.CopyTo(null, 0));
            Assert.Throws<ArgumentException>(() => tensorCollection.CopyTo(new int[3, 4], 0));
            Assert.Throws<ArgumentException>(() => tensorCollection.CopyTo(new int[5], 0));
            Assert.Throws<ArgumentException>(() => tensorCollection.CopyTo(new int[6], 1));
        }

        [Theory(DisplayName = "TestIListMembers")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void TestIListMembers(TensorConstructor constructor)
        {
            var arr = new[,]
            {
                { 1, 2, 3 },
                { 4, 5, 6 }
            };

            var tensor = constructor.CreateFromArray<int>(arr);
            IList tensorList = tensor;

            int expectedIndexValue = constructor.IsReversedStride ? 4 : 2;
            Assert.Equal(expectedIndexValue, tensorList[1]);

            tensorList[1] = 7;
            Assert.Equal(7, tensorList[1]);
            var expected = constructor.IsReversedStride ?
                new[] { 1, 7, 2, 5, 3, 6 } :
                new[] { 1, 7, 3, 4, 5, 6 };
            Assert.Equal(expected, tensor);

            Assert.True(tensorList.IsFixedSize);
            Assert.False(tensorList.IsReadOnly);

            Assert.Throws<InvalidOperationException>(() => (tensorList).Add(8));

            Assert.True(tensorList.Contains(5));
            Assert.True(tensorList.Contains(6));
            Assert.False(tensorList.Contains(0));
            Assert.False(tensorList.Contains(42));
            Assert.False(tensorList.Contains("foo"));

            Assert.Equal(constructor.IsReversedStride ? 3 : 4, tensorList.IndexOf(5));
            Assert.Equal(5, tensorList.IndexOf(6));
            Assert.Equal(-1, tensorList.IndexOf(0));
            Assert.Equal(-1, tensorList.IndexOf(42));

            Assert.Throws<InvalidOperationException>(() => (tensorList).Insert(2, 5));
            Assert.Throws<InvalidOperationException>(() => (tensorList).Remove(1));
            Assert.Throws<InvalidOperationException>(() => (tensorList).RemoveAt(0));

            tensorList.Clear();
            Assert.Equal(new[] { 0, 0, 0, 0, 0, 0 }, tensor);
        }

        [Theory(DisplayName = "TestICollectionTMembers")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void TestICollectionTMembers(TensorConstructor constructor)
        {
            var arr = new[,]
            {
                { 1, 2, 3 },
                { 4, 5, 6 }
            };

            var tensor = constructor.CreateFromArray<int>(arr);
            ICollection<int> tensorCollection = tensor;

            Assert.Equal(6, tensorCollection.Count);
            Assert.False(tensorCollection.IsReadOnly);

            Assert.Throws<InvalidOperationException>(() => tensorCollection.Add(8));
            Assert.Throws<InvalidOperationException>(() => tensorCollection.Remove(1));

            Assert.True(tensorCollection.Contains(5));
            Assert.True(tensorCollection.Contains(6));
            Assert.False(tensorCollection.Contains(0));
            Assert.False(tensorCollection.Contains(42));

            var actual = new int[tensor.Length];
            tensorCollection.CopyTo(actual, 0);
            var expected = constructor.IsReversedStride ?
                new[] { 1, 4, 2, 5, 3, 6 } :
                new[] { 1, 2, 3, 4, 5, 6 };
            Assert.Equal(expected, actual);

            actual = new int[tensor.Length + 2];
            tensorCollection.CopyTo(actual, 2);
            expected = constructor.IsReversedStride ?
                new[] { 0, 0, 1, 4, 2, 5, 3, 6 } :
                new[] { 0, 0, 1, 2, 3, 4, 5, 6 };
            Assert.Equal(expected, actual);

            Assert.Throws<ArgumentNullException>(() => tensorCollection.CopyTo(null, 0));
            Assert.Throws<ArgumentException>(() => tensorCollection.CopyTo(new int[5], 0));
            Assert.Throws<ArgumentException>(() => tensorCollection.CopyTo(new int[6], 1));

            tensorCollection.Clear();
            Assert.Equal(new[] { 0, 0, 0, 0, 0, 0 }, tensor);
        }

        [Theory(DisplayName = "TestIListTMembers")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void TestIListTMembers(TensorConstructor constructor)
        {
            var arr = new[,]
            {
                { 1, 2, 3 },
                { 4, 5, 6 }
            };

            var tensor = constructor.CreateFromArray<int>(arr);
            IList<int> tensorList = tensor;

            int expectedIndexValue = constructor.IsReversedStride ? 4 : 2;
            Assert.Equal(expectedIndexValue, tensorList[1]);

            tensorList[1] = 7;
            Assert.Equal(7, tensorList[1]);
            var expected = constructor.IsReversedStride ?
                new[] { 1, 7, 2, 5, 3, 6 } :
                new[] { 1, 7, 3, 4, 5, 6 };
            Assert.Equal(expected, tensor);

            Assert.Equal(constructor.IsReversedStride ? 3 : 4, tensorList.IndexOf(5));
            Assert.Equal(5, tensorList.IndexOf(6));
            Assert.Equal(-1, tensorList.IndexOf(0));
            Assert.Equal(-1, tensorList.IndexOf(42));

            Assert.Throws<InvalidOperationException>(() => (tensorList).Insert(2, 5));
            Assert.Throws<InvalidOperationException>(() => (tensorList).RemoveAt(0));
        }

        [Theory(DisplayName = "TestIReadOnlyTMembers")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void TestIReadOnlyTMembers(TensorConstructor constructor)
        {
            var arr = new[,]
            {
                { 1, 2, 3 },
                { 4, 5, 6 }
            };

            var tensor = constructor.CreateFromArray<int>(arr);

            IReadOnlyCollection<int> tensorCollection = tensor;
            Assert.Equal(6, tensorCollection.Count);

            IReadOnlyList<int> tensorList = tensor;
            int expectedIndexValue = constructor.IsReversedStride ? 4 : 2;
            Assert.Equal(expectedIndexValue, tensorList[1]);
        }
    }        
}