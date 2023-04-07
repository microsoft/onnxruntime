// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is copied and adapted from the following git repository -
// https://github.com/dotnet/corefx
// Commit ID: bdd0814360d4c3a58860919f292a306242f27da1
// Path: /src/System.Numerics.Tensors/tests/TensorArithmetic.cs
// Original license statement below -

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.OnnxRuntime.Tensors
{
    internal interface ITensorArithmetic<T>
    {
        T One { get; }
        T Zero { get; }
        void Add(Tensor<T> left, Tensor<T> right, Tensor<T> result);
        void Add(Tensor<T> tensor, T scalar, Tensor<T> result);
        void And(Tensor<T> left, Tensor<T> right, Tensor<T> result);
        void And(Tensor<T> tensor, T scalar, Tensor<T> result);
        void Contract(Tensor<T> left, Tensor<T> right, int[] leftAxes, int[] rightAxes, Tensor<T> result);
        void Decrement(Tensor<T> tensor, Tensor<T> result);
        void Divide(Tensor<T> left, Tensor<T> right, Tensor<T> result);
        void Divide(Tensor<T> tensor, T scalar, Tensor<T> result);
        void Equals(Tensor<T> left, Tensor<T> right, Tensor<bool> result);
        void GreaterThan(Tensor<T> left, Tensor<T> right, Tensor<bool> result);
        void GreaterThanOrEqual(Tensor<T> left, Tensor<T> right, Tensor<bool> result);
        void Increment(Tensor<T> tensor, Tensor<T> result);
        void LeftShift(Tensor<T> tensor, int value, Tensor<T> result);
        void LessThan(Tensor<T> left, Tensor<T> right, Tensor<bool> result);
        void LessThanOrEqual(Tensor<T> left, Tensor<T> right, Tensor<bool> result);
        void Modulo(Tensor<T> left, Tensor<T> right, Tensor<T> result);
        void Modulo(Tensor<T> tensor, T scalar, Tensor<T> result);
        void Multiply(Tensor<T> left, Tensor<T> right, Tensor<T> result);
        void Multiply(Tensor<T> tensor, T scalar, Tensor<T> result);
        void NotEquals(Tensor<T> left, Tensor<T> right, Tensor<bool> result);
        void Or(Tensor<T> left, Tensor<T> right, Tensor<T> result);
        void Or(Tensor<T> tensor, T scalar, Tensor<T> result);
        void RightShift(Tensor<T> tensor, int value, Tensor<T> result);
        void Subtract(Tensor<T> left, Tensor<T> right, Tensor<T> result);
        void Subtract(Tensor<T> tensor, T scalar, Tensor<T> result);
        void UnaryMinus(Tensor<T> tensor, Tensor<T> result);
        void UnaryPlus(Tensor<T> tensor, Tensor<T> result);
        void Xor(Tensor<T> left, Tensor<T> right, Tensor<T> result);
        void Xor(Tensor<T> tensor, T scalar, Tensor<T> result);
    }

    internal static class TensorArithmetic<T>
    {
        public static ITensorArithmetic<T> Instance => TensorArithmetic.GetArithmetic<T>();
    }

    internal static class TensorArithmetic
    { 
        public static ITensorArithmetic<T> GetArithmetic<T>()
        {
            if (typeof(T) == typeof(bool))
            {
                return (ITensorArithmetic<T>)new BoolArithmetic();
            }
            else if (typeof(T) == typeof(byte))
            {
                return (ITensorArithmetic<T>)new ByteArithmetic();
            }
            else if (typeof(T) == typeof(char))
            {
                return (ITensorArithmetic<T>)new CharArithmetic();
            }
            else if (typeof(T) == typeof(decimal))
            {
                return (ITensorArithmetic<T>)new DecimalArithmetic();
            }
            else if (typeof(T) == typeof(double))
            {
                return (ITensorArithmetic<T>)new DoubleArithmetic();
            }
            else if (typeof(T) == typeof(float))
            {
                return (ITensorArithmetic<T>)new FloatArithmetic();
            }
            else if (typeof(T) == typeof(int))
            {
                return (ITensorArithmetic<T>)new IntArithmetic();
            }
            else if (typeof(T) == typeof(long))
            {
                return (ITensorArithmetic<T>)new LongArithmetic();
            }
            else if (typeof(T) == typeof(sbyte))
            {
                return (ITensorArithmetic<T>)new SByteArithmetic();
            }
            else if (typeof(T) == typeof(short))
            {
                return (ITensorArithmetic<T>)new ShortArithmetic();
            }
            else if (typeof(T) == typeof(uint))
            {
                return (ITensorArithmetic<T>)new UIntArithmetic();
            }
            else if (typeof(T) == typeof(ulong))
            {
                return (ITensorArithmetic<T>)new ULongArithmetic();
            }
            else if (typeof(T) == typeof(ushort))
            {
                return (ITensorArithmetic<T>)new UShortArithmetic();
            }
            return null;
        }
    }
    
    internal class BoolArithmetic : ITensorArithmetic<bool>
    {
        public bool One => true;
        public bool Zero => false;

        public void Add(Tensor<bool> left, Tensor<bool> right, Tensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void Add(Tensor<bool> tensor, bool scalar, Tensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void And(Tensor<bool> left, Tensor<bool> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (bool)(left[indices] & right[indices]);
            }
            
        }
        public void And(Tensor<bool> tensor, bool scalar, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (bool)(tensor[indices] & scalar);
            }
            
        }
        public void Contract(Tensor<bool> left, Tensor<bool> right, int[] leftAxes, int[] rightAxes, Tensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void Decrement(Tensor<bool> tensor, Tensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void Divide(Tensor<bool> left, Tensor<bool> right, Tensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void Divide(Tensor<bool> tensor, bool scalar, Tensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void Equals(Tensor<bool> left, Tensor<bool> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] == right[indices];
            }
            
        }
        public void GreaterThan(Tensor<bool> left, Tensor<bool> right, Tensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void GreaterThanOrEqual(Tensor<bool> left, Tensor<bool> right, Tensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void Increment(Tensor<bool> tensor, Tensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void LeftShift(Tensor<bool> tensor, int value, Tensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void LessThan(Tensor<bool> left, Tensor<bool> right, Tensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void LessThanOrEqual(Tensor<bool> left, Tensor<bool> right, Tensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void Modulo(Tensor<bool> left, Tensor<bool> right, Tensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void Modulo(Tensor<bool> tensor, bool scalar, Tensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void Multiply(Tensor<bool> left, Tensor<bool> right, Tensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void Multiply(Tensor<bool> tensor, bool scalar, Tensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void NotEquals(Tensor<bool> left, Tensor<bool> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] != right[indices];
            }
            
        }
        public void Or(Tensor<bool> left, Tensor<bool> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (bool)(left[indices] | right[indices]);
            }
            
        }
        public void Or(Tensor<bool> tensor, bool scalar, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (bool)(tensor[indices] | scalar);
            }
            
        }
        public void RightShift(Tensor<bool> tensor, int value, Tensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void Subtract(Tensor<bool> left, Tensor<bool> right, Tensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void Subtract(Tensor<bool> tensor, bool scalar, Tensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void UnaryMinus(Tensor<bool> tensor, Tensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void UnaryPlus(Tensor<bool> tensor, Tensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void Xor(Tensor<bool> left, Tensor<bool> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (bool)(left[indices] ^ right[indices]);
            }
            
        }
        public void Xor(Tensor<bool> tensor, bool scalar, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (bool)(tensor[indices] ^ scalar);
            }
            
        }

        public void Add(DenseTensor<bool> left, DenseTensor<bool> right, DenseTensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void Add(DenseTensor<bool> tensor, bool scalar, DenseTensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void And(DenseTensor<bool> left, DenseTensor<bool> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (bool)(leftSpan[i] & rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (bool)(leftSpan[op1Index] & rightSpan[op2Index]);

                }
            }
        }
        public void And(DenseTensor<bool> tensor, bool scalar, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (bool)(tensorSpan[i] & scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (bool)(tensorSpan[op1Index] & scalar);

                }
            }
        }
        public void Contract(DenseTensor<bool> left, DenseTensor<bool> right, int[] leftAxes, int[] rightAxes, DenseTensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void Decrement(DenseTensor<bool> tensor, DenseTensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void Divide(DenseTensor<bool> left, DenseTensor<bool> right, DenseTensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void Divide(DenseTensor<bool> tensor, bool scalar, DenseTensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void Equals(DenseTensor<bool> left, DenseTensor<bool> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] == rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] == rightSpan[op2Index];

                }
            }
        }
        public void GreaterThan(DenseTensor<bool> left, DenseTensor<bool> right, DenseTensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void GreaterThanOrEqual(DenseTensor<bool> left, DenseTensor<bool> right, DenseTensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void Increment(DenseTensor<bool> tensor, DenseTensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void LeftShift(DenseTensor<bool> tensor, int value, DenseTensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void LessThan(DenseTensor<bool> left, DenseTensor<bool> right, DenseTensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void LessThanOrEqual(DenseTensor<bool> left, DenseTensor<bool> right, DenseTensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void Modulo(DenseTensor<bool> left, DenseTensor<bool> right, DenseTensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void Modulo(DenseTensor<bool> tensor, bool scalar, DenseTensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void Multiply(DenseTensor<bool> left, DenseTensor<bool> right, DenseTensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void Multiply(DenseTensor<bool> tensor, bool scalar, DenseTensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void NotEquals(DenseTensor<bool> left, DenseTensor<bool> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] != rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] != rightSpan[op2Index];

                }
            }
        }
        public void Or(DenseTensor<bool> left, DenseTensor<bool> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (bool)(leftSpan[i] | rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (bool)(leftSpan[op1Index] | rightSpan[op2Index]);

                }
            }
        }
        public void Or(DenseTensor<bool> tensor, bool scalar, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (bool)(tensorSpan[i] | scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (bool)(tensorSpan[op1Index] | scalar);

                }
            }
        }
        public void RightShift(DenseTensor<bool> tensor, int value, DenseTensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void Subtract(DenseTensor<bool> left, DenseTensor<bool> right, DenseTensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void Subtract(DenseTensor<bool> tensor, bool scalar, DenseTensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void UnaryMinus(DenseTensor<bool> tensor, DenseTensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void UnaryPlus(DenseTensor<bool> tensor, DenseTensor<bool> result)
        {
            throw new NotSupportedException();
        }
        public void Xor(DenseTensor<bool> left, DenseTensor<bool> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (bool)(leftSpan[i] ^ rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (bool)(leftSpan[op1Index] ^ rightSpan[op2Index]);

                }
            }
        }
        public void Xor(DenseTensor<bool> tensor, bool scalar, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (bool)(tensorSpan[i] ^ scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (bool)(tensorSpan[op1Index] ^ scalar);

                }
            }
        }
    }
    internal class ByteArithmetic : ITensorArithmetic<byte>
    {
        public byte One => 1;
        public byte Zero => 0;

        public void Add(Tensor<byte> left, Tensor<byte> right, Tensor<byte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (byte)(left[indices] + right[indices]);
            }
            
        }
        public void Add(Tensor<byte> tensor, byte scalar, Tensor<byte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (byte)(tensor[indices] + scalar);
            }
            
        }
        public void And(Tensor<byte> left, Tensor<byte> right, Tensor<byte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (byte)(left[indices] & right[indices]);
            }
            
        }
        public void And(Tensor<byte> tensor, byte scalar, Tensor<byte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (byte)(tensor[indices] & scalar);
            }
            
        }
        public void Contract(Tensor<byte> left, Tensor<byte> right, int[] leftAxes, int[] rightAxes, Tensor<byte> result)
        {
            var leftIndices = new int[left.Rank];
            var rightIndices = new int[right.Rank];
            var resultIndices = new int[result.Rank];

            var summingDimensions = new int[leftAxes.Length];
            for(int i = 0; i < leftAxes.Length; i++)
            {
                summingDimensions[i] = left.dimensions[leftAxes[i]];
            }

            var summingStrides = ArrayUtilities.GetStrides(summingDimensions);
            int summingLength = (int)ArrayUtilities.GetProduct(summingDimensions);

            var resultStrides = result.strides;

            // translates from result index to left non-summing dimensions' index portion
            // since left non-summing dimensions are given precedence in result, the end is zero-padded
            int[] leftNonSummingStrides = new int[result.Rank];

            // translates from summing index to left summing dimensions' index portion
            int[] leftSummingStrides = new int[leftAxes.Length];
            ArrayUtilities.SplitStrides(left.strides, leftAxes, leftNonSummingStrides, 0, leftSummingStrides, 0);

            // translates from result index to right non-summing dimensions' index portion
            int[] rightNonSummingStrides = new int[result.Rank];
            //  right non-summing dimensions appear after left non-summing dimensions.
            int rightNonSummingStridesOffset = (left.Rank - leftAxes.Length);

            // translates from summing index to right summing dimensions' index portion
            int[] rightSummingStrides = new int[rightAxes.Length];
            ArrayUtilities.SplitStrides(right.strides, rightAxes, rightNonSummingStrides, rightNonSummingStridesOffset, rightSummingStrides, 0);

            for (int resultIndex = 0; resultIndex < result.Length; resultIndex++)
            {
                byte sum = (byte)0;
                
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, resultIndex, resultIndices);

                int leftIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, leftNonSummingStrides);
                int rightIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, rightNonSummingStrides);

                for (int summingIndex = 0; summingIndex < summingLength; summingIndex++)
                {
                    int leftIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, leftSummingStrides);
                    int rightIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, rightSummingStrides);

                    int leftIndex = leftIndexNonSumming + leftIndexSumming;
                    int rightIndex = rightIndexNonSumming + rightIndexSumming;

                    // todo, make this more efficient
                    ArrayUtilities.GetIndices(left.strides, left.IsReversedStride, leftIndex, leftIndices);
                    ArrayUtilities.GetIndices(right.strides, right.IsReversedStride, rightIndex, rightIndices);

                    sum += (byte)(left[leftIndices] * right[rightIndices]);
                }
                
                result[resultIndices] = sum;
            }
        }
        public void Decrement(Tensor<byte> tensor, Tensor<byte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices]--;
            }
            
        }
        public void Divide(Tensor<byte> left, Tensor<byte> right, Tensor<byte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (byte)(left[indices] / right[indices]);
            }
            
        }
        public void Divide(Tensor<byte> tensor, byte scalar, Tensor<byte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (byte)(tensor[indices] / scalar);
            }
            
        }
        public void Equals(Tensor<byte> left, Tensor<byte> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] == right[indices];
            }
            
        }
        public void GreaterThan(Tensor<byte> left, Tensor<byte> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] > right[indices];
            }
            
        }
        public void GreaterThanOrEqual(Tensor<byte> left, Tensor<byte> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] >= right[indices];
            }
            
        }
        public void Increment(Tensor<byte> tensor, Tensor<byte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices]++;
            }
            
        }
        public void LeftShift(Tensor<byte> tensor, int value, Tensor<byte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (byte)(tensor[indices] << value);
            }
            
        }
        public void LessThan(Tensor<byte> left, Tensor<byte> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] < right[indices];
            }
            
        }
        public void LessThanOrEqual(Tensor<byte> left, Tensor<byte> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] <= right[indices];
            }
            
        }
        public void Modulo(Tensor<byte> left, Tensor<byte> right, Tensor<byte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (byte)(left[indices] % right[indices]);
            }
            
        }
        public void Modulo(Tensor<byte> tensor, byte scalar, Tensor<byte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (byte)(tensor[indices] % scalar);
            }
            
        }
        public void Multiply(Tensor<byte> left, Tensor<byte> right, Tensor<byte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (byte)(left[indices] * right[indices]);
            }
            
        }
        public void Multiply(Tensor<byte> tensor, byte scalar, Tensor<byte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (byte)(tensor[indices] * scalar);
            }
            
        }
        public void NotEquals(Tensor<byte> left, Tensor<byte> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] != right[indices];
            }
            
        }
        public void Or(Tensor<byte> left, Tensor<byte> right, Tensor<byte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (byte)(left[indices] | right[indices]);
            }
            
        }
        public void Or(Tensor<byte> tensor, byte scalar, Tensor<byte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (byte)(tensor[indices] | scalar);
            }
            
        }
        public void RightShift(Tensor<byte> tensor, int value, Tensor<byte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (byte)(tensor[indices] >> value);
            }
            
        }
        public void Subtract(Tensor<byte> left, Tensor<byte> right, Tensor<byte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (byte)(left[indices] - right[indices]);
            }
            
        }
        public void Subtract(Tensor<byte> tensor, byte scalar, Tensor<byte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (byte)(tensor[indices] - scalar);
            }
            
        }
        public void UnaryMinus(Tensor<byte> tensor, Tensor<byte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (byte)-tensor[indices];
            }
            
        }
        public void UnaryPlus(Tensor<byte> tensor, Tensor<byte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (byte)+tensor[indices];
            }
            
        }
        public void Xor(Tensor<byte> left, Tensor<byte> right, Tensor<byte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (byte)(left[indices] ^ right[indices]);
            }
            
        }
        public void Xor(Tensor<byte> tensor, byte scalar, Tensor<byte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (byte)(tensor[indices] ^ scalar);
            }
            
        }

        public void Add(DenseTensor<byte> left, DenseTensor<byte> right, DenseTensor<byte> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (byte)(leftSpan[i] + rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (byte)(leftSpan[op1Index] + rightSpan[op2Index]);

                }
            }
        }
        public void Add(DenseTensor<byte> tensor, byte scalar, DenseTensor<byte> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (byte)(tensorSpan[i] + scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (byte)(tensorSpan[op1Index] + scalar);

                }
            }
        }
        public void And(DenseTensor<byte> left, DenseTensor<byte> right, DenseTensor<byte> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (byte)(leftSpan[i] & rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (byte)(leftSpan[op1Index] & rightSpan[op2Index]);

                }
            }
        }
        public void And(DenseTensor<byte> tensor, byte scalar, DenseTensor<byte> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (byte)(tensorSpan[i] & scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (byte)(tensorSpan[op1Index] & scalar);

                }
            }
        }
        public void Contract(DenseTensor<byte> left, DenseTensor<byte> right, int[] leftAxes, int[] rightAxes, DenseTensor<byte> result)
        {
            var summingDimensions = new int[leftAxes.Length];
            for(int i = 0; i < leftAxes.Length; i++)
            {
                summingDimensions[i] = left.dimensions[leftAxes[i]];
            }

            var summingStrides = ArrayUtilities.GetStrides(summingDimensions);
            int summingLength = (int)ArrayUtilities.GetProduct(summingDimensions);

            var resultStrides = result.strides;

            // translates from result index to left non-summing dimensions' index portion
            // since left non-summing dimensions are given precedence in result, the end is zero-padded
            int[] leftNonSummingStrides = new int[result.Rank];

            // translates from summing index to left summing dimensions' index portion
            int[] leftSummingStrides = new int[leftAxes.Length];
            ArrayUtilities.SplitStrides(left.strides, leftAxes, leftNonSummingStrides, 0, leftSummingStrides, 0);

            // translates from result index to right non-summing dimensions' index portion
            int[] rightNonSummingStrides = new int[result.Rank];
            //  right non-summing dimensions appear after left non-summing dimensions.
            int rightNonSummingStridesOffset = (left.Rank - leftAxes.Length);

            // translates from summing index to right summing dimensions' index portion
            int[] rightSummingStrides = new int[rightAxes.Length];
            ArrayUtilities.SplitStrides(right.strides, rightAxes, rightNonSummingStrides, rightNonSummingStridesOffset, rightSummingStrides, 0);
            
            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;

            for (int resultIndex = 0; resultIndex < resultSpan.Length; resultIndex++)
            {
                byte sum = (byte)0;

                int leftIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, leftNonSummingStrides);
                int rightIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, rightNonSummingStrides);

                for (int summingIndex = 0; summingIndex < summingLength; summingIndex++)
                {
                    int leftIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, leftSummingStrides);
                    int rightIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, rightSummingStrides);

                    int leftIndex = leftIndexNonSumming + leftIndexSumming;
                    int rightIndex = rightIndexNonSumming + rightIndexSumming;

                    sum += (byte)(leftSpan[leftIndex] * rightSpan[rightIndex]);
                }

                resultSpan[resultIndex] = sum;
            }
        }
        public void Decrement(DenseTensor<byte> tensor, DenseTensor<byte> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            for(int i = 0; i < resultSpan.Length; i++)
            {
                resultSpan[i]--;
            }
        }
        public void Divide(DenseTensor<byte> left, DenseTensor<byte> right, DenseTensor<byte> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (byte)(leftSpan[i] / rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (byte)(leftSpan[op1Index] / rightSpan[op2Index]);

                }
            }
        }
        public void Divide(DenseTensor<byte> tensor, byte scalar, DenseTensor<byte> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (byte)(tensorSpan[i] / scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (byte)(tensorSpan[op1Index] / scalar);

                }
            }
        }
        public void Equals(DenseTensor<byte> left, DenseTensor<byte> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] == rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] == rightSpan[op2Index];

                }
            }
        }
        public void GreaterThan(DenseTensor<byte> left, DenseTensor<byte> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] > rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] > rightSpan[op2Index];

                }
            }
        }
        public void GreaterThanOrEqual(DenseTensor<byte> left, DenseTensor<byte> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] >= rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] >= rightSpan[op2Index];

                }
            }
        }
        public void Increment(DenseTensor<byte> tensor, DenseTensor<byte> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            for(int i = 0; i < resultSpan.Length; i++)
            {
                resultSpan[i]++;
            }
        }
        public void LeftShift(DenseTensor<byte> tensor, int value, DenseTensor<byte> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (byte)(tensorSpan[i] << value);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (byte)(tensorSpan[op1Index] << value);

                }
            }
        }
        public void LessThan(DenseTensor<byte> left, DenseTensor<byte> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] < rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] < rightSpan[op2Index];

                }
            }
        }
        public void LessThanOrEqual(DenseTensor<byte> left, DenseTensor<byte> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] <= rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] <= rightSpan[op2Index];

                }
            }
        }
        public void Modulo(DenseTensor<byte> left, DenseTensor<byte> right, DenseTensor<byte> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (byte)(leftSpan[i] % rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (byte)(leftSpan[op1Index] % rightSpan[op2Index]);

                }
            }
        }
        public void Modulo(DenseTensor<byte> tensor, byte scalar, DenseTensor<byte> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (byte)(tensorSpan[i] % scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (byte)(tensorSpan[op1Index] % scalar);

                }
            }
        }
        public void Multiply(DenseTensor<byte> left, DenseTensor<byte> right, DenseTensor<byte> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (byte)(leftSpan[i] * rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (byte)(leftSpan[op1Index] * rightSpan[op2Index]);

                }
            }
        }
        public void Multiply(DenseTensor<byte> tensor, byte scalar, DenseTensor<byte> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (byte)(tensorSpan[i] * scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (byte)(tensorSpan[op1Index] * scalar);

                }
            }
        }
        public void NotEquals(DenseTensor<byte> left, DenseTensor<byte> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] != rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] != rightSpan[op2Index];

                }
            }
        }
        public void Or(DenseTensor<byte> left, DenseTensor<byte> right, DenseTensor<byte> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (byte)(leftSpan[i] | rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (byte)(leftSpan[op1Index] | rightSpan[op2Index]);

                }
            }
        }
        public void Or(DenseTensor<byte> tensor, byte scalar, DenseTensor<byte> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (byte)(tensorSpan[i] | scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (byte)(tensorSpan[op1Index] | scalar);

                }
            }
        }
        public void RightShift(DenseTensor<byte> tensor, int value, DenseTensor<byte> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (byte)(tensorSpan[i] >> value);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (byte)(tensorSpan[op1Index] >> value);

                }
            }
        }
        public void Subtract(DenseTensor<byte> left, DenseTensor<byte> right, DenseTensor<byte> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (byte)(leftSpan[i] - rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (byte)(leftSpan[op1Index] - rightSpan[op2Index]);

                }
            }
        }
        public void Subtract(DenseTensor<byte> tensor, byte scalar, DenseTensor<byte> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (byte)(tensorSpan[i] - scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (byte)(tensorSpan[op1Index] - scalar);

                }
            }
        }
        public void UnaryMinus(DenseTensor<byte> tensor, DenseTensor<byte> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (byte)-tensorSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (byte)-tensorSpan[op1Index];

                }
            }
        }
        public void UnaryPlus(DenseTensor<byte> tensor, DenseTensor<byte> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (byte)+tensorSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (byte)+tensorSpan[op1Index];

                }
            }
        }
        public void Xor(DenseTensor<byte> left, DenseTensor<byte> right, DenseTensor<byte> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (byte)(leftSpan[i] ^ rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (byte)(leftSpan[op1Index] ^ rightSpan[op2Index]);

                }
            }
        }
        public void Xor(DenseTensor<byte> tensor, byte scalar, DenseTensor<byte> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (byte)(tensorSpan[i] ^ scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (byte)(tensorSpan[op1Index] ^ scalar);

                }
            }
        }
    }
    internal class CharArithmetic : ITensorArithmetic<char>
    {
        public char One => (char)1;
        public char Zero => (char)0;

        public void Add(Tensor<char> left, Tensor<char> right, Tensor<char> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (char)(left[indices] + right[indices]);
            }
            
        }
        public void Add(Tensor<char> tensor, char scalar, Tensor<char> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (char)(tensor[indices] + scalar);
            }
            
        }
        public void And(Tensor<char> left, Tensor<char> right, Tensor<char> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (char)(left[indices] & right[indices]);
            }
            
        }
        public void And(Tensor<char> tensor, char scalar, Tensor<char> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (char)(tensor[indices] & scalar);
            }
            
        }
        public void Contract(Tensor<char> left, Tensor<char> right, int[] leftAxes, int[] rightAxes, Tensor<char> result)
        {
            var leftIndices = new int[left.Rank];
            var rightIndices = new int[right.Rank];
            var resultIndices = new int[result.Rank];

            var summingDimensions = new int[leftAxes.Length];
            for(int i = 0; i < leftAxes.Length; i++)
            {
                summingDimensions[i] = left.dimensions[leftAxes[i]];
            }

            var summingStrides = ArrayUtilities.GetStrides(summingDimensions);
            int summingLength = (int)ArrayUtilities.GetProduct(summingDimensions);

            var resultStrides = result.strides;

            // translates from result index to left non-summing dimensions' index portion
            // since left non-summing dimensions are given precedence in result, the end is zero-padded
            int[] leftNonSummingStrides = new int[result.Rank];

            // translates from summing index to left summing dimensions' index portion
            int[] leftSummingStrides = new int[leftAxes.Length];
            ArrayUtilities.SplitStrides(left.strides, leftAxes, leftNonSummingStrides, 0, leftSummingStrides, 0);

            // translates from result index to right non-summing dimensions' index portion
            int[] rightNonSummingStrides = new int[result.Rank];
            //  right non-summing dimensions appear after left non-summing dimensions.
            int rightNonSummingStridesOffset = (left.Rank - leftAxes.Length);

            // translates from summing index to right summing dimensions' index portion
            int[] rightSummingStrides = new int[rightAxes.Length];
            ArrayUtilities.SplitStrides(right.strides, rightAxes, rightNonSummingStrides, rightNonSummingStridesOffset, rightSummingStrides, 0);

            for (int resultIndex = 0; resultIndex < result.Length; resultIndex++)
            {
                char sum = (char)0;
                
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, resultIndex, resultIndices);

                int leftIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, leftNonSummingStrides);
                int rightIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, rightNonSummingStrides);

                for (int summingIndex = 0; summingIndex < summingLength; summingIndex++)
                {
                    int leftIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, leftSummingStrides);
                    int rightIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, rightSummingStrides);

                    int leftIndex = leftIndexNonSumming + leftIndexSumming;
                    int rightIndex = rightIndexNonSumming + rightIndexSumming;

                    // todo, make this more efficient
                    ArrayUtilities.GetIndices(left.strides, left.IsReversedStride, leftIndex, leftIndices);
                    ArrayUtilities.GetIndices(right.strides, right.IsReversedStride, rightIndex, rightIndices);

                    sum += (char)(left[leftIndices] * right[rightIndices]);
                }
                
                result[resultIndices] = sum;
            }
        }
        public void Decrement(Tensor<char> tensor, Tensor<char> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices]--;
            }
            
        }
        public void Divide(Tensor<char> left, Tensor<char> right, Tensor<char> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (char)(left[indices] / right[indices]);
            }
            
        }
        public void Divide(Tensor<char> tensor, char scalar, Tensor<char> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (char)(tensor[indices] / scalar);
            }
            
        }
        public void Equals(Tensor<char> left, Tensor<char> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] == right[indices];
            }
            
        }
        public void GreaterThan(Tensor<char> left, Tensor<char> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] > right[indices];
            }
            
        }
        public void GreaterThanOrEqual(Tensor<char> left, Tensor<char> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] >= right[indices];
            }
            
        }
        public void Increment(Tensor<char> tensor, Tensor<char> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices]++;
            }
            
        }
        public void LeftShift(Tensor<char> tensor, int value, Tensor<char> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (char)(tensor[indices] << value);
            }
            
        }
        public void LessThan(Tensor<char> left, Tensor<char> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] < right[indices];
            }
            
        }
        public void LessThanOrEqual(Tensor<char> left, Tensor<char> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] <= right[indices];
            }
            
        }
        public void Modulo(Tensor<char> left, Tensor<char> right, Tensor<char> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (char)(left[indices] % right[indices]);
            }
            
        }
        public void Modulo(Tensor<char> tensor, char scalar, Tensor<char> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (char)(tensor[indices] % scalar);
            }
            
        }
        public void Multiply(Tensor<char> left, Tensor<char> right, Tensor<char> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (char)(left[indices] * right[indices]);
            }
            
        }
        public void Multiply(Tensor<char> tensor, char scalar, Tensor<char> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (char)(tensor[indices] * scalar);
            }
            
        }
        public void NotEquals(Tensor<char> left, Tensor<char> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] != right[indices];
            }
            
        }
        public void Or(Tensor<char> left, Tensor<char> right, Tensor<char> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (char)(left[indices] | right[indices]);
            }
            
        }
        public void Or(Tensor<char> tensor, char scalar, Tensor<char> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (char)(tensor[indices] | scalar);
            }
            
        }
        public void RightShift(Tensor<char> tensor, int value, Tensor<char> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (char)(tensor[indices] >> value);
            }
            
        }
        public void Subtract(Tensor<char> left, Tensor<char> right, Tensor<char> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (char)(left[indices] - right[indices]);
            }
            
        }
        public void Subtract(Tensor<char> tensor, char scalar, Tensor<char> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (char)(tensor[indices] - scalar);
            }
            
        }
        public void UnaryMinus(Tensor<char> tensor, Tensor<char> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (char)-tensor[indices];
            }
            
        }
        public void UnaryPlus(Tensor<char> tensor, Tensor<char> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (char)+tensor[indices];
            }
            
        }
        public void Xor(Tensor<char> left, Tensor<char> right, Tensor<char> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (char)(left[indices] ^ right[indices]);
            }
            
        }
        public void Xor(Tensor<char> tensor, char scalar, Tensor<char> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (char)(tensor[indices] ^ scalar);
            }
            
        }

        public void Add(DenseTensor<char> left, DenseTensor<char> right, DenseTensor<char> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (char)(leftSpan[i] + rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (char)(leftSpan[op1Index] + rightSpan[op2Index]);

                }
            }
        }
        public void Add(DenseTensor<char> tensor, char scalar, DenseTensor<char> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (char)(tensorSpan[i] + scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (char)(tensorSpan[op1Index] + scalar);

                }
            }
        }
        public void And(DenseTensor<char> left, DenseTensor<char> right, DenseTensor<char> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (char)(leftSpan[i] & rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (char)(leftSpan[op1Index] & rightSpan[op2Index]);

                }
            }
        }
        public void And(DenseTensor<char> tensor, char scalar, DenseTensor<char> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (char)(tensorSpan[i] & scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (char)(tensorSpan[op1Index] & scalar);

                }
            }
        }
        public void Contract(DenseTensor<char> left, DenseTensor<char> right, int[] leftAxes, int[] rightAxes, DenseTensor<char> result)
        {
            var summingDimensions = new int[leftAxes.Length];
            for(int i = 0; i < leftAxes.Length; i++)
            {
                summingDimensions[i] = left.dimensions[leftAxes[i]];
            }

            var summingStrides = ArrayUtilities.GetStrides(summingDimensions);
            int summingLength = (int)ArrayUtilities.GetProduct(summingDimensions);

            var resultStrides = result.strides;

            // translates from result index to left non-summing dimensions' index portion
            // since left non-summing dimensions are given precedence in result, the end is zero-padded
            int[] leftNonSummingStrides = new int[result.Rank];

            // translates from summing index to left summing dimensions' index portion
            int[] leftSummingStrides = new int[leftAxes.Length];
            ArrayUtilities.SplitStrides(left.strides, leftAxes, leftNonSummingStrides, 0, leftSummingStrides, 0);

            // translates from result index to right non-summing dimensions' index portion
            int[] rightNonSummingStrides = new int[result.Rank];
            //  right non-summing dimensions appear after left non-summing dimensions.
            int rightNonSummingStridesOffset = (left.Rank - leftAxes.Length);

            // translates from summing index to right summing dimensions' index portion
            int[] rightSummingStrides = new int[rightAxes.Length];
            ArrayUtilities.SplitStrides(right.strides, rightAxes, rightNonSummingStrides, rightNonSummingStridesOffset, rightSummingStrides, 0);
            
            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;

            for (int resultIndex = 0; resultIndex < resultSpan.Length; resultIndex++)
            {
                char sum = (char)0;

                int leftIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, leftNonSummingStrides);
                int rightIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, rightNonSummingStrides);

                for (int summingIndex = 0; summingIndex < summingLength; summingIndex++)
                {
                    int leftIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, leftSummingStrides);
                    int rightIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, rightSummingStrides);

                    int leftIndex = leftIndexNonSumming + leftIndexSumming;
                    int rightIndex = rightIndexNonSumming + rightIndexSumming;

                    sum += (char)(leftSpan[leftIndex] * rightSpan[rightIndex]);
                }

                resultSpan[resultIndex] = sum;
            }
        }
        public void Decrement(DenseTensor<char> tensor, DenseTensor<char> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            for(int i = 0; i < resultSpan.Length; i++)
            {
                resultSpan[i]--;
            }
        }
        public void Divide(DenseTensor<char> left, DenseTensor<char> right, DenseTensor<char> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (char)(leftSpan[i] / rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (char)(leftSpan[op1Index] / rightSpan[op2Index]);

                }
            }
        }
        public void Divide(DenseTensor<char> tensor, char scalar, DenseTensor<char> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (char)(tensorSpan[i] / scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (char)(tensorSpan[op1Index] / scalar);

                }
            }
        }
        public void Equals(DenseTensor<char> left, DenseTensor<char> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] == rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] == rightSpan[op2Index];

                }
            }
        }
        public void GreaterThan(DenseTensor<char> left, DenseTensor<char> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] > rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] > rightSpan[op2Index];

                }
            }
        }
        public void GreaterThanOrEqual(DenseTensor<char> left, DenseTensor<char> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] >= rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] >= rightSpan[op2Index];

                }
            }
        }
        public void Increment(DenseTensor<char> tensor, DenseTensor<char> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            for(int i = 0; i < resultSpan.Length; i++)
            {
                resultSpan[i]++;
            }
        }
        public void LeftShift(DenseTensor<char> tensor, int value, DenseTensor<char> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (char)(tensorSpan[i] << value);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (char)(tensorSpan[op1Index] << value);

                }
            }
        }
        public void LessThan(DenseTensor<char> left, DenseTensor<char> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] < rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] < rightSpan[op2Index];

                }
            }
        }
        public void LessThanOrEqual(DenseTensor<char> left, DenseTensor<char> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] <= rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] <= rightSpan[op2Index];

                }
            }
        }
        public void Modulo(DenseTensor<char> left, DenseTensor<char> right, DenseTensor<char> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (char)(leftSpan[i] % rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (char)(leftSpan[op1Index] % rightSpan[op2Index]);

                }
            }
        }
        public void Modulo(DenseTensor<char> tensor, char scalar, DenseTensor<char> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (char)(tensorSpan[i] % scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (char)(tensorSpan[op1Index] % scalar);

                }
            }
        }
        public void Multiply(DenseTensor<char> left, DenseTensor<char> right, DenseTensor<char> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (char)(leftSpan[i] * rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (char)(leftSpan[op1Index] * rightSpan[op2Index]);

                }
            }
        }
        public void Multiply(DenseTensor<char> tensor, char scalar, DenseTensor<char> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (char)(tensorSpan[i] * scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (char)(tensorSpan[op1Index] * scalar);

                }
            }
        }
        public void NotEquals(DenseTensor<char> left, DenseTensor<char> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] != rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] != rightSpan[op2Index];

                }
            }
        }
        public void Or(DenseTensor<char> left, DenseTensor<char> right, DenseTensor<char> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (char)(leftSpan[i] | rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (char)(leftSpan[op1Index] | rightSpan[op2Index]);

                }
            }
        }
        public void Or(DenseTensor<char> tensor, char scalar, DenseTensor<char> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (char)(tensorSpan[i] | scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (char)(tensorSpan[op1Index] | scalar);

                }
            }
        }
        public void RightShift(DenseTensor<char> tensor, int value, DenseTensor<char> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (char)(tensorSpan[i] >> value);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (char)(tensorSpan[op1Index] >> value);

                }
            }
        }
        public void Subtract(DenseTensor<char> left, DenseTensor<char> right, DenseTensor<char> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (char)(leftSpan[i] - rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (char)(leftSpan[op1Index] - rightSpan[op2Index]);

                }
            }
        }
        public void Subtract(DenseTensor<char> tensor, char scalar, DenseTensor<char> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (char)(tensorSpan[i] - scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (char)(tensorSpan[op1Index] - scalar);

                }
            }
        }
        public void UnaryMinus(DenseTensor<char> tensor, DenseTensor<char> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (char)-tensorSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (char)-tensorSpan[op1Index];

                }
            }
        }
        public void UnaryPlus(DenseTensor<char> tensor, DenseTensor<char> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (char)+tensorSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (char)+tensorSpan[op1Index];

                }
            }
        }
        public void Xor(DenseTensor<char> left, DenseTensor<char> right, DenseTensor<char> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (char)(leftSpan[i] ^ rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (char)(leftSpan[op1Index] ^ rightSpan[op2Index]);

                }
            }
        }
        public void Xor(DenseTensor<char> tensor, char scalar, DenseTensor<char> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (char)(tensorSpan[i] ^ scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (char)(tensorSpan[op1Index] ^ scalar);

                }
            }
        }
    }
    internal class DecimalArithmetic : ITensorArithmetic<decimal>
    {
        public decimal One => 1;
        public decimal Zero => 0;

        public void Add(Tensor<decimal> left, Tensor<decimal> right, Tensor<decimal> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (decimal)(left[indices] + right[indices]);
            }
            
        }
        public void Add(Tensor<decimal> tensor, decimal scalar, Tensor<decimal> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (decimal)(tensor[indices] + scalar);
            }
            
        }
        public void And(Tensor<decimal> left, Tensor<decimal> right, Tensor<decimal> result)
        {
            throw new NotSupportedException();
        }
        public void And(Tensor<decimal> tensor, decimal scalar, Tensor<decimal> result)
        {
            throw new NotSupportedException();
        }
        public void Contract(Tensor<decimal> left, Tensor<decimal> right, int[] leftAxes, int[] rightAxes, Tensor<decimal> result)
        {
            var leftIndices = new int[left.Rank];
            var rightIndices = new int[right.Rank];
            var resultIndices = new int[result.Rank];

            var summingDimensions = new int[leftAxes.Length];
            for(int i = 0; i < leftAxes.Length; i++)
            {
                summingDimensions[i] = left.dimensions[leftAxes[i]];
            }

            var summingStrides = ArrayUtilities.GetStrides(summingDimensions);
            int summingLength = (int)ArrayUtilities.GetProduct(summingDimensions);

            var resultStrides = result.strides;

            // translates from result index to left non-summing dimensions' index portion
            // since left non-summing dimensions are given precedence in result, the end is zero-padded
            int[] leftNonSummingStrides = new int[result.Rank];

            // translates from summing index to left summing dimensions' index portion
            int[] leftSummingStrides = new int[leftAxes.Length];
            ArrayUtilities.SplitStrides(left.strides, leftAxes, leftNonSummingStrides, 0, leftSummingStrides, 0);

            // translates from result index to right non-summing dimensions' index portion
            int[] rightNonSummingStrides = new int[result.Rank];
            //  right non-summing dimensions appear after left non-summing dimensions.
            int rightNonSummingStridesOffset = (left.Rank - leftAxes.Length);

            // translates from summing index to right summing dimensions' index portion
            int[] rightSummingStrides = new int[rightAxes.Length];
            ArrayUtilities.SplitStrides(right.strides, rightAxes, rightNonSummingStrides, rightNonSummingStridesOffset, rightSummingStrides, 0);

            for (int resultIndex = 0; resultIndex < result.Length; resultIndex++)
            {
                decimal sum = (decimal)0;
                
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, resultIndex, resultIndices);

                int leftIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, leftNonSummingStrides);
                int rightIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, rightNonSummingStrides);

                for (int summingIndex = 0; summingIndex < summingLength; summingIndex++)
                {
                    int leftIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, leftSummingStrides);
                    int rightIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, rightSummingStrides);

                    int leftIndex = leftIndexNonSumming + leftIndexSumming;
                    int rightIndex = rightIndexNonSumming + rightIndexSumming;

                    // todo, make this more efficient
                    ArrayUtilities.GetIndices(left.strides, left.IsReversedStride, leftIndex, leftIndices);
                    ArrayUtilities.GetIndices(right.strides, right.IsReversedStride, rightIndex, rightIndices);

                    sum += (decimal)(left[leftIndices] * right[rightIndices]);
                }
                
                result[resultIndices] = sum;
            }
        }
        public void Decrement(Tensor<decimal> tensor, Tensor<decimal> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices]--;
            }
            
        }
        public void Divide(Tensor<decimal> left, Tensor<decimal> right, Tensor<decimal> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (decimal)(left[indices] / right[indices]);
            }
            
        }
        public void Divide(Tensor<decimal> tensor, decimal scalar, Tensor<decimal> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (decimal)(tensor[indices] / scalar);
            }
            
        }
        public void Equals(Tensor<decimal> left, Tensor<decimal> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] == right[indices];
            }
            
        }
        public void GreaterThan(Tensor<decimal> left, Tensor<decimal> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] > right[indices];
            }
            
        }
        public void GreaterThanOrEqual(Tensor<decimal> left, Tensor<decimal> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] >= right[indices];
            }
            
        }
        public void Increment(Tensor<decimal> tensor, Tensor<decimal> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices]++;
            }
            
        }
        public void LeftShift(Tensor<decimal> tensor, int value, Tensor<decimal> result)
        {
            throw new NotSupportedException();
        }
        public void LessThan(Tensor<decimal> left, Tensor<decimal> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] < right[indices];
            }
            
        }
        public void LessThanOrEqual(Tensor<decimal> left, Tensor<decimal> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] <= right[indices];
            }
            
        }
        public void Modulo(Tensor<decimal> left, Tensor<decimal> right, Tensor<decimal> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (decimal)(left[indices] % right[indices]);
            }
            
        }
        public void Modulo(Tensor<decimal> tensor, decimal scalar, Tensor<decimal> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (decimal)(tensor[indices] % scalar);
            }
            
        }
        public void Multiply(Tensor<decimal> left, Tensor<decimal> right, Tensor<decimal> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (decimal)(left[indices] * right[indices]);
            }
            
        }
        public void Multiply(Tensor<decimal> tensor, decimal scalar, Tensor<decimal> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (decimal)(tensor[indices] * scalar);
            }
            
        }
        public void NotEquals(Tensor<decimal> left, Tensor<decimal> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] != right[indices];
            }
            
        }
        public void Or(Tensor<decimal> left, Tensor<decimal> right, Tensor<decimal> result)
        {
            throw new NotSupportedException();
        }
        public void Or(Tensor<decimal> tensor, decimal scalar, Tensor<decimal> result)
        {
            throw new NotSupportedException();
        }
        public void RightShift(Tensor<decimal> tensor, int value, Tensor<decimal> result)
        {
            throw new NotSupportedException();
        }
        public void Subtract(Tensor<decimal> left, Tensor<decimal> right, Tensor<decimal> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (decimal)(left[indices] - right[indices]);
            }
            
        }
        public void Subtract(Tensor<decimal> tensor, decimal scalar, Tensor<decimal> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (decimal)(tensor[indices] - scalar);
            }
            
        }
        public void UnaryMinus(Tensor<decimal> tensor, Tensor<decimal> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (decimal)-tensor[indices];
            }
            
        }
        public void UnaryPlus(Tensor<decimal> tensor, Tensor<decimal> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (decimal)+tensor[indices];
            }
            
        }
        public void Xor(Tensor<decimal> left, Tensor<decimal> right, Tensor<decimal> result)
        {
            throw new NotSupportedException();
        }
        public void Xor(Tensor<decimal> tensor, decimal scalar, Tensor<decimal> result)
        {
            throw new NotSupportedException();
        }

        public void Add(DenseTensor<decimal> left, DenseTensor<decimal> right, DenseTensor<decimal> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (decimal)(leftSpan[i] + rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (decimal)(leftSpan[op1Index] + rightSpan[op2Index]);

                }
            }
        }
        public void Add(DenseTensor<decimal> tensor, decimal scalar, DenseTensor<decimal> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (decimal)(tensorSpan[i] + scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (decimal)(tensorSpan[op1Index] + scalar);

                }
            }
        }
        public void And(DenseTensor<decimal> left, DenseTensor<decimal> right, DenseTensor<decimal> result)
        {
            throw new NotSupportedException();
        }
        public void And(DenseTensor<decimal> tensor, decimal scalar, DenseTensor<decimal> result)
        {
            throw new NotSupportedException();
        }
        public void Contract(DenseTensor<decimal> left, DenseTensor<decimal> right, int[] leftAxes, int[] rightAxes, DenseTensor<decimal> result)
        {
            var summingDimensions = new int[leftAxes.Length];
            for(int i = 0; i < leftAxes.Length; i++)
            {
                summingDimensions[i] = left.dimensions[leftAxes[i]];
            }

            var summingStrides = ArrayUtilities.GetStrides(summingDimensions);
            int summingLength = (int)ArrayUtilities.GetProduct(summingDimensions);

            var resultStrides = result.strides;

            // translates from result index to left non-summing dimensions' index portion
            // since left non-summing dimensions are given precedence in result, the end is zero-padded
            int[] leftNonSummingStrides = new int[result.Rank];

            // translates from summing index to left summing dimensions' index portion
            int[] leftSummingStrides = new int[leftAxes.Length];
            ArrayUtilities.SplitStrides(left.strides, leftAxes, leftNonSummingStrides, 0, leftSummingStrides, 0);

            // translates from result index to right non-summing dimensions' index portion
            int[] rightNonSummingStrides = new int[result.Rank];
            //  right non-summing dimensions appear after left non-summing dimensions.
            int rightNonSummingStridesOffset = (left.Rank - leftAxes.Length);

            // translates from summing index to right summing dimensions' index portion
            int[] rightSummingStrides = new int[rightAxes.Length];
            ArrayUtilities.SplitStrides(right.strides, rightAxes, rightNonSummingStrides, rightNonSummingStridesOffset, rightSummingStrides, 0);
            
            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;

            for (int resultIndex = 0; resultIndex < resultSpan.Length; resultIndex++)
            {
                decimal sum = (decimal)0;

                int leftIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, leftNonSummingStrides);
                int rightIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, rightNonSummingStrides);

                for (int summingIndex = 0; summingIndex < summingLength; summingIndex++)
                {
                    int leftIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, leftSummingStrides);
                    int rightIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, rightSummingStrides);

                    int leftIndex = leftIndexNonSumming + leftIndexSumming;
                    int rightIndex = rightIndexNonSumming + rightIndexSumming;

                    sum += (decimal)(leftSpan[leftIndex] * rightSpan[rightIndex]);
                }

                resultSpan[resultIndex] = sum;
            }
        }
        public void Decrement(DenseTensor<decimal> tensor, DenseTensor<decimal> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            for(int i = 0; i < resultSpan.Length; i++)
            {
                resultSpan[i]--;
            }
        }
        public void Divide(DenseTensor<decimal> left, DenseTensor<decimal> right, DenseTensor<decimal> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (decimal)(leftSpan[i] / rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (decimal)(leftSpan[op1Index] / rightSpan[op2Index]);

                }
            }
        }
        public void Divide(DenseTensor<decimal> tensor, decimal scalar, DenseTensor<decimal> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (decimal)(tensorSpan[i] / scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (decimal)(tensorSpan[op1Index] / scalar);

                }
            }
        }
        public void Equals(DenseTensor<decimal> left, DenseTensor<decimal> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] == rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] == rightSpan[op2Index];

                }
            }
        }
        public void GreaterThan(DenseTensor<decimal> left, DenseTensor<decimal> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] > rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] > rightSpan[op2Index];

                }
            }
        }
        public void GreaterThanOrEqual(DenseTensor<decimal> left, DenseTensor<decimal> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] >= rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] >= rightSpan[op2Index];

                }
            }
        }
        public void Increment(DenseTensor<decimal> tensor, DenseTensor<decimal> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            for(int i = 0; i < resultSpan.Length; i++)
            {
                resultSpan[i]++;
            }
        }
        public void LeftShift(DenseTensor<decimal> tensor, int value, DenseTensor<decimal> result)
        {
            throw new NotSupportedException();
        }
        public void LessThan(DenseTensor<decimal> left, DenseTensor<decimal> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] < rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] < rightSpan[op2Index];

                }
            }
        }
        public void LessThanOrEqual(DenseTensor<decimal> left, DenseTensor<decimal> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] <= rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] <= rightSpan[op2Index];

                }
            }
        }
        public void Modulo(DenseTensor<decimal> left, DenseTensor<decimal> right, DenseTensor<decimal> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (decimal)(leftSpan[i] % rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (decimal)(leftSpan[op1Index] % rightSpan[op2Index]);

                }
            }
        }
        public void Modulo(DenseTensor<decimal> tensor, decimal scalar, DenseTensor<decimal> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (decimal)(tensorSpan[i] % scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (decimal)(tensorSpan[op1Index] % scalar);

                }
            }
        }
        public void Multiply(DenseTensor<decimal> left, DenseTensor<decimal> right, DenseTensor<decimal> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (decimal)(leftSpan[i] * rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (decimal)(leftSpan[op1Index] * rightSpan[op2Index]);

                }
            }
        }
        public void Multiply(DenseTensor<decimal> tensor, decimal scalar, DenseTensor<decimal> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (decimal)(tensorSpan[i] * scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (decimal)(tensorSpan[op1Index] * scalar);

                }
            }
        }
        public void NotEquals(DenseTensor<decimal> left, DenseTensor<decimal> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] != rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] != rightSpan[op2Index];

                }
            }
        }
        public void Or(DenseTensor<decimal> left, DenseTensor<decimal> right, DenseTensor<decimal> result)
        {
            throw new NotSupportedException();
        }
        public void Or(DenseTensor<decimal> tensor, decimal scalar, DenseTensor<decimal> result)
        {
            throw new NotSupportedException();
        }
        public void RightShift(DenseTensor<decimal> tensor, int value, DenseTensor<decimal> result)
        {
            throw new NotSupportedException();
        }
        public void Subtract(DenseTensor<decimal> left, DenseTensor<decimal> right, DenseTensor<decimal> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (decimal)(leftSpan[i] - rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (decimal)(leftSpan[op1Index] - rightSpan[op2Index]);

                }
            }
        }
        public void Subtract(DenseTensor<decimal> tensor, decimal scalar, DenseTensor<decimal> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (decimal)(tensorSpan[i] - scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (decimal)(tensorSpan[op1Index] - scalar);

                }
            }
        }
        public void UnaryMinus(DenseTensor<decimal> tensor, DenseTensor<decimal> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (decimal)-tensorSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (decimal)-tensorSpan[op1Index];

                }
            }
        }
        public void UnaryPlus(DenseTensor<decimal> tensor, DenseTensor<decimal> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (decimal)+tensorSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (decimal)+tensorSpan[op1Index];

                }
            }
        }
        public void Xor(DenseTensor<decimal> left, DenseTensor<decimal> right, DenseTensor<decimal> result)
        {
            throw new NotSupportedException();
        }
        public void Xor(DenseTensor<decimal> tensor, decimal scalar, DenseTensor<decimal> result)
        {
            throw new NotSupportedException();
        }
    }
    internal class DoubleArithmetic : ITensorArithmetic<double>
    {
        public double One => 1.0;
        public double Zero => 0;

        public void Add(Tensor<double> left, Tensor<double> right, Tensor<double> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (double)(left[indices] + right[indices]);
            }
            
        }
        public void Add(Tensor<double> tensor, double scalar, Tensor<double> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (double)(tensor[indices] + scalar);
            }
            
        }
        public void And(Tensor<double> left, Tensor<double> right, Tensor<double> result)
        {
            throw new NotSupportedException();
        }
        public void And(Tensor<double> tensor, double scalar, Tensor<double> result)
        {
            throw new NotSupportedException();
        }
        public void Contract(Tensor<double> left, Tensor<double> right, int[] leftAxes, int[] rightAxes, Tensor<double> result)
        {
            var leftIndices = new int[left.Rank];
            var rightIndices = new int[right.Rank];
            var resultIndices = new int[result.Rank];

            var summingDimensions = new int[leftAxes.Length];
            for(int i = 0; i < leftAxes.Length; i++)
            {
                summingDimensions[i] = left.dimensions[leftAxes[i]];
            }

            var summingStrides = ArrayUtilities.GetStrides(summingDimensions);
            int summingLength = (int)ArrayUtilities.GetProduct(summingDimensions);

            var resultStrides = result.strides;

            // translates from result index to left non-summing dimensions' index portion
            // since left non-summing dimensions are given precedence in result, the end is zero-padded
            int[] leftNonSummingStrides = new int[result.Rank];

            // translates from summing index to left summing dimensions' index portion
            int[] leftSummingStrides = new int[leftAxes.Length];
            ArrayUtilities.SplitStrides(left.strides, leftAxes, leftNonSummingStrides, 0, leftSummingStrides, 0);

            // translates from result index to right non-summing dimensions' index portion
            int[] rightNonSummingStrides = new int[result.Rank];
            //  right non-summing dimensions appear after left non-summing dimensions.
            int rightNonSummingStridesOffset = (left.Rank - leftAxes.Length);

            // translates from summing index to right summing dimensions' index portion
            int[] rightSummingStrides = new int[rightAxes.Length];
            ArrayUtilities.SplitStrides(right.strides, rightAxes, rightNonSummingStrides, rightNonSummingStridesOffset, rightSummingStrides, 0);

            for (int resultIndex = 0; resultIndex < result.Length; resultIndex++)
            {
                double sum = (double)0;
                
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, resultIndex, resultIndices);

                int leftIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, leftNonSummingStrides);
                int rightIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, rightNonSummingStrides);

                for (int summingIndex = 0; summingIndex < summingLength; summingIndex++)
                {
                    int leftIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, leftSummingStrides);
                    int rightIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, rightSummingStrides);

                    int leftIndex = leftIndexNonSumming + leftIndexSumming;
                    int rightIndex = rightIndexNonSumming + rightIndexSumming;

                    // todo, make this more efficient
                    ArrayUtilities.GetIndices(left.strides, left.IsReversedStride, leftIndex, leftIndices);
                    ArrayUtilities.GetIndices(right.strides, right.IsReversedStride, rightIndex, rightIndices);

                    sum += (double)(left[leftIndices] * right[rightIndices]);
                }
                
                result[resultIndices] = sum;
            }
        }
        public void Decrement(Tensor<double> tensor, Tensor<double> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices]--;
            }
            
        }
        public void Divide(Tensor<double> left, Tensor<double> right, Tensor<double> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (double)(left[indices] / right[indices]);
            }
            
        }
        public void Divide(Tensor<double> tensor, double scalar, Tensor<double> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (double)(tensor[indices] / scalar);
            }
            
        }
        public void Equals(Tensor<double> left, Tensor<double> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] == right[indices];
            }
            
        }
        public void GreaterThan(Tensor<double> left, Tensor<double> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] > right[indices];
            }
            
        }
        public void GreaterThanOrEqual(Tensor<double> left, Tensor<double> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] >= right[indices];
            }
            
        }
        public void Increment(Tensor<double> tensor, Tensor<double> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices]++;
            }
            
        }
        public void LeftShift(Tensor<double> tensor, int value, Tensor<double> result)
        {
            throw new NotSupportedException();
        }
        public void LessThan(Tensor<double> left, Tensor<double> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] < right[indices];
            }
            
        }
        public void LessThanOrEqual(Tensor<double> left, Tensor<double> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] <= right[indices];
            }
            
        }
        public void Modulo(Tensor<double> left, Tensor<double> right, Tensor<double> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (double)(left[indices] % right[indices]);
            }
            
        }
        public void Modulo(Tensor<double> tensor, double scalar, Tensor<double> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (double)(tensor[indices] % scalar);
            }
            
        }
        public void Multiply(Tensor<double> left, Tensor<double> right, Tensor<double> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (double)(left[indices] * right[indices]);
            }
            
        }
        public void Multiply(Tensor<double> tensor, double scalar, Tensor<double> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (double)(tensor[indices] * scalar);
            }
            
        }
        public void NotEquals(Tensor<double> left, Tensor<double> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] != right[indices];
            }
            
        }
        public void Or(Tensor<double> left, Tensor<double> right, Tensor<double> result)
        {
            throw new NotSupportedException();
        }
        public void Or(Tensor<double> tensor, double scalar, Tensor<double> result)
        {
            throw new NotSupportedException();
        }
        public void RightShift(Tensor<double> tensor, int value, Tensor<double> result)
        {
            throw new NotSupportedException();
        }
        public void Subtract(Tensor<double> left, Tensor<double> right, Tensor<double> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (double)(left[indices] - right[indices]);
            }
            
        }
        public void Subtract(Tensor<double> tensor, double scalar, Tensor<double> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (double)(tensor[indices] - scalar);
            }
            
        }
        public void UnaryMinus(Tensor<double> tensor, Tensor<double> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (double)-tensor[indices];
            }
            
        }
        public void UnaryPlus(Tensor<double> tensor, Tensor<double> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (double)+tensor[indices];
            }
            
        }
        public void Xor(Tensor<double> left, Tensor<double> right, Tensor<double> result)
        {
            throw new NotSupportedException();
        }
        public void Xor(Tensor<double> tensor, double scalar, Tensor<double> result)
        {
            throw new NotSupportedException();
        }

        public void Add(DenseTensor<double> left, DenseTensor<double> right, DenseTensor<double> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (double)(leftSpan[i] + rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (double)(leftSpan[op1Index] + rightSpan[op2Index]);

                }
            }
        }
        public void Add(DenseTensor<double> tensor, double scalar, DenseTensor<double> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (double)(tensorSpan[i] + scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (double)(tensorSpan[op1Index] + scalar);

                }
            }
        }
        public void And(DenseTensor<double> left, DenseTensor<double> right, DenseTensor<double> result)
        {
            throw new NotSupportedException();
        }
        public void And(DenseTensor<double> tensor, double scalar, DenseTensor<double> result)
        {
            throw new NotSupportedException();
        }
        public void Contract(DenseTensor<double> left, DenseTensor<double> right, int[] leftAxes, int[] rightAxes, DenseTensor<double> result)
        {
            var summingDimensions = new int[leftAxes.Length];
            for(int i = 0; i < leftAxes.Length; i++)
            {
                summingDimensions[i] = left.dimensions[leftAxes[i]];
            }

            var summingStrides = ArrayUtilities.GetStrides(summingDimensions);
            int summingLength = (int)ArrayUtilities.GetProduct(summingDimensions);

            var resultStrides = result.strides;

            // translates from result index to left non-summing dimensions' index portion
            // since left non-summing dimensions are given precedence in result, the end is zero-padded
            int[] leftNonSummingStrides = new int[result.Rank];

            // translates from summing index to left summing dimensions' index portion
            int[] leftSummingStrides = new int[leftAxes.Length];
            ArrayUtilities.SplitStrides(left.strides, leftAxes, leftNonSummingStrides, 0, leftSummingStrides, 0);

            // translates from result index to right non-summing dimensions' index portion
            int[] rightNonSummingStrides = new int[result.Rank];
            //  right non-summing dimensions appear after left non-summing dimensions.
            int rightNonSummingStridesOffset = (left.Rank - leftAxes.Length);

            // translates from summing index to right summing dimensions' index portion
            int[] rightSummingStrides = new int[rightAxes.Length];
            ArrayUtilities.SplitStrides(right.strides, rightAxes, rightNonSummingStrides, rightNonSummingStridesOffset, rightSummingStrides, 0);
            
            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;

            for (int resultIndex = 0; resultIndex < resultSpan.Length; resultIndex++)
            {
                double sum = (double)0;

                int leftIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, leftNonSummingStrides);
                int rightIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, rightNonSummingStrides);

                for (int summingIndex = 0; summingIndex < summingLength; summingIndex++)
                {
                    int leftIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, leftSummingStrides);
                    int rightIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, rightSummingStrides);

                    int leftIndex = leftIndexNonSumming + leftIndexSumming;
                    int rightIndex = rightIndexNonSumming + rightIndexSumming;

                    sum += (double)(leftSpan[leftIndex] * rightSpan[rightIndex]);
                }

                resultSpan[resultIndex] = sum;
            }
        }
        public void Decrement(DenseTensor<double> tensor, DenseTensor<double> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            for(int i = 0; i < resultSpan.Length; i++)
            {
                resultSpan[i]--;
            }
        }
        public void Divide(DenseTensor<double> left, DenseTensor<double> right, DenseTensor<double> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (double)(leftSpan[i] / rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (double)(leftSpan[op1Index] / rightSpan[op2Index]);

                }
            }
        }
        public void Divide(DenseTensor<double> tensor, double scalar, DenseTensor<double> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (double)(tensorSpan[i] / scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (double)(tensorSpan[op1Index] / scalar);

                }
            }
        }
        public void Equals(DenseTensor<double> left, DenseTensor<double> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] == rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] == rightSpan[op2Index];

                }
            }
        }
        public void GreaterThan(DenseTensor<double> left, DenseTensor<double> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] > rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] > rightSpan[op2Index];

                }
            }
        }
        public void GreaterThanOrEqual(DenseTensor<double> left, DenseTensor<double> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] >= rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] >= rightSpan[op2Index];

                }
            }
        }
        public void Increment(DenseTensor<double> tensor, DenseTensor<double> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            for(int i = 0; i < resultSpan.Length; i++)
            {
                resultSpan[i]++;
            }
        }
        public void LeftShift(DenseTensor<double> tensor, int value, DenseTensor<double> result)
        {
            throw new NotSupportedException();
        }
        public void LessThan(DenseTensor<double> left, DenseTensor<double> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] < rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] < rightSpan[op2Index];

                }
            }
        }
        public void LessThanOrEqual(DenseTensor<double> left, DenseTensor<double> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] <= rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] <= rightSpan[op2Index];

                }
            }
        }
        public void Modulo(DenseTensor<double> left, DenseTensor<double> right, DenseTensor<double> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (double)(leftSpan[i] % rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (double)(leftSpan[op1Index] % rightSpan[op2Index]);

                }
            }
        }
        public void Modulo(DenseTensor<double> tensor, double scalar, DenseTensor<double> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (double)(tensorSpan[i] % scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (double)(tensorSpan[op1Index] % scalar);

                }
            }
        }
        public void Multiply(DenseTensor<double> left, DenseTensor<double> right, DenseTensor<double> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (double)(leftSpan[i] * rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (double)(leftSpan[op1Index] * rightSpan[op2Index]);

                }
            }
        }
        public void Multiply(DenseTensor<double> tensor, double scalar, DenseTensor<double> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (double)(tensorSpan[i] * scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (double)(tensorSpan[op1Index] * scalar);

                }
            }
        }
        public void NotEquals(DenseTensor<double> left, DenseTensor<double> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] != rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] != rightSpan[op2Index];

                }
            }
        }
        public void Or(DenseTensor<double> left, DenseTensor<double> right, DenseTensor<double> result)
        {
            throw new NotSupportedException();
        }
        public void Or(DenseTensor<double> tensor, double scalar, DenseTensor<double> result)
        {
            throw new NotSupportedException();
        }
        public void RightShift(DenseTensor<double> tensor, int value, DenseTensor<double> result)
        {
            throw new NotSupportedException();
        }
        public void Subtract(DenseTensor<double> left, DenseTensor<double> right, DenseTensor<double> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (double)(leftSpan[i] - rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (double)(leftSpan[op1Index] - rightSpan[op2Index]);

                }
            }
        }
        public void Subtract(DenseTensor<double> tensor, double scalar, DenseTensor<double> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (double)(tensorSpan[i] - scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (double)(tensorSpan[op1Index] - scalar);

                }
            }
        }
        public void UnaryMinus(DenseTensor<double> tensor, DenseTensor<double> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (double)-tensorSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (double)-tensorSpan[op1Index];

                }
            }
        }
        public void UnaryPlus(DenseTensor<double> tensor, DenseTensor<double> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (double)+tensorSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (double)+tensorSpan[op1Index];

                }
            }
        }
        public void Xor(DenseTensor<double> left, DenseTensor<double> right, DenseTensor<double> result)
        {
            throw new NotSupportedException();
        }
        public void Xor(DenseTensor<double> tensor, double scalar, DenseTensor<double> result)
        {
            throw new NotSupportedException();
        }
    }
    internal class FloatArithmetic : ITensorArithmetic<float>
    {
        public float One => 1.0f;
        public float Zero => 0;

        public void Add(Tensor<float> left, Tensor<float> right, Tensor<float> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (float)(left[indices] + right[indices]);
            }
            
        }
        public void Add(Tensor<float> tensor, float scalar, Tensor<float> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (float)(tensor[indices] + scalar);
            }
            
        }
        public void And(Tensor<float> left, Tensor<float> right, Tensor<float> result)
        {
            throw new NotSupportedException();
        }
        public void And(Tensor<float> tensor, float scalar, Tensor<float> result)
        {
            throw new NotSupportedException();
        }
        public void Contract(Tensor<float> left, Tensor<float> right, int[] leftAxes, int[] rightAxes, Tensor<float> result)
        {
            var leftIndices = new int[left.Rank];
            var rightIndices = new int[right.Rank];
            var resultIndices = new int[result.Rank];

            var summingDimensions = new int[leftAxes.Length];
            for(int i = 0; i < leftAxes.Length; i++)
            {
                summingDimensions[i] = left.dimensions[leftAxes[i]];
            }

            var summingStrides = ArrayUtilities.GetStrides(summingDimensions);
            int summingLength = (int)ArrayUtilities.GetProduct(summingDimensions);

            var resultStrides = result.strides;

            // translates from result index to left non-summing dimensions' index portion
            // since left non-summing dimensions are given precedence in result, the end is zero-padded
            int[] leftNonSummingStrides = new int[result.Rank];

            // translates from summing index to left summing dimensions' index portion
            int[] leftSummingStrides = new int[leftAxes.Length];
            ArrayUtilities.SplitStrides(left.strides, leftAxes, leftNonSummingStrides, 0, leftSummingStrides, 0);

            // translates from result index to right non-summing dimensions' index portion
            int[] rightNonSummingStrides = new int[result.Rank];
            //  right non-summing dimensions appear after left non-summing dimensions.
            int rightNonSummingStridesOffset = (left.Rank - leftAxes.Length);

            // translates from summing index to right summing dimensions' index portion
            int[] rightSummingStrides = new int[rightAxes.Length];
            ArrayUtilities.SplitStrides(right.strides, rightAxes, rightNonSummingStrides, rightNonSummingStridesOffset, rightSummingStrides, 0);

            for (int resultIndex = 0; resultIndex < result.Length; resultIndex++)
            {
                float sum = (float)0;
                
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, resultIndex, resultIndices);

                int leftIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, leftNonSummingStrides);
                int rightIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, rightNonSummingStrides);

                for (int summingIndex = 0; summingIndex < summingLength; summingIndex++)
                {
                    int leftIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, leftSummingStrides);
                    int rightIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, rightSummingStrides);

                    int leftIndex = leftIndexNonSumming + leftIndexSumming;
                    int rightIndex = rightIndexNonSumming + rightIndexSumming;

                    // todo, make this more efficient
                    ArrayUtilities.GetIndices(left.strides, left.IsReversedStride, leftIndex, leftIndices);
                    ArrayUtilities.GetIndices(right.strides, right.IsReversedStride, rightIndex, rightIndices);

                    sum += (float)(left[leftIndices] * right[rightIndices]);
                }
                
                result[resultIndices] = sum;
            }
        }
        public void Decrement(Tensor<float> tensor, Tensor<float> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices]--;
            }
            
        }
        public void Divide(Tensor<float> left, Tensor<float> right, Tensor<float> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (float)(left[indices] / right[indices]);
            }
            
        }
        public void Divide(Tensor<float> tensor, float scalar, Tensor<float> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (float)(tensor[indices] / scalar);
            }
            
        }
        public void Equals(Tensor<float> left, Tensor<float> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] == right[indices];
            }
            
        }
        public void GreaterThan(Tensor<float> left, Tensor<float> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] > right[indices];
            }
            
        }
        public void GreaterThanOrEqual(Tensor<float> left, Tensor<float> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] >= right[indices];
            }
            
        }
        public void Increment(Tensor<float> tensor, Tensor<float> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices]++;
            }
            
        }
        public void LeftShift(Tensor<float> tensor, int value, Tensor<float> result)
        {
            throw new NotSupportedException();
        }
        public void LessThan(Tensor<float> left, Tensor<float> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] < right[indices];
            }
            
        }
        public void LessThanOrEqual(Tensor<float> left, Tensor<float> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] <= right[indices];
            }
            
        }
        public void Modulo(Tensor<float> left, Tensor<float> right, Tensor<float> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (float)(left[indices] % right[indices]);
            }
            
        }
        public void Modulo(Tensor<float> tensor, float scalar, Tensor<float> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (float)(tensor[indices] % scalar);
            }
            
        }
        public void Multiply(Tensor<float> left, Tensor<float> right, Tensor<float> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (float)(left[indices] * right[indices]);
            }
            
        }
        public void Multiply(Tensor<float> tensor, float scalar, Tensor<float> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (float)(tensor[indices] * scalar);
            }
            
        }
        public void NotEquals(Tensor<float> left, Tensor<float> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] != right[indices];
            }
            
        }
        public void Or(Tensor<float> left, Tensor<float> right, Tensor<float> result)
        {
            throw new NotSupportedException();
        }
        public void Or(Tensor<float> tensor, float scalar, Tensor<float> result)
        {
            throw new NotSupportedException();
        }
        public void RightShift(Tensor<float> tensor, int value, Tensor<float> result)
        {
            throw new NotSupportedException();
        }
        public void Subtract(Tensor<float> left, Tensor<float> right, Tensor<float> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (float)(left[indices] - right[indices]);
            }
            
        }
        public void Subtract(Tensor<float> tensor, float scalar, Tensor<float> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (float)(tensor[indices] - scalar);
            }
            
        }
        public void UnaryMinus(Tensor<float> tensor, Tensor<float> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (float)-tensor[indices];
            }
            
        }
        public void UnaryPlus(Tensor<float> tensor, Tensor<float> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (float)+tensor[indices];
            }
            
        }
        public void Xor(Tensor<float> left, Tensor<float> right, Tensor<float> result)
        {
            throw new NotSupportedException();
        }
        public void Xor(Tensor<float> tensor, float scalar, Tensor<float> result)
        {
            throw new NotSupportedException();
        }

        public void Add(DenseTensor<float> left, DenseTensor<float> right, DenseTensor<float> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (float)(leftSpan[i] + rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (float)(leftSpan[op1Index] + rightSpan[op2Index]);

                }
            }
        }
        public void Add(DenseTensor<float> tensor, float scalar, DenseTensor<float> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (float)(tensorSpan[i] + scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (float)(tensorSpan[op1Index] + scalar);

                }
            }
        }
        public void And(DenseTensor<float> left, DenseTensor<float> right, DenseTensor<float> result)
        {
            throw new NotSupportedException();
        }
        public void And(DenseTensor<float> tensor, float scalar, DenseTensor<float> result)
        {
            throw new NotSupportedException();
        }
        public void Contract(DenseTensor<float> left, DenseTensor<float> right, int[] leftAxes, int[] rightAxes, DenseTensor<float> result)
        {
            var summingDimensions = new int[leftAxes.Length];
            for(int i = 0; i < leftAxes.Length; i++)
            {
                summingDimensions[i] = left.dimensions[leftAxes[i]];
            }

            var summingStrides = ArrayUtilities.GetStrides(summingDimensions);
            int summingLength = (int)ArrayUtilities.GetProduct(summingDimensions);

            var resultStrides = result.strides;

            // translates from result index to left non-summing dimensions' index portion
            // since left non-summing dimensions are given precedence in result, the end is zero-padded
            int[] leftNonSummingStrides = new int[result.Rank];

            // translates from summing index to left summing dimensions' index portion
            int[] leftSummingStrides = new int[leftAxes.Length];
            ArrayUtilities.SplitStrides(left.strides, leftAxes, leftNonSummingStrides, 0, leftSummingStrides, 0);

            // translates from result index to right non-summing dimensions' index portion
            int[] rightNonSummingStrides = new int[result.Rank];
            //  right non-summing dimensions appear after left non-summing dimensions.
            int rightNonSummingStridesOffset = (left.Rank - leftAxes.Length);

            // translates from summing index to right summing dimensions' index portion
            int[] rightSummingStrides = new int[rightAxes.Length];
            ArrayUtilities.SplitStrides(right.strides, rightAxes, rightNonSummingStrides, rightNonSummingStridesOffset, rightSummingStrides, 0);
            
            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;

            for (int resultIndex = 0; resultIndex < resultSpan.Length; resultIndex++)
            {
                float sum = (float)0;

                int leftIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, leftNonSummingStrides);
                int rightIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, rightNonSummingStrides);

                for (int summingIndex = 0; summingIndex < summingLength; summingIndex++)
                {
                    int leftIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, leftSummingStrides);
                    int rightIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, rightSummingStrides);

                    int leftIndex = leftIndexNonSumming + leftIndexSumming;
                    int rightIndex = rightIndexNonSumming + rightIndexSumming;

                    sum += (float)(leftSpan[leftIndex] * rightSpan[rightIndex]);
                }

                resultSpan[resultIndex] = sum;
            }
        }
        public void Decrement(DenseTensor<float> tensor, DenseTensor<float> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            for(int i = 0; i < resultSpan.Length; i++)
            {
                resultSpan[i]--;
            }
        }
        public void Divide(DenseTensor<float> left, DenseTensor<float> right, DenseTensor<float> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (float)(leftSpan[i] / rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (float)(leftSpan[op1Index] / rightSpan[op2Index]);

                }
            }
        }
        public void Divide(DenseTensor<float> tensor, float scalar, DenseTensor<float> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (float)(tensorSpan[i] / scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (float)(tensorSpan[op1Index] / scalar);

                }
            }
        }
        public void Equals(DenseTensor<float> left, DenseTensor<float> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] == rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] == rightSpan[op2Index];

                }
            }
        }
        public void GreaterThan(DenseTensor<float> left, DenseTensor<float> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] > rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] > rightSpan[op2Index];

                }
            }
        }
        public void GreaterThanOrEqual(DenseTensor<float> left, DenseTensor<float> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] >= rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] >= rightSpan[op2Index];

                }
            }
        }
        public void Increment(DenseTensor<float> tensor, DenseTensor<float> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            for(int i = 0; i < resultSpan.Length; i++)
            {
                resultSpan[i]++;
            }
        }
        public void LeftShift(DenseTensor<float> tensor, int value, DenseTensor<float> result)
        {
            throw new NotSupportedException();
        }
        public void LessThan(DenseTensor<float> left, DenseTensor<float> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] < rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] < rightSpan[op2Index];

                }
            }
        }
        public void LessThanOrEqual(DenseTensor<float> left, DenseTensor<float> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] <= rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] <= rightSpan[op2Index];

                }
            }
        }
        public void Modulo(DenseTensor<float> left, DenseTensor<float> right, DenseTensor<float> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (float)(leftSpan[i] % rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (float)(leftSpan[op1Index] % rightSpan[op2Index]);

                }
            }
        }
        public void Modulo(DenseTensor<float> tensor, float scalar, DenseTensor<float> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (float)(tensorSpan[i] % scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (float)(tensorSpan[op1Index] % scalar);

                }
            }
        }
        public void Multiply(DenseTensor<float> left, DenseTensor<float> right, DenseTensor<float> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (float)(leftSpan[i] * rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (float)(leftSpan[op1Index] * rightSpan[op2Index]);

                }
            }
        }
        public void Multiply(DenseTensor<float> tensor, float scalar, DenseTensor<float> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (float)(tensorSpan[i] * scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (float)(tensorSpan[op1Index] * scalar);

                }
            }
        }
        public void NotEquals(DenseTensor<float> left, DenseTensor<float> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] != rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] != rightSpan[op2Index];

                }
            }
        }
        public void Or(DenseTensor<float> left, DenseTensor<float> right, DenseTensor<float> result)
        {
            throw new NotSupportedException();
        }
        public void Or(DenseTensor<float> tensor, float scalar, DenseTensor<float> result)
        {
            throw new NotSupportedException();
        }
        public void RightShift(DenseTensor<float> tensor, int value, DenseTensor<float> result)
        {
            throw new NotSupportedException();
        }
        public void Subtract(DenseTensor<float> left, DenseTensor<float> right, DenseTensor<float> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (float)(leftSpan[i] - rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (float)(leftSpan[op1Index] - rightSpan[op2Index]);

                }
            }
        }
        public void Subtract(DenseTensor<float> tensor, float scalar, DenseTensor<float> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (float)(tensorSpan[i] - scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (float)(tensorSpan[op1Index] - scalar);

                }
            }
        }
        public void UnaryMinus(DenseTensor<float> tensor, DenseTensor<float> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (float)-tensorSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (float)-tensorSpan[op1Index];

                }
            }
        }
        public void UnaryPlus(DenseTensor<float> tensor, DenseTensor<float> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (float)+tensorSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (float)+tensorSpan[op1Index];

                }
            }
        }
        public void Xor(DenseTensor<float> left, DenseTensor<float> right, DenseTensor<float> result)
        {
            throw new NotSupportedException();
        }
        public void Xor(DenseTensor<float> tensor, float scalar, DenseTensor<float> result)
        {
            throw new NotSupportedException();
        }
    }
    internal class IntArithmetic : ITensorArithmetic<int>
    {
        public int One => 1;
        public int Zero => 0;

        public void Add(Tensor<int> left, Tensor<int> right, Tensor<int> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (int)(left[indices] + right[indices]);
            }
            
        }
        public void Add(Tensor<int> tensor, int scalar, Tensor<int> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (int)(tensor[indices] + scalar);
            }
            
        }
        public void And(Tensor<int> left, Tensor<int> right, Tensor<int> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (int)(left[indices] & right[indices]);
            }
            
        }
        public void And(Tensor<int> tensor, int scalar, Tensor<int> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (int)(tensor[indices] & scalar);
            }
            
        }
        public void Contract(Tensor<int> left, Tensor<int> right, int[] leftAxes, int[] rightAxes, Tensor<int> result)
        {
            var leftIndices = new int[left.Rank];
            var rightIndices = new int[right.Rank];
            var resultIndices = new int[result.Rank];

            var summingDimensions = new int[leftAxes.Length];
            for(int i = 0; i < leftAxes.Length; i++)
            {
                summingDimensions[i] = left.dimensions[leftAxes[i]];
            }

            var summingStrides = ArrayUtilities.GetStrides(summingDimensions);
            int summingLength = (int)ArrayUtilities.GetProduct(summingDimensions);

            var resultStrides = result.strides;

            // translates from result index to left non-summing dimensions' index portion
            // since left non-summing dimensions are given precedence in result, the end is zero-padded
            int[] leftNonSummingStrides = new int[result.Rank];

            // translates from summing index to left summing dimensions' index portion
            int[] leftSummingStrides = new int[leftAxes.Length];
            ArrayUtilities.SplitStrides(left.strides, leftAxes, leftNonSummingStrides, 0, leftSummingStrides, 0);

            // translates from result index to right non-summing dimensions' index portion
            int[] rightNonSummingStrides = new int[result.Rank];
            //  right non-summing dimensions appear after left non-summing dimensions.
            int rightNonSummingStridesOffset = (left.Rank - leftAxes.Length);

            // translates from summing index to right summing dimensions' index portion
            int[] rightSummingStrides = new int[rightAxes.Length];
            ArrayUtilities.SplitStrides(right.strides, rightAxes, rightNonSummingStrides, rightNonSummingStridesOffset, rightSummingStrides, 0);

            for (int resultIndex = 0; resultIndex < result.Length; resultIndex++)
            {
                int sum = (int)0;
                
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, resultIndex, resultIndices);

                int leftIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, leftNonSummingStrides);
                int rightIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, rightNonSummingStrides);

                for (int summingIndex = 0; summingIndex < summingLength; summingIndex++)
                {
                    int leftIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, leftSummingStrides);
                    int rightIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, rightSummingStrides);

                    int leftIndex = leftIndexNonSumming + leftIndexSumming;
                    int rightIndex = rightIndexNonSumming + rightIndexSumming;

                    // todo, make this more efficient
                    ArrayUtilities.GetIndices(left.strides, left.IsReversedStride, leftIndex, leftIndices);
                    ArrayUtilities.GetIndices(right.strides, right.IsReversedStride, rightIndex, rightIndices);

                    sum += (int)(left[leftIndices] * right[rightIndices]);
                }
                
                result[resultIndices] = sum;
            }
        }
        public void Decrement(Tensor<int> tensor, Tensor<int> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices]--;
            }
            
        }
        public void Divide(Tensor<int> left, Tensor<int> right, Tensor<int> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (int)(left[indices] / right[indices]);
            }
            
        }
        public void Divide(Tensor<int> tensor, int scalar, Tensor<int> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (int)(tensor[indices] / scalar);
            }
            
        }
        public void Equals(Tensor<int> left, Tensor<int> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] == right[indices];
            }
            
        }
        public void GreaterThan(Tensor<int> left, Tensor<int> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] > right[indices];
            }
            
        }
        public void GreaterThanOrEqual(Tensor<int> left, Tensor<int> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] >= right[indices];
            }
            
        }
        public void Increment(Tensor<int> tensor, Tensor<int> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices]++;
            }
            
        }
        public void LeftShift(Tensor<int> tensor, int value, Tensor<int> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (int)(tensor[indices] << value);
            }
            
        }
        public void LessThan(Tensor<int> left, Tensor<int> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] < right[indices];
            }
            
        }
        public void LessThanOrEqual(Tensor<int> left, Tensor<int> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] <= right[indices];
            }
            
        }
        public void Modulo(Tensor<int> left, Tensor<int> right, Tensor<int> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (int)(left[indices] % right[indices]);
            }
            
        }
        public void Modulo(Tensor<int> tensor, int scalar, Tensor<int> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (int)(tensor[indices] % scalar);
            }
            
        }
        public void Multiply(Tensor<int> left, Tensor<int> right, Tensor<int> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (int)(left[indices] * right[indices]);
            }
            
        }
        public void Multiply(Tensor<int> tensor, int scalar, Tensor<int> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (int)(tensor[indices] * scalar);
            }
            
        }
        public void NotEquals(Tensor<int> left, Tensor<int> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] != right[indices];
            }
            
        }
        public void Or(Tensor<int> left, Tensor<int> right, Tensor<int> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (int)(left[indices] | right[indices]);
            }
            
        }
        public void Or(Tensor<int> tensor, int scalar, Tensor<int> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (int)(tensor[indices] | scalar);
            }
            
        }
        public void RightShift(Tensor<int> tensor, int value, Tensor<int> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (int)(tensor[indices] >> value);
            }
            
        }
        public void Subtract(Tensor<int> left, Tensor<int> right, Tensor<int> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (int)(left[indices] - right[indices]);
            }
            
        }
        public void Subtract(Tensor<int> tensor, int scalar, Tensor<int> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (int)(tensor[indices] - scalar);
            }
            
        }
        public void UnaryMinus(Tensor<int> tensor, Tensor<int> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (int)-tensor[indices];
            }
            
        }
        public void UnaryPlus(Tensor<int> tensor, Tensor<int> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (int)+tensor[indices];
            }
            
        }
        public void Xor(Tensor<int> left, Tensor<int> right, Tensor<int> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (int)(left[indices] ^ right[indices]);
            }
            
        }
        public void Xor(Tensor<int> tensor, int scalar, Tensor<int> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (int)(tensor[indices] ^ scalar);
            }
            
        }

        public void Add(DenseTensor<int> left, DenseTensor<int> right, DenseTensor<int> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (int)(leftSpan[i] + rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (int)(leftSpan[op1Index] + rightSpan[op2Index]);

                }
            }
        }
        public void Add(DenseTensor<int> tensor, int scalar, DenseTensor<int> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (int)(tensorSpan[i] + scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (int)(tensorSpan[op1Index] + scalar);

                }
            }
        }
        public void And(DenseTensor<int> left, DenseTensor<int> right, DenseTensor<int> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (int)(leftSpan[i] & rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (int)(leftSpan[op1Index] & rightSpan[op2Index]);

                }
            }
        }
        public void And(DenseTensor<int> tensor, int scalar, DenseTensor<int> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (int)(tensorSpan[i] & scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (int)(tensorSpan[op1Index] & scalar);

                }
            }
        }
        public void Contract(DenseTensor<int> left, DenseTensor<int> right, int[] leftAxes, int[] rightAxes, DenseTensor<int> result)
        {
            var summingDimensions = new int[leftAxes.Length];
            for(int i = 0; i < leftAxes.Length; i++)
            {
                summingDimensions[i] = left.dimensions[leftAxes[i]];
            }

            var summingStrides = ArrayUtilities.GetStrides(summingDimensions);
            int summingLength = (int)ArrayUtilities.GetProduct(summingDimensions);

            var resultStrides = result.strides;

            // translates from result index to left non-summing dimensions' index portion
            // since left non-summing dimensions are given precedence in result, the end is zero-padded
            int[] leftNonSummingStrides = new int[result.Rank];

            // translates from summing index to left summing dimensions' index portion
            int[] leftSummingStrides = new int[leftAxes.Length];
            ArrayUtilities.SplitStrides(left.strides, leftAxes, leftNonSummingStrides, 0, leftSummingStrides, 0);

            // translates from result index to right non-summing dimensions' index portion
            int[] rightNonSummingStrides = new int[result.Rank];
            //  right non-summing dimensions appear after left non-summing dimensions.
            int rightNonSummingStridesOffset = (left.Rank - leftAxes.Length);

            // translates from summing index to right summing dimensions' index portion
            int[] rightSummingStrides = new int[rightAxes.Length];
            ArrayUtilities.SplitStrides(right.strides, rightAxes, rightNonSummingStrides, rightNonSummingStridesOffset, rightSummingStrides, 0);
            
            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;

            for (int resultIndex = 0; resultIndex < resultSpan.Length; resultIndex++)
            {
                int sum = (int)0;

                int leftIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, leftNonSummingStrides);
                int rightIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, rightNonSummingStrides);

                for (int summingIndex = 0; summingIndex < summingLength; summingIndex++)
                {
                    int leftIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, leftSummingStrides);
                    int rightIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, rightSummingStrides);

                    int leftIndex = leftIndexNonSumming + leftIndexSumming;
                    int rightIndex = rightIndexNonSumming + rightIndexSumming;

                    sum += (int)(leftSpan[leftIndex] * rightSpan[rightIndex]);
                }

                resultSpan[resultIndex] = sum;
            }
        }
        public void Decrement(DenseTensor<int> tensor, DenseTensor<int> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            for(int i = 0; i < resultSpan.Length; i++)
            {
                resultSpan[i]--;
            }
        }
        public void Divide(DenseTensor<int> left, DenseTensor<int> right, DenseTensor<int> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (int)(leftSpan[i] / rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (int)(leftSpan[op1Index] / rightSpan[op2Index]);

                }
            }
        }
        public void Divide(DenseTensor<int> tensor, int scalar, DenseTensor<int> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (int)(tensorSpan[i] / scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (int)(tensorSpan[op1Index] / scalar);

                }
            }
        }
        public void Equals(DenseTensor<int> left, DenseTensor<int> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] == rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] == rightSpan[op2Index];

                }
            }
        }
        public void GreaterThan(DenseTensor<int> left, DenseTensor<int> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] > rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] > rightSpan[op2Index];

                }
            }
        }
        public void GreaterThanOrEqual(DenseTensor<int> left, DenseTensor<int> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] >= rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] >= rightSpan[op2Index];

                }
            }
        }
        public void Increment(DenseTensor<int> tensor, DenseTensor<int> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            for(int i = 0; i < resultSpan.Length; i++)
            {
                resultSpan[i]++;
            }
        }
        public void LeftShift(DenseTensor<int> tensor, int value, DenseTensor<int> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (int)(tensorSpan[i] << value);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (int)(tensorSpan[op1Index] << value);

                }
            }
        }
        public void LessThan(DenseTensor<int> left, DenseTensor<int> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] < rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] < rightSpan[op2Index];

                }
            }
        }
        public void LessThanOrEqual(DenseTensor<int> left, DenseTensor<int> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] <= rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] <= rightSpan[op2Index];

                }
            }
        }
        public void Modulo(DenseTensor<int> left, DenseTensor<int> right, DenseTensor<int> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (int)(leftSpan[i] % rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (int)(leftSpan[op1Index] % rightSpan[op2Index]);

                }
            }
        }
        public void Modulo(DenseTensor<int> tensor, int scalar, DenseTensor<int> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (int)(tensorSpan[i] % scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (int)(tensorSpan[op1Index] % scalar);

                }
            }
        }
        public void Multiply(DenseTensor<int> left, DenseTensor<int> right, DenseTensor<int> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (int)(leftSpan[i] * rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (int)(leftSpan[op1Index] * rightSpan[op2Index]);

                }
            }
        }
        public void Multiply(DenseTensor<int> tensor, int scalar, DenseTensor<int> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (int)(tensorSpan[i] * scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (int)(tensorSpan[op1Index] * scalar);

                }
            }
        }
        public void NotEquals(DenseTensor<int> left, DenseTensor<int> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] != rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] != rightSpan[op2Index];

                }
            }
        }
        public void Or(DenseTensor<int> left, DenseTensor<int> right, DenseTensor<int> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (int)(leftSpan[i] | rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (int)(leftSpan[op1Index] | rightSpan[op2Index]);

                }
            }
        }
        public void Or(DenseTensor<int> tensor, int scalar, DenseTensor<int> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (int)(tensorSpan[i] | scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (int)(tensorSpan[op1Index] | scalar);

                }
            }
        }
        public void RightShift(DenseTensor<int> tensor, int value, DenseTensor<int> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (int)(tensorSpan[i] >> value);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (int)(tensorSpan[op1Index] >> value);

                }
            }
        }
        public void Subtract(DenseTensor<int> left, DenseTensor<int> right, DenseTensor<int> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (int)(leftSpan[i] - rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (int)(leftSpan[op1Index] - rightSpan[op2Index]);

                }
            }
        }
        public void Subtract(DenseTensor<int> tensor, int scalar, DenseTensor<int> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (int)(tensorSpan[i] - scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (int)(tensorSpan[op1Index] - scalar);

                }
            }
        }
        public void UnaryMinus(DenseTensor<int> tensor, DenseTensor<int> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (int)-tensorSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (int)-tensorSpan[op1Index];

                }
            }
        }
        public void UnaryPlus(DenseTensor<int> tensor, DenseTensor<int> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (int)+tensorSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (int)+tensorSpan[op1Index];

                }
            }
        }
        public void Xor(DenseTensor<int> left, DenseTensor<int> right, DenseTensor<int> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (int)(leftSpan[i] ^ rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (int)(leftSpan[op1Index] ^ rightSpan[op2Index]);

                }
            }
        }
        public void Xor(DenseTensor<int> tensor, int scalar, DenseTensor<int> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (int)(tensorSpan[i] ^ scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (int)(tensorSpan[op1Index] ^ scalar);

                }
            }
        }
    }
    internal class LongArithmetic : ITensorArithmetic<long>
    {
        public long One => 1;
        public long Zero => 0;

        public void Add(Tensor<long> left, Tensor<long> right, Tensor<long> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (long)(left[indices] + right[indices]);
            }
            
        }
        public void Add(Tensor<long> tensor, long scalar, Tensor<long> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (long)(tensor[indices] + scalar);
            }
            
        }
        public void And(Tensor<long> left, Tensor<long> right, Tensor<long> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (long)(left[indices] & right[indices]);
            }
            
        }
        public void And(Tensor<long> tensor, long scalar, Tensor<long> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (long)(tensor[indices] & scalar);
            }
            
        }
        public void Contract(Tensor<long> left, Tensor<long> right, int[] leftAxes, int[] rightAxes, Tensor<long> result)
        {
            var leftIndices = new int[left.Rank];
            var rightIndices = new int[right.Rank];
            var resultIndices = new int[result.Rank];

            var summingDimensions = new int[leftAxes.Length];
            for(int i = 0; i < leftAxes.Length; i++)
            {
                summingDimensions[i] = left.dimensions[leftAxes[i]];
            }

            var summingStrides = ArrayUtilities.GetStrides(summingDimensions);
            int summingLength = (int)ArrayUtilities.GetProduct(summingDimensions);

            var resultStrides = result.strides;

            // translates from result index to left non-summing dimensions' index portion
            // since left non-summing dimensions are given precedence in result, the end is zero-padded
            int[] leftNonSummingStrides = new int[result.Rank];

            // translates from summing index to left summing dimensions' index portion
            int[] leftSummingStrides = new int[leftAxes.Length];
            ArrayUtilities.SplitStrides(left.strides, leftAxes, leftNonSummingStrides, 0, leftSummingStrides, 0);

            // translates from result index to right non-summing dimensions' index portion
            int[] rightNonSummingStrides = new int[result.Rank];
            //  right non-summing dimensions appear after left non-summing dimensions.
            int rightNonSummingStridesOffset = (left.Rank - leftAxes.Length);

            // translates from summing index to right summing dimensions' index portion
            int[] rightSummingStrides = new int[rightAxes.Length];
            ArrayUtilities.SplitStrides(right.strides, rightAxes, rightNonSummingStrides, rightNonSummingStridesOffset, rightSummingStrides, 0);

            for (int resultIndex = 0; resultIndex < result.Length; resultIndex++)
            {
                long sum = (long)0;
                
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, resultIndex, resultIndices);

                int leftIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, leftNonSummingStrides);
                int rightIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, rightNonSummingStrides);

                for (int summingIndex = 0; summingIndex < summingLength; summingIndex++)
                {
                    int leftIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, leftSummingStrides);
                    int rightIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, rightSummingStrides);

                    int leftIndex = leftIndexNonSumming + leftIndexSumming;
                    int rightIndex = rightIndexNonSumming + rightIndexSumming;

                    // todo, make this more efficient
                    ArrayUtilities.GetIndices(left.strides, left.IsReversedStride, leftIndex, leftIndices);
                    ArrayUtilities.GetIndices(right.strides, right.IsReversedStride, rightIndex, rightIndices);

                    sum += (long)(left[leftIndices] * right[rightIndices]);
                }
                
                result[resultIndices] = sum;
            }
        }
        public void Decrement(Tensor<long> tensor, Tensor<long> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices]--;
            }
            
        }
        public void Divide(Tensor<long> left, Tensor<long> right, Tensor<long> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (long)(left[indices] / right[indices]);
            }
            
        }
        public void Divide(Tensor<long> tensor, long scalar, Tensor<long> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (long)(tensor[indices] / scalar);
            }
            
        }
        public void Equals(Tensor<long> left, Tensor<long> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] == right[indices];
            }
            
        }
        public void GreaterThan(Tensor<long> left, Tensor<long> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] > right[indices];
            }
            
        }
        public void GreaterThanOrEqual(Tensor<long> left, Tensor<long> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] >= right[indices];
            }
            
        }
        public void Increment(Tensor<long> tensor, Tensor<long> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices]++;
            }
            
        }
        public void LeftShift(Tensor<long> tensor, int value, Tensor<long> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (long)(tensor[indices] << value);
            }
            
        }
        public void LessThan(Tensor<long> left, Tensor<long> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] < right[indices];
            }
            
        }
        public void LessThanOrEqual(Tensor<long> left, Tensor<long> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] <= right[indices];
            }
            
        }
        public void Modulo(Tensor<long> left, Tensor<long> right, Tensor<long> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (long)(left[indices] % right[indices]);
            }
            
        }
        public void Modulo(Tensor<long> tensor, long scalar, Tensor<long> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (long)(tensor[indices] % scalar);
            }
            
        }
        public void Multiply(Tensor<long> left, Tensor<long> right, Tensor<long> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (long)(left[indices] * right[indices]);
            }
            
        }
        public void Multiply(Tensor<long> tensor, long scalar, Tensor<long> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (long)(tensor[indices] * scalar);
            }
            
        }
        public void NotEquals(Tensor<long> left, Tensor<long> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] != right[indices];
            }
            
        }
        public void Or(Tensor<long> left, Tensor<long> right, Tensor<long> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (long)(left[indices] | right[indices]);
            }
            
        }
        public void Or(Tensor<long> tensor, long scalar, Tensor<long> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (long)(tensor[indices] | scalar);
            }
            
        }
        public void RightShift(Tensor<long> tensor, int value, Tensor<long> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (long)(tensor[indices] >> value);
            }
            
        }
        public void Subtract(Tensor<long> left, Tensor<long> right, Tensor<long> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (long)(left[indices] - right[indices]);
            }
            
        }
        public void Subtract(Tensor<long> tensor, long scalar, Tensor<long> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (long)(tensor[indices] - scalar);
            }
            
        }
        public void UnaryMinus(Tensor<long> tensor, Tensor<long> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (long)-tensor[indices];
            }
            
        }
        public void UnaryPlus(Tensor<long> tensor, Tensor<long> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (long)+tensor[indices];
            }
            
        }
        public void Xor(Tensor<long> left, Tensor<long> right, Tensor<long> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (long)(left[indices] ^ right[indices]);
            }
            
        }
        public void Xor(Tensor<long> tensor, long scalar, Tensor<long> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (long)(tensor[indices] ^ scalar);
            }
            
        }

        public void Add(DenseTensor<long> left, DenseTensor<long> right, DenseTensor<long> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (long)(leftSpan[i] + rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (long)(leftSpan[op1Index] + rightSpan[op2Index]);

                }
            }
        }
        public void Add(DenseTensor<long> tensor, long scalar, DenseTensor<long> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (long)(tensorSpan[i] + scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (long)(tensorSpan[op1Index] + scalar);

                }
            }
        }
        public void And(DenseTensor<long> left, DenseTensor<long> right, DenseTensor<long> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (long)(leftSpan[i] & rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (long)(leftSpan[op1Index] & rightSpan[op2Index]);

                }
            }
        }
        public void And(DenseTensor<long> tensor, long scalar, DenseTensor<long> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (long)(tensorSpan[i] & scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (long)(tensorSpan[op1Index] & scalar);

                }
            }
        }
        public void Contract(DenseTensor<long> left, DenseTensor<long> right, int[] leftAxes, int[] rightAxes, DenseTensor<long> result)
        {
            var summingDimensions = new int[leftAxes.Length];
            for(int i = 0; i < leftAxes.Length; i++)
            {
                summingDimensions[i] = left.dimensions[leftAxes[i]];
            }

            var summingStrides = ArrayUtilities.GetStrides(summingDimensions);
            int summingLength = (int)ArrayUtilities.GetProduct(summingDimensions);

            var resultStrides = result.strides;

            // translates from result index to left non-summing dimensions' index portion
            // since left non-summing dimensions are given precedence in result, the end is zero-padded
            int[] leftNonSummingStrides = new int[result.Rank];

            // translates from summing index to left summing dimensions' index portion
            int[] leftSummingStrides = new int[leftAxes.Length];
            ArrayUtilities.SplitStrides(left.strides, leftAxes, leftNonSummingStrides, 0, leftSummingStrides, 0);

            // translates from result index to right non-summing dimensions' index portion
            int[] rightNonSummingStrides = new int[result.Rank];
            //  right non-summing dimensions appear after left non-summing dimensions.
            int rightNonSummingStridesOffset = (left.Rank - leftAxes.Length);

            // translates from summing index to right summing dimensions' index portion
            int[] rightSummingStrides = new int[rightAxes.Length];
            ArrayUtilities.SplitStrides(right.strides, rightAxes, rightNonSummingStrides, rightNonSummingStridesOffset, rightSummingStrides, 0);
            
            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;

            for (int resultIndex = 0; resultIndex < resultSpan.Length; resultIndex++)
            {
                long sum = (long)0;

                int leftIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, leftNonSummingStrides);
                int rightIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, rightNonSummingStrides);

                for (int summingIndex = 0; summingIndex < summingLength; summingIndex++)
                {
                    int leftIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, leftSummingStrides);
                    int rightIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, rightSummingStrides);

                    int leftIndex = leftIndexNonSumming + leftIndexSumming;
                    int rightIndex = rightIndexNonSumming + rightIndexSumming;

                    sum += (long)(leftSpan[leftIndex] * rightSpan[rightIndex]);
                }

                resultSpan[resultIndex] = sum;
            }
        }
        public void Decrement(DenseTensor<long> tensor, DenseTensor<long> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            for(int i = 0; i < resultSpan.Length; i++)
            {
                resultSpan[i]--;
            }
        }
        public void Divide(DenseTensor<long> left, DenseTensor<long> right, DenseTensor<long> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (long)(leftSpan[i] / rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (long)(leftSpan[op1Index] / rightSpan[op2Index]);

                }
            }
        }
        public void Divide(DenseTensor<long> tensor, long scalar, DenseTensor<long> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (long)(tensorSpan[i] / scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (long)(tensorSpan[op1Index] / scalar);

                }
            }
        }
        public void Equals(DenseTensor<long> left, DenseTensor<long> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] == rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] == rightSpan[op2Index];

                }
            }
        }
        public void GreaterThan(DenseTensor<long> left, DenseTensor<long> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] > rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] > rightSpan[op2Index];

                }
            }
        }
        public void GreaterThanOrEqual(DenseTensor<long> left, DenseTensor<long> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] >= rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] >= rightSpan[op2Index];

                }
            }
        }
        public void Increment(DenseTensor<long> tensor, DenseTensor<long> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            for(int i = 0; i < resultSpan.Length; i++)
            {
                resultSpan[i]++;
            }
        }
        public void LeftShift(DenseTensor<long> tensor, int value, DenseTensor<long> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (long)(tensorSpan[i] << value);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (long)(tensorSpan[op1Index] << value);

                }
            }
        }
        public void LessThan(DenseTensor<long> left, DenseTensor<long> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] < rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] < rightSpan[op2Index];

                }
            }
        }
        public void LessThanOrEqual(DenseTensor<long> left, DenseTensor<long> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] <= rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] <= rightSpan[op2Index];

                }
            }
        }
        public void Modulo(DenseTensor<long> left, DenseTensor<long> right, DenseTensor<long> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (long)(leftSpan[i] % rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (long)(leftSpan[op1Index] % rightSpan[op2Index]);

                }
            }
        }
        public void Modulo(DenseTensor<long> tensor, long scalar, DenseTensor<long> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (long)(tensorSpan[i] % scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (long)(tensorSpan[op1Index] % scalar);

                }
            }
        }
        public void Multiply(DenseTensor<long> left, DenseTensor<long> right, DenseTensor<long> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (long)(leftSpan[i] * rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (long)(leftSpan[op1Index] * rightSpan[op2Index]);

                }
            }
        }
        public void Multiply(DenseTensor<long> tensor, long scalar, DenseTensor<long> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (long)(tensorSpan[i] * scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (long)(tensorSpan[op1Index] * scalar);

                }
            }
        }
        public void NotEquals(DenseTensor<long> left, DenseTensor<long> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] != rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] != rightSpan[op2Index];

                }
            }
        }
        public void Or(DenseTensor<long> left, DenseTensor<long> right, DenseTensor<long> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (long)(leftSpan[i] | rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (long)(leftSpan[op1Index] | rightSpan[op2Index]);

                }
            }
        }
        public void Or(DenseTensor<long> tensor, long scalar, DenseTensor<long> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (long)(tensorSpan[i] | scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (long)(tensorSpan[op1Index] | scalar);

                }
            }
        }
        public void RightShift(DenseTensor<long> tensor, int value, DenseTensor<long> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (long)(tensorSpan[i] >> value);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (long)(tensorSpan[op1Index] >> value);

                }
            }
        }
        public void Subtract(DenseTensor<long> left, DenseTensor<long> right, DenseTensor<long> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (long)(leftSpan[i] - rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (long)(leftSpan[op1Index] - rightSpan[op2Index]);

                }
            }
        }
        public void Subtract(DenseTensor<long> tensor, long scalar, DenseTensor<long> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (long)(tensorSpan[i] - scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (long)(tensorSpan[op1Index] - scalar);

                }
            }
        }
        public void UnaryMinus(DenseTensor<long> tensor, DenseTensor<long> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (long)-tensorSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (long)-tensorSpan[op1Index];

                }
            }
        }
        public void UnaryPlus(DenseTensor<long> tensor, DenseTensor<long> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (long)+tensorSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (long)+tensorSpan[op1Index];

                }
            }
        }
        public void Xor(DenseTensor<long> left, DenseTensor<long> right, DenseTensor<long> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (long)(leftSpan[i] ^ rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (long)(leftSpan[op1Index] ^ rightSpan[op2Index]);

                }
            }
        }
        public void Xor(DenseTensor<long> tensor, long scalar, DenseTensor<long> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (long)(tensorSpan[i] ^ scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (long)(tensorSpan[op1Index] ^ scalar);

                }
            }
        }
    }
    internal class SByteArithmetic : ITensorArithmetic<sbyte>
    {
        public sbyte One => 1;
        public sbyte Zero => 0;

        public void Add(Tensor<sbyte> left, Tensor<sbyte> right, Tensor<sbyte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (sbyte)(left[indices] + right[indices]);
            }
            
        }
        public void Add(Tensor<sbyte> tensor, sbyte scalar, Tensor<sbyte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (sbyte)(tensor[indices] + scalar);
            }
            
        }
        public void And(Tensor<sbyte> left, Tensor<sbyte> right, Tensor<sbyte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (sbyte)(left[indices] & right[indices]);
            }
            
        }
        public void And(Tensor<sbyte> tensor, sbyte scalar, Tensor<sbyte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (sbyte)(tensor[indices] & scalar);
            }
            
        }
        public void Contract(Tensor<sbyte> left, Tensor<sbyte> right, int[] leftAxes, int[] rightAxes, Tensor<sbyte> result)
        {
            var leftIndices = new int[left.Rank];
            var rightIndices = new int[right.Rank];
            var resultIndices = new int[result.Rank];

            var summingDimensions = new int[leftAxes.Length];
            for(int i = 0; i < leftAxes.Length; i++)
            {
                summingDimensions[i] = left.dimensions[leftAxes[i]];
            }

            var summingStrides = ArrayUtilities.GetStrides(summingDimensions);
            int summingLength = (int)ArrayUtilities.GetProduct(summingDimensions);

            var resultStrides = result.strides;

            // translates from result index to left non-summing dimensions' index portion
            // since left non-summing dimensions are given precedence in result, the end is zero-padded
            int[] leftNonSummingStrides = new int[result.Rank];

            // translates from summing index to left summing dimensions' index portion
            int[] leftSummingStrides = new int[leftAxes.Length];
            ArrayUtilities.SplitStrides(left.strides, leftAxes, leftNonSummingStrides, 0, leftSummingStrides, 0);

            // translates from result index to right non-summing dimensions' index portion
            int[] rightNonSummingStrides = new int[result.Rank];
            //  right non-summing dimensions appear after left non-summing dimensions.
            int rightNonSummingStridesOffset = (left.Rank - leftAxes.Length);

            // translates from summing index to right summing dimensions' index portion
            int[] rightSummingStrides = new int[rightAxes.Length];
            ArrayUtilities.SplitStrides(right.strides, rightAxes, rightNonSummingStrides, rightNonSummingStridesOffset, rightSummingStrides, 0);

            for (int resultIndex = 0; resultIndex < result.Length; resultIndex++)
            {
                sbyte sum = (sbyte)0;
                
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, resultIndex, resultIndices);

                int leftIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, leftNonSummingStrides);
                int rightIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, rightNonSummingStrides);

                for (int summingIndex = 0; summingIndex < summingLength; summingIndex++)
                {
                    int leftIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, leftSummingStrides);
                    int rightIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, rightSummingStrides);

                    int leftIndex = leftIndexNonSumming + leftIndexSumming;
                    int rightIndex = rightIndexNonSumming + rightIndexSumming;

                    // todo, make this more efficient
                    ArrayUtilities.GetIndices(left.strides, left.IsReversedStride, leftIndex, leftIndices);
                    ArrayUtilities.GetIndices(right.strides, right.IsReversedStride, rightIndex, rightIndices);

                    sum += (sbyte)(left[leftIndices] * right[rightIndices]);
                }
                
                result[resultIndices] = sum;
            }
        }
        public void Decrement(Tensor<sbyte> tensor, Tensor<sbyte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices]--;
            }
            
        }
        public void Divide(Tensor<sbyte> left, Tensor<sbyte> right, Tensor<sbyte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (sbyte)(left[indices] / right[indices]);
            }
            
        }
        public void Divide(Tensor<sbyte> tensor, sbyte scalar, Tensor<sbyte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (sbyte)(tensor[indices] / scalar);
            }
            
        }
        public void Equals(Tensor<sbyte> left, Tensor<sbyte> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] == right[indices];
            }
            
        }
        public void GreaterThan(Tensor<sbyte> left, Tensor<sbyte> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] > right[indices];
            }
            
        }
        public void GreaterThanOrEqual(Tensor<sbyte> left, Tensor<sbyte> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] >= right[indices];
            }
            
        }
        public void Increment(Tensor<sbyte> tensor, Tensor<sbyte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices]++;
            }
            
        }
        public void LeftShift(Tensor<sbyte> tensor, int value, Tensor<sbyte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (sbyte)(tensor[indices] << value);
            }
            
        }
        public void LessThan(Tensor<sbyte> left, Tensor<sbyte> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] < right[indices];
            }
            
        }
        public void LessThanOrEqual(Tensor<sbyte> left, Tensor<sbyte> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] <= right[indices];
            }
            
        }
        public void Modulo(Tensor<sbyte> left, Tensor<sbyte> right, Tensor<sbyte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (sbyte)(left[indices] % right[indices]);
            }
            
        }
        public void Modulo(Tensor<sbyte> tensor, sbyte scalar, Tensor<sbyte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (sbyte)(tensor[indices] % scalar);
            }
            
        }
        public void Multiply(Tensor<sbyte> left, Tensor<sbyte> right, Tensor<sbyte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (sbyte)(left[indices] * right[indices]);
            }
            
        }
        public void Multiply(Tensor<sbyte> tensor, sbyte scalar, Tensor<sbyte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (sbyte)(tensor[indices] * scalar);
            }
            
        }
        public void NotEquals(Tensor<sbyte> left, Tensor<sbyte> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] != right[indices];
            }
            
        }
        public void Or(Tensor<sbyte> left, Tensor<sbyte> right, Tensor<sbyte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (sbyte)(left[indices] | right[indices]);
            }
            
        }
        public void Or(Tensor<sbyte> tensor, sbyte scalar, Tensor<sbyte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (sbyte)(tensor[indices] | scalar);
            }
            
        }
        public void RightShift(Tensor<sbyte> tensor, int value, Tensor<sbyte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (sbyte)(tensor[indices] >> value);
            }
            
        }
        public void Subtract(Tensor<sbyte> left, Tensor<sbyte> right, Tensor<sbyte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (sbyte)(left[indices] - right[indices]);
            }
            
        }
        public void Subtract(Tensor<sbyte> tensor, sbyte scalar, Tensor<sbyte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (sbyte)(tensor[indices] - scalar);
            }
            
        }
        public void UnaryMinus(Tensor<sbyte> tensor, Tensor<sbyte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (sbyte)-tensor[indices];
            }
            
        }
        public void UnaryPlus(Tensor<sbyte> tensor, Tensor<sbyte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (sbyte)+tensor[indices];
            }
            
        }
        public void Xor(Tensor<sbyte> left, Tensor<sbyte> right, Tensor<sbyte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (sbyte)(left[indices] ^ right[indices]);
            }
            
        }
        public void Xor(Tensor<sbyte> tensor, sbyte scalar, Tensor<sbyte> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (sbyte)(tensor[indices] ^ scalar);
            }
            
        }

        public void Add(DenseTensor<sbyte> left, DenseTensor<sbyte> right, DenseTensor<sbyte> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (sbyte)(leftSpan[i] + rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (sbyte)(leftSpan[op1Index] + rightSpan[op2Index]);

                }
            }
        }
        public void Add(DenseTensor<sbyte> tensor, sbyte scalar, DenseTensor<sbyte> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (sbyte)(tensorSpan[i] + scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (sbyte)(tensorSpan[op1Index] + scalar);

                }
            }
        }
        public void And(DenseTensor<sbyte> left, DenseTensor<sbyte> right, DenseTensor<sbyte> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (sbyte)(leftSpan[i] & rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (sbyte)(leftSpan[op1Index] & rightSpan[op2Index]);

                }
            }
        }
        public void And(DenseTensor<sbyte> tensor, sbyte scalar, DenseTensor<sbyte> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (sbyte)(tensorSpan[i] & scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (sbyte)(tensorSpan[op1Index] & scalar);

                }
            }
        }
        public void Contract(DenseTensor<sbyte> left, DenseTensor<sbyte> right, int[] leftAxes, int[] rightAxes, DenseTensor<sbyte> result)
        {
            var summingDimensions = new int[leftAxes.Length];
            for(int i = 0; i < leftAxes.Length; i++)
            {
                summingDimensions[i] = left.dimensions[leftAxes[i]];
            }

            var summingStrides = ArrayUtilities.GetStrides(summingDimensions);
            int summingLength = (int)ArrayUtilities.GetProduct(summingDimensions);

            var resultStrides = result.strides;

            // translates from result index to left non-summing dimensions' index portion
            // since left non-summing dimensions are given precedence in result, the end is zero-padded
            int[] leftNonSummingStrides = new int[result.Rank];

            // translates from summing index to left summing dimensions' index portion
            int[] leftSummingStrides = new int[leftAxes.Length];
            ArrayUtilities.SplitStrides(left.strides, leftAxes, leftNonSummingStrides, 0, leftSummingStrides, 0);

            // translates from result index to right non-summing dimensions' index portion
            int[] rightNonSummingStrides = new int[result.Rank];
            //  right non-summing dimensions appear after left non-summing dimensions.
            int rightNonSummingStridesOffset = (left.Rank - leftAxes.Length);

            // translates from summing index to right summing dimensions' index portion
            int[] rightSummingStrides = new int[rightAxes.Length];
            ArrayUtilities.SplitStrides(right.strides, rightAxes, rightNonSummingStrides, rightNonSummingStridesOffset, rightSummingStrides, 0);
            
            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;

            for (int resultIndex = 0; resultIndex < resultSpan.Length; resultIndex++)
            {
                sbyte sum = (sbyte)0;

                int leftIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, leftNonSummingStrides);
                int rightIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, rightNonSummingStrides);

                for (int summingIndex = 0; summingIndex < summingLength; summingIndex++)
                {
                    int leftIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, leftSummingStrides);
                    int rightIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, rightSummingStrides);

                    int leftIndex = leftIndexNonSumming + leftIndexSumming;
                    int rightIndex = rightIndexNonSumming + rightIndexSumming;

                    sum += (sbyte)(leftSpan[leftIndex] * rightSpan[rightIndex]);
                }

                resultSpan[resultIndex] = sum;
            }
        }
        public void Decrement(DenseTensor<sbyte> tensor, DenseTensor<sbyte> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            for(int i = 0; i < resultSpan.Length; i++)
            {
                resultSpan[i]--;
            }
        }
        public void Divide(DenseTensor<sbyte> left, DenseTensor<sbyte> right, DenseTensor<sbyte> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (sbyte)(leftSpan[i] / rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (sbyte)(leftSpan[op1Index] / rightSpan[op2Index]);

                }
            }
        }
        public void Divide(DenseTensor<sbyte> tensor, sbyte scalar, DenseTensor<sbyte> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (sbyte)(tensorSpan[i] / scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (sbyte)(tensorSpan[op1Index] / scalar);

                }
            }
        }
        public void Equals(DenseTensor<sbyte> left, DenseTensor<sbyte> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] == rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] == rightSpan[op2Index];

                }
            }
        }
        public void GreaterThan(DenseTensor<sbyte> left, DenseTensor<sbyte> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] > rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] > rightSpan[op2Index];

                }
            }
        }
        public void GreaterThanOrEqual(DenseTensor<sbyte> left, DenseTensor<sbyte> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] >= rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] >= rightSpan[op2Index];

                }
            }
        }
        public void Increment(DenseTensor<sbyte> tensor, DenseTensor<sbyte> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            for(int i = 0; i < resultSpan.Length; i++)
            {
                resultSpan[i]++;
            }
        }
        public void LeftShift(DenseTensor<sbyte> tensor, int value, DenseTensor<sbyte> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (sbyte)(tensorSpan[i] << value);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (sbyte)(tensorSpan[op1Index] << value);

                }
            }
        }
        public void LessThan(DenseTensor<sbyte> left, DenseTensor<sbyte> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] < rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] < rightSpan[op2Index];

                }
            }
        }
        public void LessThanOrEqual(DenseTensor<sbyte> left, DenseTensor<sbyte> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] <= rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] <= rightSpan[op2Index];

                }
            }
        }
        public void Modulo(DenseTensor<sbyte> left, DenseTensor<sbyte> right, DenseTensor<sbyte> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (sbyte)(leftSpan[i] % rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (sbyte)(leftSpan[op1Index] % rightSpan[op2Index]);

                }
            }
        }
        public void Modulo(DenseTensor<sbyte> tensor, sbyte scalar, DenseTensor<sbyte> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (sbyte)(tensorSpan[i] % scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (sbyte)(tensorSpan[op1Index] % scalar);

                }
            }
        }
        public void Multiply(DenseTensor<sbyte> left, DenseTensor<sbyte> right, DenseTensor<sbyte> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (sbyte)(leftSpan[i] * rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (sbyte)(leftSpan[op1Index] * rightSpan[op2Index]);

                }
            }
        }
        public void Multiply(DenseTensor<sbyte> tensor, sbyte scalar, DenseTensor<sbyte> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (sbyte)(tensorSpan[i] * scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (sbyte)(tensorSpan[op1Index] * scalar);

                }
            }
        }
        public void NotEquals(DenseTensor<sbyte> left, DenseTensor<sbyte> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] != rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] != rightSpan[op2Index];

                }
            }
        }
        public void Or(DenseTensor<sbyte> left, DenseTensor<sbyte> right, DenseTensor<sbyte> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (sbyte)(leftSpan[i] | rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (sbyte)(leftSpan[op1Index] | rightSpan[op2Index]);

                }
            }
        }
        public void Or(DenseTensor<sbyte> tensor, sbyte scalar, DenseTensor<sbyte> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (sbyte)(tensorSpan[i] | scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (sbyte)(tensorSpan[op1Index] | scalar);

                }
            }
        }
        public void RightShift(DenseTensor<sbyte> tensor, int value, DenseTensor<sbyte> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (sbyte)(tensorSpan[i] >> value);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (sbyte)(tensorSpan[op1Index] >> value);

                }
            }
        }
        public void Subtract(DenseTensor<sbyte> left, DenseTensor<sbyte> right, DenseTensor<sbyte> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (sbyte)(leftSpan[i] - rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (sbyte)(leftSpan[op1Index] - rightSpan[op2Index]);

                }
            }
        }
        public void Subtract(DenseTensor<sbyte> tensor, sbyte scalar, DenseTensor<sbyte> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (sbyte)(tensorSpan[i] - scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (sbyte)(tensorSpan[op1Index] - scalar);

                }
            }
        }
        public void UnaryMinus(DenseTensor<sbyte> tensor, DenseTensor<sbyte> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (sbyte)-tensorSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (sbyte)-tensorSpan[op1Index];

                }
            }
        }
        public void UnaryPlus(DenseTensor<sbyte> tensor, DenseTensor<sbyte> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (sbyte)+tensorSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (sbyte)+tensorSpan[op1Index];

                }
            }
        }
        public void Xor(DenseTensor<sbyte> left, DenseTensor<sbyte> right, DenseTensor<sbyte> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (sbyte)(leftSpan[i] ^ rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (sbyte)(leftSpan[op1Index] ^ rightSpan[op2Index]);

                }
            }
        }
        public void Xor(DenseTensor<sbyte> tensor, sbyte scalar, DenseTensor<sbyte> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (sbyte)(tensorSpan[i] ^ scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (sbyte)(tensorSpan[op1Index] ^ scalar);

                }
            }
        }
    }
    internal class ShortArithmetic : ITensorArithmetic<short>
    {
        public short One => 1;
        public short Zero => 0;

        public void Add(Tensor<short> left, Tensor<short> right, Tensor<short> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (short)(left[indices] + right[indices]);
            }
            
        }
        public void Add(Tensor<short> tensor, short scalar, Tensor<short> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (short)(tensor[indices] + scalar);
            }
            
        }
        public void And(Tensor<short> left, Tensor<short> right, Tensor<short> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (short)(left[indices] & right[indices]);
            }
            
        }
        public void And(Tensor<short> tensor, short scalar, Tensor<short> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (short)(tensor[indices] & scalar);
            }
            
        }
        public void Contract(Tensor<short> left, Tensor<short> right, int[] leftAxes, int[] rightAxes, Tensor<short> result)
        {
            var leftIndices = new int[left.Rank];
            var rightIndices = new int[right.Rank];
            var resultIndices = new int[result.Rank];

            var summingDimensions = new int[leftAxes.Length];
            for(int i = 0; i < leftAxes.Length; i++)
            {
                summingDimensions[i] = left.dimensions[leftAxes[i]];
            }

            var summingStrides = ArrayUtilities.GetStrides(summingDimensions);
            int summingLength = (int)ArrayUtilities.GetProduct(summingDimensions);

            var resultStrides = result.strides;

            // translates from result index to left non-summing dimensions' index portion
            // since left non-summing dimensions are given precedence in result, the end is zero-padded
            int[] leftNonSummingStrides = new int[result.Rank];

            // translates from summing index to left summing dimensions' index portion
            int[] leftSummingStrides = new int[leftAxes.Length];
            ArrayUtilities.SplitStrides(left.strides, leftAxes, leftNonSummingStrides, 0, leftSummingStrides, 0);

            // translates from result index to right non-summing dimensions' index portion
            int[] rightNonSummingStrides = new int[result.Rank];
            //  right non-summing dimensions appear after left non-summing dimensions.
            int rightNonSummingStridesOffset = (left.Rank - leftAxes.Length);

            // translates from summing index to right summing dimensions' index portion
            int[] rightSummingStrides = new int[rightAxes.Length];
            ArrayUtilities.SplitStrides(right.strides, rightAxes, rightNonSummingStrides, rightNonSummingStridesOffset, rightSummingStrides, 0);

            for (int resultIndex = 0; resultIndex < result.Length; resultIndex++)
            {
                short sum = (short)0;
                
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, resultIndex, resultIndices);

                int leftIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, leftNonSummingStrides);
                int rightIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, rightNonSummingStrides);

                for (int summingIndex = 0; summingIndex < summingLength; summingIndex++)
                {
                    int leftIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, leftSummingStrides);
                    int rightIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, rightSummingStrides);

                    int leftIndex = leftIndexNonSumming + leftIndexSumming;
                    int rightIndex = rightIndexNonSumming + rightIndexSumming;

                    // todo, make this more efficient
                    ArrayUtilities.GetIndices(left.strides, left.IsReversedStride, leftIndex, leftIndices);
                    ArrayUtilities.GetIndices(right.strides, right.IsReversedStride, rightIndex, rightIndices);

                    sum += (short)(left[leftIndices] * right[rightIndices]);
                }
                
                result[resultIndices] = sum;
            }
        }
        public void Decrement(Tensor<short> tensor, Tensor<short> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices]--;
            }
            
        }
        public void Divide(Tensor<short> left, Tensor<short> right, Tensor<short> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (short)(left[indices] / right[indices]);
            }
            
        }
        public void Divide(Tensor<short> tensor, short scalar, Tensor<short> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (short)(tensor[indices] / scalar);
            }
            
        }
        public void Equals(Tensor<short> left, Tensor<short> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] == right[indices];
            }
            
        }
        public void GreaterThan(Tensor<short> left, Tensor<short> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] > right[indices];
            }
            
        }
        public void GreaterThanOrEqual(Tensor<short> left, Tensor<short> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] >= right[indices];
            }
            
        }
        public void Increment(Tensor<short> tensor, Tensor<short> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices]++;
            }
            
        }
        public void LeftShift(Tensor<short> tensor, int value, Tensor<short> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (short)(tensor[indices] << value);
            }
            
        }
        public void LessThan(Tensor<short> left, Tensor<short> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] < right[indices];
            }
            
        }
        public void LessThanOrEqual(Tensor<short> left, Tensor<short> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] <= right[indices];
            }
            
        }
        public void Modulo(Tensor<short> left, Tensor<short> right, Tensor<short> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (short)(left[indices] % right[indices]);
            }
            
        }
        public void Modulo(Tensor<short> tensor, short scalar, Tensor<short> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (short)(tensor[indices] % scalar);
            }
            
        }
        public void Multiply(Tensor<short> left, Tensor<short> right, Tensor<short> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (short)(left[indices] * right[indices]);
            }
            
        }
        public void Multiply(Tensor<short> tensor, short scalar, Tensor<short> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (short)(tensor[indices] * scalar);
            }
            
        }
        public void NotEquals(Tensor<short> left, Tensor<short> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] != right[indices];
            }
            
        }
        public void Or(Tensor<short> left, Tensor<short> right, Tensor<short> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (short)(left[indices] | right[indices]);
            }
            
        }
        public void Or(Tensor<short> tensor, short scalar, Tensor<short> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (short)(tensor[indices] | scalar);
            }
            
        }
        public void RightShift(Tensor<short> tensor, int value, Tensor<short> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (short)(tensor[indices] >> value);
            }
            
        }
        public void Subtract(Tensor<short> left, Tensor<short> right, Tensor<short> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (short)(left[indices] - right[indices]);
            }
            
        }
        public void Subtract(Tensor<short> tensor, short scalar, Tensor<short> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (short)(tensor[indices] - scalar);
            }
            
        }
        public void UnaryMinus(Tensor<short> tensor, Tensor<short> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (short)-tensor[indices];
            }
            
        }
        public void UnaryPlus(Tensor<short> tensor, Tensor<short> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (short)+tensor[indices];
            }
            
        }
        public void Xor(Tensor<short> left, Tensor<short> right, Tensor<short> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (short)(left[indices] ^ right[indices]);
            }
            
        }
        public void Xor(Tensor<short> tensor, short scalar, Tensor<short> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (short)(tensor[indices] ^ scalar);
            }
            
        }

        public void Add(DenseTensor<short> left, DenseTensor<short> right, DenseTensor<short> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (short)(leftSpan[i] + rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (short)(leftSpan[op1Index] + rightSpan[op2Index]);

                }
            }
        }
        public void Add(DenseTensor<short> tensor, short scalar, DenseTensor<short> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (short)(tensorSpan[i] + scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (short)(tensorSpan[op1Index] + scalar);

                }
            }
        }
        public void And(DenseTensor<short> left, DenseTensor<short> right, DenseTensor<short> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (short)(leftSpan[i] & rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (short)(leftSpan[op1Index] & rightSpan[op2Index]);

                }
            }
        }
        public void And(DenseTensor<short> tensor, short scalar, DenseTensor<short> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (short)(tensorSpan[i] & scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (short)(tensorSpan[op1Index] & scalar);

                }
            }
        }
        public void Contract(DenseTensor<short> left, DenseTensor<short> right, int[] leftAxes, int[] rightAxes, DenseTensor<short> result)
        {
            var summingDimensions = new int[leftAxes.Length];
            for(int i = 0; i < leftAxes.Length; i++)
            {
                summingDimensions[i] = left.dimensions[leftAxes[i]];
            }

            var summingStrides = ArrayUtilities.GetStrides(summingDimensions);
            int summingLength = (int)ArrayUtilities.GetProduct(summingDimensions);

            var resultStrides = result.strides;

            // translates from result index to left non-summing dimensions' index portion
            // since left non-summing dimensions are given precedence in result, the end is zero-padded
            int[] leftNonSummingStrides = new int[result.Rank];

            // translates from summing index to left summing dimensions' index portion
            int[] leftSummingStrides = new int[leftAxes.Length];
            ArrayUtilities.SplitStrides(left.strides, leftAxes, leftNonSummingStrides, 0, leftSummingStrides, 0);

            // translates from result index to right non-summing dimensions' index portion
            int[] rightNonSummingStrides = new int[result.Rank];
            //  right non-summing dimensions appear after left non-summing dimensions.
            int rightNonSummingStridesOffset = (left.Rank - leftAxes.Length);

            // translates from summing index to right summing dimensions' index portion
            int[] rightSummingStrides = new int[rightAxes.Length];
            ArrayUtilities.SplitStrides(right.strides, rightAxes, rightNonSummingStrides, rightNonSummingStridesOffset, rightSummingStrides, 0);
            
            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;

            for (int resultIndex = 0; resultIndex < resultSpan.Length; resultIndex++)
            {
                short sum = (short)0;

                int leftIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, leftNonSummingStrides);
                int rightIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, rightNonSummingStrides);

                for (int summingIndex = 0; summingIndex < summingLength; summingIndex++)
                {
                    int leftIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, leftSummingStrides);
                    int rightIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, rightSummingStrides);

                    int leftIndex = leftIndexNonSumming + leftIndexSumming;
                    int rightIndex = rightIndexNonSumming + rightIndexSumming;

                    sum += (short)(leftSpan[leftIndex] * rightSpan[rightIndex]);
                }

                resultSpan[resultIndex] = sum;
            }
        }
        public void Decrement(DenseTensor<short> tensor, DenseTensor<short> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            for(int i = 0; i < resultSpan.Length; i++)
            {
                resultSpan[i]--;
            }
        }
        public void Divide(DenseTensor<short> left, DenseTensor<short> right, DenseTensor<short> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (short)(leftSpan[i] / rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (short)(leftSpan[op1Index] / rightSpan[op2Index]);

                }
            }
        }
        public void Divide(DenseTensor<short> tensor, short scalar, DenseTensor<short> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (short)(tensorSpan[i] / scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (short)(tensorSpan[op1Index] / scalar);

                }
            }
        }
        public void Equals(DenseTensor<short> left, DenseTensor<short> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] == rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] == rightSpan[op2Index];

                }
            }
        }
        public void GreaterThan(DenseTensor<short> left, DenseTensor<short> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] > rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] > rightSpan[op2Index];

                }
            }
        }
        public void GreaterThanOrEqual(DenseTensor<short> left, DenseTensor<short> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] >= rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] >= rightSpan[op2Index];

                }
            }
        }
        public void Increment(DenseTensor<short> tensor, DenseTensor<short> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            for(int i = 0; i < resultSpan.Length; i++)
            {
                resultSpan[i]++;
            }
        }
        public void LeftShift(DenseTensor<short> tensor, int value, DenseTensor<short> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (short)(tensorSpan[i] << value);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (short)(tensorSpan[op1Index] << value);

                }
            }
        }
        public void LessThan(DenseTensor<short> left, DenseTensor<short> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] < rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] < rightSpan[op2Index];

                }
            }
        }
        public void LessThanOrEqual(DenseTensor<short> left, DenseTensor<short> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] <= rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] <= rightSpan[op2Index];

                }
            }
        }
        public void Modulo(DenseTensor<short> left, DenseTensor<short> right, DenseTensor<short> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (short)(leftSpan[i] % rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (short)(leftSpan[op1Index] % rightSpan[op2Index]);

                }
            }
        }
        public void Modulo(DenseTensor<short> tensor, short scalar, DenseTensor<short> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (short)(tensorSpan[i] % scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (short)(tensorSpan[op1Index] % scalar);

                }
            }
        }
        public void Multiply(DenseTensor<short> left, DenseTensor<short> right, DenseTensor<short> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (short)(leftSpan[i] * rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (short)(leftSpan[op1Index] * rightSpan[op2Index]);

                }
            }
        }
        public void Multiply(DenseTensor<short> tensor, short scalar, DenseTensor<short> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (short)(tensorSpan[i] * scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (short)(tensorSpan[op1Index] * scalar);

                }
            }
        }
        public void NotEquals(DenseTensor<short> left, DenseTensor<short> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] != rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] != rightSpan[op2Index];

                }
            }
        }
        public void Or(DenseTensor<short> left, DenseTensor<short> right, DenseTensor<short> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (short)(leftSpan[i] | rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (short)(leftSpan[op1Index] | rightSpan[op2Index]);

                }
            }
        }
        public void Or(DenseTensor<short> tensor, short scalar, DenseTensor<short> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (short)(tensorSpan[i] | scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (short)(tensorSpan[op1Index] | scalar);

                }
            }
        }
        public void RightShift(DenseTensor<short> tensor, int value, DenseTensor<short> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (short)(tensorSpan[i] >> value);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (short)(tensorSpan[op1Index] >> value);

                }
            }
        }
        public void Subtract(DenseTensor<short> left, DenseTensor<short> right, DenseTensor<short> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (short)(leftSpan[i] - rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (short)(leftSpan[op1Index] - rightSpan[op2Index]);

                }
            }
        }
        public void Subtract(DenseTensor<short> tensor, short scalar, DenseTensor<short> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (short)(tensorSpan[i] - scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (short)(tensorSpan[op1Index] - scalar);

                }
            }
        }
        public void UnaryMinus(DenseTensor<short> tensor, DenseTensor<short> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (short)-tensorSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (short)-tensorSpan[op1Index];

                }
            }
        }
        public void UnaryPlus(DenseTensor<short> tensor, DenseTensor<short> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (short)+tensorSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (short)+tensorSpan[op1Index];

                }
            }
        }
        public void Xor(DenseTensor<short> left, DenseTensor<short> right, DenseTensor<short> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (short)(leftSpan[i] ^ rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (short)(leftSpan[op1Index] ^ rightSpan[op2Index]);

                }
            }
        }
        public void Xor(DenseTensor<short> tensor, short scalar, DenseTensor<short> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (short)(tensorSpan[i] ^ scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (short)(tensorSpan[op1Index] ^ scalar);

                }
            }
        }
    }
    internal class UIntArithmetic : ITensorArithmetic<uint>
    {
        public uint One => 1;
        public uint Zero => 0;

        public void Add(Tensor<uint> left, Tensor<uint> right, Tensor<uint> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (uint)(left[indices] + right[indices]);
            }
            
        }
        public void Add(Tensor<uint> tensor, uint scalar, Tensor<uint> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (uint)(tensor[indices] + scalar);
            }
            
        }
        public void And(Tensor<uint> left, Tensor<uint> right, Tensor<uint> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (uint)(left[indices] & right[indices]);
            }
            
        }
        public void And(Tensor<uint> tensor, uint scalar, Tensor<uint> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (uint)(tensor[indices] & scalar);
            }
            
        }
        public void Contract(Tensor<uint> left, Tensor<uint> right, int[] leftAxes, int[] rightAxes, Tensor<uint> result)
        {
            var leftIndices = new int[left.Rank];
            var rightIndices = new int[right.Rank];
            var resultIndices = new int[result.Rank];

            var summingDimensions = new int[leftAxes.Length];
            for(int i = 0; i < leftAxes.Length; i++)
            {
                summingDimensions[i] = left.dimensions[leftAxes[i]];
            }

            var summingStrides = ArrayUtilities.GetStrides(summingDimensions);
            int summingLength = (int)ArrayUtilities.GetProduct(summingDimensions);

            var resultStrides = result.strides;

            // translates from result index to left non-summing dimensions' index portion
            // since left non-summing dimensions are given precedence in result, the end is zero-padded
            int[] leftNonSummingStrides = new int[result.Rank];

            // translates from summing index to left summing dimensions' index portion
            int[] leftSummingStrides = new int[leftAxes.Length];
            ArrayUtilities.SplitStrides(left.strides, leftAxes, leftNonSummingStrides, 0, leftSummingStrides, 0);

            // translates from result index to right non-summing dimensions' index portion
            int[] rightNonSummingStrides = new int[result.Rank];
            //  right non-summing dimensions appear after left non-summing dimensions.
            int rightNonSummingStridesOffset = (left.Rank - leftAxes.Length);

            // translates from summing index to right summing dimensions' index portion
            int[] rightSummingStrides = new int[rightAxes.Length];
            ArrayUtilities.SplitStrides(right.strides, rightAxes, rightNonSummingStrides, rightNonSummingStridesOffset, rightSummingStrides, 0);

            for (int resultIndex = 0; resultIndex < result.Length; resultIndex++)
            {
                uint sum = (uint)0;
                
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, resultIndex, resultIndices);

                int leftIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, leftNonSummingStrides);
                int rightIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, rightNonSummingStrides);

                for (int summingIndex = 0; summingIndex < summingLength; summingIndex++)
                {
                    int leftIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, leftSummingStrides);
                    int rightIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, rightSummingStrides);

                    int leftIndex = leftIndexNonSumming + leftIndexSumming;
                    int rightIndex = rightIndexNonSumming + rightIndexSumming;

                    // todo, make this more efficient
                    ArrayUtilities.GetIndices(left.strides, left.IsReversedStride, leftIndex, leftIndices);
                    ArrayUtilities.GetIndices(right.strides, right.IsReversedStride, rightIndex, rightIndices);

                    sum += (uint)(left[leftIndices] * right[rightIndices]);
                }
                
                result[resultIndices] = sum;
            }
        }
        public void Decrement(Tensor<uint> tensor, Tensor<uint> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices]--;
            }
            
        }
        public void Divide(Tensor<uint> left, Tensor<uint> right, Tensor<uint> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (uint)(left[indices] / right[indices]);
            }
            
        }
        public void Divide(Tensor<uint> tensor, uint scalar, Tensor<uint> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (uint)(tensor[indices] / scalar);
            }
            
        }
        public void Equals(Tensor<uint> left, Tensor<uint> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] == right[indices];
            }
            
        }
        public void GreaterThan(Tensor<uint> left, Tensor<uint> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] > right[indices];
            }
            
        }
        public void GreaterThanOrEqual(Tensor<uint> left, Tensor<uint> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] >= right[indices];
            }
            
        }
        public void Increment(Tensor<uint> tensor, Tensor<uint> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices]++;
            }
            
        }
        public void LeftShift(Tensor<uint> tensor, int value, Tensor<uint> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (uint)(tensor[indices] << value);
            }
            
        }
        public void LessThan(Tensor<uint> left, Tensor<uint> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] < right[indices];
            }
            
        }
        public void LessThanOrEqual(Tensor<uint> left, Tensor<uint> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] <= right[indices];
            }
            
        }
        public void Modulo(Tensor<uint> left, Tensor<uint> right, Tensor<uint> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (uint)(left[indices] % right[indices]);
            }
            
        }
        public void Modulo(Tensor<uint> tensor, uint scalar, Tensor<uint> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (uint)(tensor[indices] % scalar);
            }
            
        }
        public void Multiply(Tensor<uint> left, Tensor<uint> right, Tensor<uint> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (uint)(left[indices] * right[indices]);
            }
            
        }
        public void Multiply(Tensor<uint> tensor, uint scalar, Tensor<uint> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (uint)(tensor[indices] * scalar);
            }
            
        }
        public void NotEquals(Tensor<uint> left, Tensor<uint> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] != right[indices];
            }
            
        }
        public void Or(Tensor<uint> left, Tensor<uint> right, Tensor<uint> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (uint)(left[indices] | right[indices]);
            }
            
        }
        public void Or(Tensor<uint> tensor, uint scalar, Tensor<uint> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (uint)(tensor[indices] | scalar);
            }
            
        }
        public void RightShift(Tensor<uint> tensor, int value, Tensor<uint> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (uint)(tensor[indices] >> value);
            }
            
        }
        public void Subtract(Tensor<uint> left, Tensor<uint> right, Tensor<uint> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (uint)(left[indices] - right[indices]);
            }
            
        }
        public void Subtract(Tensor<uint> tensor, uint scalar, Tensor<uint> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (uint)(tensor[indices] - scalar);
            }
            
        }
        public void UnaryMinus(Tensor<uint> tensor, Tensor<uint> result)
        {
            throw new NotSupportedException();
        }
        public void UnaryPlus(Tensor<uint> tensor, Tensor<uint> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (uint)+tensor[indices];
            }
            
        }
        public void Xor(Tensor<uint> left, Tensor<uint> right, Tensor<uint> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (uint)(left[indices] ^ right[indices]);
            }
            
        }
        public void Xor(Tensor<uint> tensor, uint scalar, Tensor<uint> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (uint)(tensor[indices] ^ scalar);
            }
            
        }

        public void Add(DenseTensor<uint> left, DenseTensor<uint> right, DenseTensor<uint> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (uint)(leftSpan[i] + rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (uint)(leftSpan[op1Index] + rightSpan[op2Index]);

                }
            }
        }
        public void Add(DenseTensor<uint> tensor, uint scalar, DenseTensor<uint> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (uint)(tensorSpan[i] + scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (uint)(tensorSpan[op1Index] + scalar);

                }
            }
        }
        public void And(DenseTensor<uint> left, DenseTensor<uint> right, DenseTensor<uint> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (uint)(leftSpan[i] & rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (uint)(leftSpan[op1Index] & rightSpan[op2Index]);

                }
            }
        }
        public void And(DenseTensor<uint> tensor, uint scalar, DenseTensor<uint> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (uint)(tensorSpan[i] & scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (uint)(tensorSpan[op1Index] & scalar);

                }
            }
        }
        public void Contract(DenseTensor<uint> left, DenseTensor<uint> right, int[] leftAxes, int[] rightAxes, DenseTensor<uint> result)
        {
            var summingDimensions = new int[leftAxes.Length];
            for(int i = 0; i < leftAxes.Length; i++)
            {
                summingDimensions[i] = left.dimensions[leftAxes[i]];
            }

            var summingStrides = ArrayUtilities.GetStrides(summingDimensions);
            int summingLength = (int)ArrayUtilities.GetProduct(summingDimensions);

            var resultStrides = result.strides;

            // translates from result index to left non-summing dimensions' index portion
            // since left non-summing dimensions are given precedence in result, the end is zero-padded
            int[] leftNonSummingStrides = new int[result.Rank];

            // translates from summing index to left summing dimensions' index portion
            int[] leftSummingStrides = new int[leftAxes.Length];
            ArrayUtilities.SplitStrides(left.strides, leftAxes, leftNonSummingStrides, 0, leftSummingStrides, 0);

            // translates from result index to right non-summing dimensions' index portion
            int[] rightNonSummingStrides = new int[result.Rank];
            //  right non-summing dimensions appear after left non-summing dimensions.
            int rightNonSummingStridesOffset = (left.Rank - leftAxes.Length);

            // translates from summing index to right summing dimensions' index portion
            int[] rightSummingStrides = new int[rightAxes.Length];
            ArrayUtilities.SplitStrides(right.strides, rightAxes, rightNonSummingStrides, rightNonSummingStridesOffset, rightSummingStrides, 0);
            
            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;

            for (int resultIndex = 0; resultIndex < resultSpan.Length; resultIndex++)
            {
                uint sum = (uint)0;

                int leftIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, leftNonSummingStrides);
                int rightIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, rightNonSummingStrides);

                for (int summingIndex = 0; summingIndex < summingLength; summingIndex++)
                {
                    int leftIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, leftSummingStrides);
                    int rightIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, rightSummingStrides);

                    int leftIndex = leftIndexNonSumming + leftIndexSumming;
                    int rightIndex = rightIndexNonSumming + rightIndexSumming;

                    sum += (uint)(leftSpan[leftIndex] * rightSpan[rightIndex]);
                }

                resultSpan[resultIndex] = sum;
            }
        }
        public void Decrement(DenseTensor<uint> tensor, DenseTensor<uint> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            for(int i = 0; i < resultSpan.Length; i++)
            {
                resultSpan[i]--;
            }
        }
        public void Divide(DenseTensor<uint> left, DenseTensor<uint> right, DenseTensor<uint> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (uint)(leftSpan[i] / rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (uint)(leftSpan[op1Index] / rightSpan[op2Index]);

                }
            }
        }
        public void Divide(DenseTensor<uint> tensor, uint scalar, DenseTensor<uint> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (uint)(tensorSpan[i] / scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (uint)(tensorSpan[op1Index] / scalar);

                }
            }
        }
        public void Equals(DenseTensor<uint> left, DenseTensor<uint> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] == rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] == rightSpan[op2Index];

                }
            }
        }
        public void GreaterThan(DenseTensor<uint> left, DenseTensor<uint> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] > rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] > rightSpan[op2Index];

                }
            }
        }
        public void GreaterThanOrEqual(DenseTensor<uint> left, DenseTensor<uint> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] >= rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] >= rightSpan[op2Index];

                }
            }
        }
        public void Increment(DenseTensor<uint> tensor, DenseTensor<uint> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            for(int i = 0; i < resultSpan.Length; i++)
            {
                resultSpan[i]++;
            }
        }
        public void LeftShift(DenseTensor<uint> tensor, int value, DenseTensor<uint> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (uint)(tensorSpan[i] << value);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (uint)(tensorSpan[op1Index] << value);

                }
            }
        }
        public void LessThan(DenseTensor<uint> left, DenseTensor<uint> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] < rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] < rightSpan[op2Index];

                }
            }
        }
        public void LessThanOrEqual(DenseTensor<uint> left, DenseTensor<uint> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] <= rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] <= rightSpan[op2Index];

                }
            }
        }
        public void Modulo(DenseTensor<uint> left, DenseTensor<uint> right, DenseTensor<uint> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (uint)(leftSpan[i] % rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (uint)(leftSpan[op1Index] % rightSpan[op2Index]);

                }
            }
        }
        public void Modulo(DenseTensor<uint> tensor, uint scalar, DenseTensor<uint> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (uint)(tensorSpan[i] % scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (uint)(tensorSpan[op1Index] % scalar);

                }
            }
        }
        public void Multiply(DenseTensor<uint> left, DenseTensor<uint> right, DenseTensor<uint> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (uint)(leftSpan[i] * rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (uint)(leftSpan[op1Index] * rightSpan[op2Index]);

                }
            }
        }
        public void Multiply(DenseTensor<uint> tensor, uint scalar, DenseTensor<uint> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (uint)(tensorSpan[i] * scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (uint)(tensorSpan[op1Index] * scalar);

                }
            }
        }
        public void NotEquals(DenseTensor<uint> left, DenseTensor<uint> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] != rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] != rightSpan[op2Index];

                }
            }
        }
        public void Or(DenseTensor<uint> left, DenseTensor<uint> right, DenseTensor<uint> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (uint)(leftSpan[i] | rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (uint)(leftSpan[op1Index] | rightSpan[op2Index]);

                }
            }
        }
        public void Or(DenseTensor<uint> tensor, uint scalar, DenseTensor<uint> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (uint)(tensorSpan[i] | scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (uint)(tensorSpan[op1Index] | scalar);

                }
            }
        }
        public void RightShift(DenseTensor<uint> tensor, int value, DenseTensor<uint> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (uint)(tensorSpan[i] >> value);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (uint)(tensorSpan[op1Index] >> value);

                }
            }
        }
        public void Subtract(DenseTensor<uint> left, DenseTensor<uint> right, DenseTensor<uint> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (uint)(leftSpan[i] - rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (uint)(leftSpan[op1Index] - rightSpan[op2Index]);

                }
            }
        }
        public void Subtract(DenseTensor<uint> tensor, uint scalar, DenseTensor<uint> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (uint)(tensorSpan[i] - scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (uint)(tensorSpan[op1Index] - scalar);

                }
            }
        }
        public void UnaryMinus(DenseTensor<uint> tensor, DenseTensor<uint> result)
        {
            throw new NotSupportedException();
        }
        public void UnaryPlus(DenseTensor<uint> tensor, DenseTensor<uint> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (uint)+tensorSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (uint)+tensorSpan[op1Index];

                }
            }
        }
        public void Xor(DenseTensor<uint> left, DenseTensor<uint> right, DenseTensor<uint> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (uint)(leftSpan[i] ^ rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (uint)(leftSpan[op1Index] ^ rightSpan[op2Index]);

                }
            }
        }
        public void Xor(DenseTensor<uint> tensor, uint scalar, DenseTensor<uint> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (uint)(tensorSpan[i] ^ scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (uint)(tensorSpan[op1Index] ^ scalar);

                }
            }
        }
    }
    internal class ULongArithmetic : ITensorArithmetic<ulong>
    {
        public ulong One => 1;
        public ulong Zero => 0;

        public void Add(Tensor<ulong> left, Tensor<ulong> right, Tensor<ulong> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ulong)(left[indices] + right[indices]);
            }
            
        }
        public void Add(Tensor<ulong> tensor, ulong scalar, Tensor<ulong> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ulong)(tensor[indices] + scalar);
            }
            
        }
        public void And(Tensor<ulong> left, Tensor<ulong> right, Tensor<ulong> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ulong)(left[indices] & right[indices]);
            }
            
        }
        public void And(Tensor<ulong> tensor, ulong scalar, Tensor<ulong> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ulong)(tensor[indices] & scalar);
            }
            
        }
        public void Contract(Tensor<ulong> left, Tensor<ulong> right, int[] leftAxes, int[] rightAxes, Tensor<ulong> result)
        {
            var leftIndices = new int[left.Rank];
            var rightIndices = new int[right.Rank];
            var resultIndices = new int[result.Rank];

            var summingDimensions = new int[leftAxes.Length];
            for(int i = 0; i < leftAxes.Length; i++)
            {
                summingDimensions[i] = left.dimensions[leftAxes[i]];
            }

            var summingStrides = ArrayUtilities.GetStrides(summingDimensions);
            int summingLength = (int)ArrayUtilities.GetProduct(summingDimensions);

            var resultStrides = result.strides;

            // translates from result index to left non-summing dimensions' index portion
            // since left non-summing dimensions are given precedence in result, the end is zero-padded
            int[] leftNonSummingStrides = new int[result.Rank];

            // translates from summing index to left summing dimensions' index portion
            int[] leftSummingStrides = new int[leftAxes.Length];
            ArrayUtilities.SplitStrides(left.strides, leftAxes, leftNonSummingStrides, 0, leftSummingStrides, 0);

            // translates from result index to right non-summing dimensions' index portion
            int[] rightNonSummingStrides = new int[result.Rank];
            //  right non-summing dimensions appear after left non-summing dimensions.
            int rightNonSummingStridesOffset = (left.Rank - leftAxes.Length);

            // translates from summing index to right summing dimensions' index portion
            int[] rightSummingStrides = new int[rightAxes.Length];
            ArrayUtilities.SplitStrides(right.strides, rightAxes, rightNonSummingStrides, rightNonSummingStridesOffset, rightSummingStrides, 0);

            for (int resultIndex = 0; resultIndex < result.Length; resultIndex++)
            {
                ulong sum = (ulong)0;
                
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, resultIndex, resultIndices);

                int leftIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, leftNonSummingStrides);
                int rightIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, rightNonSummingStrides);

                for (int summingIndex = 0; summingIndex < summingLength; summingIndex++)
                {
                    int leftIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, leftSummingStrides);
                    int rightIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, rightSummingStrides);

                    int leftIndex = leftIndexNonSumming + leftIndexSumming;
                    int rightIndex = rightIndexNonSumming + rightIndexSumming;

                    // todo, make this more efficient
                    ArrayUtilities.GetIndices(left.strides, left.IsReversedStride, leftIndex, leftIndices);
                    ArrayUtilities.GetIndices(right.strides, right.IsReversedStride, rightIndex, rightIndices);

                    sum += (ulong)(left[leftIndices] * right[rightIndices]);
                }
                
                result[resultIndices] = sum;
            }
        }
        public void Decrement(Tensor<ulong> tensor, Tensor<ulong> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices]--;
            }
            
        }
        public void Divide(Tensor<ulong> left, Tensor<ulong> right, Tensor<ulong> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ulong)(left[indices] / right[indices]);
            }
            
        }
        public void Divide(Tensor<ulong> tensor, ulong scalar, Tensor<ulong> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ulong)(tensor[indices] / scalar);
            }
            
        }
        public void Equals(Tensor<ulong> left, Tensor<ulong> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] == right[indices];
            }
            
        }
        public void GreaterThan(Tensor<ulong> left, Tensor<ulong> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] > right[indices];
            }
            
        }
        public void GreaterThanOrEqual(Tensor<ulong> left, Tensor<ulong> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] >= right[indices];
            }
            
        }
        public void Increment(Tensor<ulong> tensor, Tensor<ulong> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices]++;
            }
            
        }
        public void LeftShift(Tensor<ulong> tensor, int value, Tensor<ulong> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ulong)(tensor[indices] << value);
            }
            
        }
        public void LessThan(Tensor<ulong> left, Tensor<ulong> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] < right[indices];
            }
            
        }
        public void LessThanOrEqual(Tensor<ulong> left, Tensor<ulong> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] <= right[indices];
            }
            
        }
        public void Modulo(Tensor<ulong> left, Tensor<ulong> right, Tensor<ulong> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ulong)(left[indices] % right[indices]);
            }
            
        }
        public void Modulo(Tensor<ulong> tensor, ulong scalar, Tensor<ulong> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ulong)(tensor[indices] % scalar);
            }
            
        }
        public void Multiply(Tensor<ulong> left, Tensor<ulong> right, Tensor<ulong> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ulong)(left[indices] * right[indices]);
            }
            
        }
        public void Multiply(Tensor<ulong> tensor, ulong scalar, Tensor<ulong> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ulong)(tensor[indices] * scalar);
            }
            
        }
        public void NotEquals(Tensor<ulong> left, Tensor<ulong> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] != right[indices];
            }
            
        }
        public void Or(Tensor<ulong> left, Tensor<ulong> right, Tensor<ulong> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ulong)(left[indices] | right[indices]);
            }
            
        }
        public void Or(Tensor<ulong> tensor, ulong scalar, Tensor<ulong> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ulong)(tensor[indices] | scalar);
            }
            
        }
        public void RightShift(Tensor<ulong> tensor, int value, Tensor<ulong> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ulong)(tensor[indices] >> value);
            }
            
        }
        public void Subtract(Tensor<ulong> left, Tensor<ulong> right, Tensor<ulong> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ulong)(left[indices] - right[indices]);
            }
            
        }
        public void Subtract(Tensor<ulong> tensor, ulong scalar, Tensor<ulong> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ulong)(tensor[indices] - scalar);
            }
            
        }
        public void UnaryMinus(Tensor<ulong> tensor, Tensor<ulong> result)
        {
            throw new NotSupportedException();
        }
        public void UnaryPlus(Tensor<ulong> tensor, Tensor<ulong> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ulong)+tensor[indices];
            }
            
        }
        public void Xor(Tensor<ulong> left, Tensor<ulong> right, Tensor<ulong> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ulong)(left[indices] ^ right[indices]);
            }
            
        }
        public void Xor(Tensor<ulong> tensor, ulong scalar, Tensor<ulong> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ulong)(tensor[indices] ^ scalar);
            }
            
        }

        public void Add(DenseTensor<ulong> left, DenseTensor<ulong> right, DenseTensor<ulong> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ulong)(leftSpan[i] + rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ulong)(leftSpan[op1Index] + rightSpan[op2Index]);

                }
            }
        }
        public void Add(DenseTensor<ulong> tensor, ulong scalar, DenseTensor<ulong> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ulong)(tensorSpan[i] + scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ulong)(tensorSpan[op1Index] + scalar);

                }
            }
        }
        public void And(DenseTensor<ulong> left, DenseTensor<ulong> right, DenseTensor<ulong> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ulong)(leftSpan[i] & rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ulong)(leftSpan[op1Index] & rightSpan[op2Index]);

                }
            }
        }
        public void And(DenseTensor<ulong> tensor, ulong scalar, DenseTensor<ulong> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ulong)(tensorSpan[i] & scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ulong)(tensorSpan[op1Index] & scalar);

                }
            }
        }
        public void Contract(DenseTensor<ulong> left, DenseTensor<ulong> right, int[] leftAxes, int[] rightAxes, DenseTensor<ulong> result)
        {
            var summingDimensions = new int[leftAxes.Length];
            for(int i = 0; i < leftAxes.Length; i++)
            {
                summingDimensions[i] = left.dimensions[leftAxes[i]];
            }

            var summingStrides = ArrayUtilities.GetStrides(summingDimensions);
            int summingLength = (int)ArrayUtilities.GetProduct(summingDimensions);

            var resultStrides = result.strides;

            // translates from result index to left non-summing dimensions' index portion
            // since left non-summing dimensions are given precedence in result, the end is zero-padded
            int[] leftNonSummingStrides = new int[result.Rank];

            // translates from summing index to left summing dimensions' index portion
            int[] leftSummingStrides = new int[leftAxes.Length];
            ArrayUtilities.SplitStrides(left.strides, leftAxes, leftNonSummingStrides, 0, leftSummingStrides, 0);

            // translates from result index to right non-summing dimensions' index portion
            int[] rightNonSummingStrides = new int[result.Rank];
            //  right non-summing dimensions appear after left non-summing dimensions.
            int rightNonSummingStridesOffset = (left.Rank - leftAxes.Length);

            // translates from summing index to right summing dimensions' index portion
            int[] rightSummingStrides = new int[rightAxes.Length];
            ArrayUtilities.SplitStrides(right.strides, rightAxes, rightNonSummingStrides, rightNonSummingStridesOffset, rightSummingStrides, 0);
            
            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;

            for (int resultIndex = 0; resultIndex < resultSpan.Length; resultIndex++)
            {
                ulong sum = (ulong)0;

                int leftIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, leftNonSummingStrides);
                int rightIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, rightNonSummingStrides);

                for (int summingIndex = 0; summingIndex < summingLength; summingIndex++)
                {
                    int leftIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, leftSummingStrides);
                    int rightIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, rightSummingStrides);

                    int leftIndex = leftIndexNonSumming + leftIndexSumming;
                    int rightIndex = rightIndexNonSumming + rightIndexSumming;

                    sum += (ulong)(leftSpan[leftIndex] * rightSpan[rightIndex]);
                }

                resultSpan[resultIndex] = sum;
            }
        }
        public void Decrement(DenseTensor<ulong> tensor, DenseTensor<ulong> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            for(int i = 0; i < resultSpan.Length; i++)
            {
                resultSpan[i]--;
            }
        }
        public void Divide(DenseTensor<ulong> left, DenseTensor<ulong> right, DenseTensor<ulong> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ulong)(leftSpan[i] / rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ulong)(leftSpan[op1Index] / rightSpan[op2Index]);

                }
            }
        }
        public void Divide(DenseTensor<ulong> tensor, ulong scalar, DenseTensor<ulong> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ulong)(tensorSpan[i] / scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ulong)(tensorSpan[op1Index] / scalar);

                }
            }
        }
        public void Equals(DenseTensor<ulong> left, DenseTensor<ulong> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] == rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] == rightSpan[op2Index];

                }
            }
        }
        public void GreaterThan(DenseTensor<ulong> left, DenseTensor<ulong> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] > rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] > rightSpan[op2Index];

                }
            }
        }
        public void GreaterThanOrEqual(DenseTensor<ulong> left, DenseTensor<ulong> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] >= rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] >= rightSpan[op2Index];

                }
            }
        }
        public void Increment(DenseTensor<ulong> tensor, DenseTensor<ulong> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            for(int i = 0; i < resultSpan.Length; i++)
            {
                resultSpan[i]++;
            }
        }
        public void LeftShift(DenseTensor<ulong> tensor, int value, DenseTensor<ulong> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ulong)(tensorSpan[i] << value);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ulong)(tensorSpan[op1Index] << value);

                }
            }
        }
        public void LessThan(DenseTensor<ulong> left, DenseTensor<ulong> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] < rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] < rightSpan[op2Index];

                }
            }
        }
        public void LessThanOrEqual(DenseTensor<ulong> left, DenseTensor<ulong> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] <= rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] <= rightSpan[op2Index];

                }
            }
        }
        public void Modulo(DenseTensor<ulong> left, DenseTensor<ulong> right, DenseTensor<ulong> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ulong)(leftSpan[i] % rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ulong)(leftSpan[op1Index] % rightSpan[op2Index]);

                }
            }
        }
        public void Modulo(DenseTensor<ulong> tensor, ulong scalar, DenseTensor<ulong> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ulong)(tensorSpan[i] % scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ulong)(tensorSpan[op1Index] % scalar);

                }
            }
        }
        public void Multiply(DenseTensor<ulong> left, DenseTensor<ulong> right, DenseTensor<ulong> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ulong)(leftSpan[i] * rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ulong)(leftSpan[op1Index] * rightSpan[op2Index]);

                }
            }
        }
        public void Multiply(DenseTensor<ulong> tensor, ulong scalar, DenseTensor<ulong> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ulong)(tensorSpan[i] * scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ulong)(tensorSpan[op1Index] * scalar);

                }
            }
        }
        public void NotEquals(DenseTensor<ulong> left, DenseTensor<ulong> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] != rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] != rightSpan[op2Index];

                }
            }
        }
        public void Or(DenseTensor<ulong> left, DenseTensor<ulong> right, DenseTensor<ulong> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ulong)(leftSpan[i] | rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ulong)(leftSpan[op1Index] | rightSpan[op2Index]);

                }
            }
        }
        public void Or(DenseTensor<ulong> tensor, ulong scalar, DenseTensor<ulong> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ulong)(tensorSpan[i] | scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ulong)(tensorSpan[op1Index] | scalar);

                }
            }
        }
        public void RightShift(DenseTensor<ulong> tensor, int value, DenseTensor<ulong> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ulong)(tensorSpan[i] >> value);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ulong)(tensorSpan[op1Index] >> value);

                }
            }
        }
        public void Subtract(DenseTensor<ulong> left, DenseTensor<ulong> right, DenseTensor<ulong> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ulong)(leftSpan[i] - rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ulong)(leftSpan[op1Index] - rightSpan[op2Index]);

                }
            }
        }
        public void Subtract(DenseTensor<ulong> tensor, ulong scalar, DenseTensor<ulong> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ulong)(tensorSpan[i] - scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ulong)(tensorSpan[op1Index] - scalar);

                }
            }
        }
        public void UnaryMinus(DenseTensor<ulong> tensor, DenseTensor<ulong> result)
        {
            throw new NotSupportedException();
        }
        public void UnaryPlus(DenseTensor<ulong> tensor, DenseTensor<ulong> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ulong)+tensorSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ulong)+tensorSpan[op1Index];

                }
            }
        }
        public void Xor(DenseTensor<ulong> left, DenseTensor<ulong> right, DenseTensor<ulong> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ulong)(leftSpan[i] ^ rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ulong)(leftSpan[op1Index] ^ rightSpan[op2Index]);

                }
            }
        }
        public void Xor(DenseTensor<ulong> tensor, ulong scalar, DenseTensor<ulong> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ulong)(tensorSpan[i] ^ scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ulong)(tensorSpan[op1Index] ^ scalar);

                }
            }
        }
    }
    internal class UShortArithmetic : ITensorArithmetic<ushort>
    {
        public ushort One => 1;
        public ushort Zero => 0;

        public void Add(Tensor<ushort> left, Tensor<ushort> right, Tensor<ushort> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ushort)(left[indices] + right[indices]);
            }
            
        }
        public void Add(Tensor<ushort> tensor, ushort scalar, Tensor<ushort> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ushort)(tensor[indices] + scalar);
            }
            
        }
        public void And(Tensor<ushort> left, Tensor<ushort> right, Tensor<ushort> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ushort)(left[indices] & right[indices]);
            }
            
        }
        public void And(Tensor<ushort> tensor, ushort scalar, Tensor<ushort> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ushort)(tensor[indices] & scalar);
            }
            
        }
        public void Contract(Tensor<ushort> left, Tensor<ushort> right, int[] leftAxes, int[] rightAxes, Tensor<ushort> result)
        {
            var leftIndices = new int[left.Rank];
            var rightIndices = new int[right.Rank];
            var resultIndices = new int[result.Rank];

            var summingDimensions = new int[leftAxes.Length];
            for(int i = 0; i < leftAxes.Length; i++)
            {
                summingDimensions[i] = left.dimensions[leftAxes[i]];
            }

            var summingStrides = ArrayUtilities.GetStrides(summingDimensions);
            int summingLength = (int)ArrayUtilities.GetProduct(summingDimensions);

            var resultStrides = result.strides;

            // translates from result index to left non-summing dimensions' index portion
            // since left non-summing dimensions are given precedence in result, the end is zero-padded
            int[] leftNonSummingStrides = new int[result.Rank];

            // translates from summing index to left summing dimensions' index portion
            int[] leftSummingStrides = new int[leftAxes.Length];
            ArrayUtilities.SplitStrides(left.strides, leftAxes, leftNonSummingStrides, 0, leftSummingStrides, 0);

            // translates from result index to right non-summing dimensions' index portion
            int[] rightNonSummingStrides = new int[result.Rank];
            //  right non-summing dimensions appear after left non-summing dimensions.
            int rightNonSummingStridesOffset = (left.Rank - leftAxes.Length);

            // translates from summing index to right summing dimensions' index portion
            int[] rightSummingStrides = new int[rightAxes.Length];
            ArrayUtilities.SplitStrides(right.strides, rightAxes, rightNonSummingStrides, rightNonSummingStridesOffset, rightSummingStrides, 0);

            for (int resultIndex = 0; resultIndex < result.Length; resultIndex++)
            {
                ushort sum = (ushort)0;
                
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, resultIndex, resultIndices);

                int leftIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, leftNonSummingStrides);
                int rightIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, rightNonSummingStrides);

                for (int summingIndex = 0; summingIndex < summingLength; summingIndex++)
                {
                    int leftIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, leftSummingStrides);
                    int rightIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, rightSummingStrides);

                    int leftIndex = leftIndexNonSumming + leftIndexSumming;
                    int rightIndex = rightIndexNonSumming + rightIndexSumming;

                    // todo, make this more efficient
                    ArrayUtilities.GetIndices(left.strides, left.IsReversedStride, leftIndex, leftIndices);
                    ArrayUtilities.GetIndices(right.strides, right.IsReversedStride, rightIndex, rightIndices);

                    sum += (ushort)(left[leftIndices] * right[rightIndices]);
                }
                
                result[resultIndices] = sum;
            }
        }
        public void Decrement(Tensor<ushort> tensor, Tensor<ushort> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices]--;
            }
            
        }
        public void Divide(Tensor<ushort> left, Tensor<ushort> right, Tensor<ushort> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ushort)(left[indices] / right[indices]);
            }
            
        }
        public void Divide(Tensor<ushort> tensor, ushort scalar, Tensor<ushort> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ushort)(tensor[indices] / scalar);
            }
            
        }
        public void Equals(Tensor<ushort> left, Tensor<ushort> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] == right[indices];
            }
            
        }
        public void GreaterThan(Tensor<ushort> left, Tensor<ushort> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] > right[indices];
            }
            
        }
        public void GreaterThanOrEqual(Tensor<ushort> left, Tensor<ushort> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] >= right[indices];
            }
            
        }
        public void Increment(Tensor<ushort> tensor, Tensor<ushort> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices]++;
            }
            
        }
        public void LeftShift(Tensor<ushort> tensor, int value, Tensor<ushort> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ushort)(tensor[indices] << value);
            }
            
        }
        public void LessThan(Tensor<ushort> left, Tensor<ushort> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] < right[indices];
            }
            
        }
        public void LessThanOrEqual(Tensor<ushort> left, Tensor<ushort> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] <= right[indices];
            }
            
        }
        public void Modulo(Tensor<ushort> left, Tensor<ushort> right, Tensor<ushort> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ushort)(left[indices] % right[indices]);
            }
            
        }
        public void Modulo(Tensor<ushort> tensor, ushort scalar, Tensor<ushort> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ushort)(tensor[indices] % scalar);
            }
            
        }
        public void Multiply(Tensor<ushort> left, Tensor<ushort> right, Tensor<ushort> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ushort)(left[indices] * right[indices]);
            }
            
        }
        public void Multiply(Tensor<ushort> tensor, ushort scalar, Tensor<ushort> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ushort)(tensor[indices] * scalar);
            }
            
        }
        public void NotEquals(Tensor<ushort> left, Tensor<ushort> right, Tensor<bool> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = left[indices] != right[indices];
            }
            
        }
        public void Or(Tensor<ushort> left, Tensor<ushort> right, Tensor<ushort> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ushort)(left[indices] | right[indices]);
            }
            
        }
        public void Or(Tensor<ushort> tensor, ushort scalar, Tensor<ushort> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ushort)(tensor[indices] | scalar);
            }
            
        }
        public void RightShift(Tensor<ushort> tensor, int value, Tensor<ushort> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ushort)(tensor[indices] >> value);
            }
            
        }
        public void Subtract(Tensor<ushort> left, Tensor<ushort> right, Tensor<ushort> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ushort)(left[indices] - right[indices]);
            }
            
        }
        public void Subtract(Tensor<ushort> tensor, ushort scalar, Tensor<ushort> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ushort)(tensor[indices] - scalar);
            }
            
        }
        public void UnaryMinus(Tensor<ushort> tensor, Tensor<ushort> result)
        {
            throw new NotSupportedException();
        }
        public void UnaryPlus(Tensor<ushort> tensor, Tensor<ushort> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ushort)+tensor[indices];
            }
            
        }
        public void Xor(Tensor<ushort> left, Tensor<ushort> right, Tensor<ushort> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ushort)(left[indices] ^ right[indices]);
            }
            
        }
        public void Xor(Tensor<ushort> tensor, ushort scalar, Tensor<ushort> result)
        {

            Span<int> indices = new Span<int>(new int[result.Rank]);
            for(int i = 0; i < result.Length; i++)
            {
                ArrayUtilities.GetIndices(result.strides, result.IsReversedStride, i, indices);
                result[indices] = (ushort)(tensor[indices] ^ scalar);
            }
            
        }

        public void Add(DenseTensor<ushort> left, DenseTensor<ushort> right, DenseTensor<ushort> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ushort)(leftSpan[i] + rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ushort)(leftSpan[op1Index] + rightSpan[op2Index]);

                }
            }
        }
        public void Add(DenseTensor<ushort> tensor, ushort scalar, DenseTensor<ushort> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ushort)(tensorSpan[i] + scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ushort)(tensorSpan[op1Index] + scalar);

                }
            }
        }
        public void And(DenseTensor<ushort> left, DenseTensor<ushort> right, DenseTensor<ushort> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ushort)(leftSpan[i] & rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ushort)(leftSpan[op1Index] & rightSpan[op2Index]);

                }
            }
        }
        public void And(DenseTensor<ushort> tensor, ushort scalar, DenseTensor<ushort> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ushort)(tensorSpan[i] & scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ushort)(tensorSpan[op1Index] & scalar);

                }
            }
        }
        public void Contract(DenseTensor<ushort> left, DenseTensor<ushort> right, int[] leftAxes, int[] rightAxes, DenseTensor<ushort> result)
        {
            var summingDimensions = new int[leftAxes.Length];
            for(int i = 0; i < leftAxes.Length; i++)
            {
                summingDimensions[i] = left.dimensions[leftAxes[i]];
            }

            var summingStrides = ArrayUtilities.GetStrides(summingDimensions);
            int summingLength = (int)ArrayUtilities.GetProduct(summingDimensions);

            var resultStrides = result.strides;

            // translates from result index to left non-summing dimensions' index portion
            // since left non-summing dimensions are given precedence in result, the end is zero-padded
            int[] leftNonSummingStrides = new int[result.Rank];

            // translates from summing index to left summing dimensions' index portion
            int[] leftSummingStrides = new int[leftAxes.Length];
            ArrayUtilities.SplitStrides(left.strides, leftAxes, leftNonSummingStrides, 0, leftSummingStrides, 0);

            // translates from result index to right non-summing dimensions' index portion
            int[] rightNonSummingStrides = new int[result.Rank];
            //  right non-summing dimensions appear after left non-summing dimensions.
            int rightNonSummingStridesOffset = (left.Rank - leftAxes.Length);

            // translates from summing index to right summing dimensions' index portion
            int[] rightSummingStrides = new int[rightAxes.Length];
            ArrayUtilities.SplitStrides(right.strides, rightAxes, rightNonSummingStrides, rightNonSummingStridesOffset, rightSummingStrides, 0);
            
            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;

            for (int resultIndex = 0; resultIndex < resultSpan.Length; resultIndex++)
            {
                ushort sum = (ushort)0;

                int leftIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, leftNonSummingStrides);
                int rightIndexNonSumming = ArrayUtilities.TransformIndexByStrides(resultIndex, resultStrides, result.IsReversedStride, rightNonSummingStrides);

                for (int summingIndex = 0; summingIndex < summingLength; summingIndex++)
                {
                    int leftIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, leftSummingStrides);
                    int rightIndexSumming = ArrayUtilities.TransformIndexByStrides(summingIndex, summingStrides, false, rightSummingStrides);

                    int leftIndex = leftIndexNonSumming + leftIndexSumming;
                    int rightIndex = rightIndexNonSumming + rightIndexSumming;

                    sum += (ushort)(leftSpan[leftIndex] * rightSpan[rightIndex]);
                }

                resultSpan[resultIndex] = sum;
            }
        }
        public void Decrement(DenseTensor<ushort> tensor, DenseTensor<ushort> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            for(int i = 0; i < resultSpan.Length; i++)
            {
                resultSpan[i]--;
            }
        }
        public void Divide(DenseTensor<ushort> left, DenseTensor<ushort> right, DenseTensor<ushort> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ushort)(leftSpan[i] / rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ushort)(leftSpan[op1Index] / rightSpan[op2Index]);

                }
            }
        }
        public void Divide(DenseTensor<ushort> tensor, ushort scalar, DenseTensor<ushort> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ushort)(tensorSpan[i] / scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ushort)(tensorSpan[op1Index] / scalar);

                }
            }
        }
        public void Equals(DenseTensor<ushort> left, DenseTensor<ushort> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] == rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] == rightSpan[op2Index];

                }
            }
        }
        public void GreaterThan(DenseTensor<ushort> left, DenseTensor<ushort> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] > rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] > rightSpan[op2Index];

                }
            }
        }
        public void GreaterThanOrEqual(DenseTensor<ushort> left, DenseTensor<ushort> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] >= rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] >= rightSpan[op2Index];

                }
            }
        }
        public void Increment(DenseTensor<ushort> tensor, DenseTensor<ushort> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            for(int i = 0; i < resultSpan.Length; i++)
            {
                resultSpan[i]++;
            }
        }
        public void LeftShift(DenseTensor<ushort> tensor, int value, DenseTensor<ushort> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ushort)(tensorSpan[i] << value);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ushort)(tensorSpan[op1Index] << value);

                }
            }
        }
        public void LessThan(DenseTensor<ushort> left, DenseTensor<ushort> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] < rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] < rightSpan[op2Index];

                }
            }
        }
        public void LessThanOrEqual(DenseTensor<ushort> left, DenseTensor<ushort> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] <= rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] <= rightSpan[op2Index];

                }
            }
        }
        public void Modulo(DenseTensor<ushort> left, DenseTensor<ushort> right, DenseTensor<ushort> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ushort)(leftSpan[i] % rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ushort)(leftSpan[op1Index] % rightSpan[op2Index]);

                }
            }
        }
        public void Modulo(DenseTensor<ushort> tensor, ushort scalar, DenseTensor<ushort> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ushort)(tensorSpan[i] % scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ushort)(tensorSpan[op1Index] % scalar);

                }
            }
        }
        public void Multiply(DenseTensor<ushort> left, DenseTensor<ushort> right, DenseTensor<ushort> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ushort)(leftSpan[i] * rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ushort)(leftSpan[op1Index] * rightSpan[op2Index]);

                }
            }
        }
        public void Multiply(DenseTensor<ushort> tensor, ushort scalar, DenseTensor<ushort> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ushort)(tensorSpan[i] * scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ushort)(tensorSpan[op1Index] * scalar);

                }
            }
        }
        public void NotEquals(DenseTensor<ushort> left, DenseTensor<ushort> right, DenseTensor<bool> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = leftSpan[i] != rightSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = leftSpan[op1Index] != rightSpan[op2Index];

                }
            }
        }
        public void Or(DenseTensor<ushort> left, DenseTensor<ushort> right, DenseTensor<ushort> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ushort)(leftSpan[i] | rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ushort)(leftSpan[op1Index] | rightSpan[op2Index]);

                }
            }
        }
        public void Or(DenseTensor<ushort> tensor, ushort scalar, DenseTensor<ushort> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ushort)(tensorSpan[i] | scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ushort)(tensorSpan[op1Index] | scalar);

                }
            }
        }
        public void RightShift(DenseTensor<ushort> tensor, int value, DenseTensor<ushort> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ushort)(tensorSpan[i] >> value);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ushort)(tensorSpan[op1Index] >> value);

                }
            }
        }
        public void Subtract(DenseTensor<ushort> left, DenseTensor<ushort> right, DenseTensor<ushort> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ushort)(leftSpan[i] - rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ushort)(leftSpan[op1Index] - rightSpan[op2Index]);

                }
            }
        }
        public void Subtract(DenseTensor<ushort> tensor, ushort scalar, DenseTensor<ushort> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ushort)(tensorSpan[i] - scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ushort)(tensorSpan[op1Index] - scalar);

                }
            }
        }
        public void UnaryMinus(DenseTensor<ushort> tensor, DenseTensor<ushort> result)
        {
            throw new NotSupportedException();
        }
        public void UnaryPlus(DenseTensor<ushort> tensor, DenseTensor<ushort> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ushort)+tensorSpan[i];
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ushort)+tensorSpan[op1Index];

                }
            }
        }
        public void Xor(DenseTensor<ushort> left, DenseTensor<ushort> right, DenseTensor<ushort> result)
        {

            var resultSpan = result.Buffer.Span;
            var leftSpan = left.Buffer.Span;
            var rightSpan = right.Buffer.Span;
            if  ((result.IsReversedStride == left.IsReversedStride) && (result.IsReversedStride == right.IsReversedStride))
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ushort)(leftSpan[i] ^ rightSpan[i]);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref left.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                ref int op2Index = ref right.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;

                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      !left.IsReversedStride ? left.strides : 
                                      right.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         left.IsReversedStride ? left.strides : 
                                         right.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ushort)(leftSpan[op1Index] ^ rightSpan[op2Index]);

                }
            }
        }
        public void Xor(DenseTensor<ushort> tensor, ushort scalar, DenseTensor<ushort> result)
        {

            var resultSpan = result.Buffer.Span;
            var tensorSpan = tensor.Buffer.Span;
            if  (result.IsReversedStride == tensor.IsReversedStride)
            {
                for(int i = 0; i < resultSpan.Length; i++)
                {
                    resultSpan[i] = (ushort)(tensorSpan[i] ^ scalar);
                }
            }
            else
            {
                int rowMajorIndex = 0;
                int colMajorIndex = 0;
                
                ref int resultIndex = ref result.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                ref int op1Index = ref tensor.IsReversedStride ? ref colMajorIndex : ref rowMajorIndex;
                
                var rowMajorStrides = !result.IsReversedStride ? result.strides :
                                      tensor.strides;
                var columnMajorStrides = result.IsReversedStride ? result.strides :
                                         tensor.strides;
                for(;rowMajorIndex < resultSpan.Length; rowMajorIndex++)
                {
                    colMajorIndex = ArrayUtilities.TransformIndexByStrides(rowMajorIndex, rowMajorStrides, false, columnMajorStrides);
                    
                    resultSpan[resultIndex] = (ushort)(tensorSpan[op1Index] ^ scalar);

                }
            }
        }
    }
}
