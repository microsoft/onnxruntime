// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is copied and adapted from the following git repository -
// https://github.com/dotnet/corefx
// Commit ID: bdd0814360d4c3a58860919f292a306242f27da1
// Path: /src/System.Numerics.Tensors/src/System/Numerics/Tensors/Tensor.cs
// Original license statement below -

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

// Making this assembly's internals visible to the internal Test assembly
[assembly: InternalsVisibleTo("Microsoft.ML.OnnxRuntime.Tests," +
                              "PublicKey=002400000480000094000000060200000024000052534131000400000100010059013e94e4bc70" +
                              "136ca4c35f33acd6b62974536b698f9c7a21cee18d805c7ad860ad9eebfdc47a96ba2f8d03f4cf" +
                              "1c36b9d30787e276c7b9833b5bf2a6eba7e919e6b90083078a352262aed1d842e5f70a3085cbcf" +
                              "4c56ae851b161137920961c23fcc246598d61d258ccc615c927b2441359eea666a99ce1c3c07dc" +
                              "a18fb0e1")]


namespace Microsoft.ML.OnnxRuntime.Tensors
{
    /// <summary>
    /// Supported Tensor DataType
    /// </summary>
    public enum TensorElementType
    {
        Float = 1,
        UInt8 = 2,
        Int8 = 3,
        UInt16 = 4,
        Int16 = 5,
        Int32 = 6,
        Int64 = 7,
        String = 8,
        Bool = 9,
        Float16 = 10,
        Double = 11,
        UInt32 = 12,
        UInt64 = 13,
        Complex64 = 14,
        Complex128 = 15,
        BFloat16 = 16,
        DataTypeMax = 17
    }

    /// <summary>
    /// This value type represents A Float16 value
    /// it is blittable as defined in https://docs.microsoft.com/en-us/dotnet/framework/interop/blittable-and-non-blittable-types
    /// and as such, represented the same way in managed and native memories. This means that arrays of this type
    /// do not have to be copied to be passed to native memory but simply pinnned and read by native code. Thus,
    /// one can create a Tensor on top of an array of these structures and feed it directly to Onnxruntime library.
    /// Binary wise, it is the same as ushort[] (uint16_t in C++). However, we would like a separate type for type dispatching.
    /// </summary>
    public struct Float16
    {
        /// <summary>
        /// float16 representation bits
        /// </summary>
        public ushort value;
        /// <summary>
        /// Ctor
        /// </summary>
        /// <param name="v"></param>
        public Float16(ushort v)
        {
            value = v;
        }
        /// <summary>
        /// Converts to ushort
        /// </summary>
        /// <param name="f">instance of Float16</param>
        /// <returns>value member</returns>
        public static implicit operator ushort (Float16 f) { return f.value; }
        /// <summary>
        /// Converts a 16-bit unsigned integer to a Float16.
        /// </summary>
        /// <param name="value">A 16-bit unsigned integer.</param>
        /// <returns>A Float16 that represents the converted 16-bit unsigned integer.</returns>
        public static implicit operator Float16(ushort value) { return new Float16(value); }
        /// <summary>
        /// Compares values of two Float16 for binary equality
        /// </summary>
        /// <param name="lhs"></param>
        /// <param name="rhs"></param>
        /// <returns>result of value comparisons</returns>
        public static bool operator ==(Float16 lhs, Float16 rhs) { return lhs.value == rhs.value; }
        /// <summary>
        /// Compares values of two Float16 for binary inequality
        /// </summary>
        /// <param name="lhs"></param>
        /// <param name="rhs"></param>
        /// <returns>result of value comparisons</returns>
        public static bool operator !=(Float16 lhs, Float16 rhs) { return lhs.value != rhs.value; }
        /// <summary>
        /// Returns a value indicating whether this instance and other Float16 represent the same value.
        /// </summary>
        /// <param name="other">A Float16 object to compare to this instance.</param>
        /// <returns>true if other.value is equal to this instance; otherwise, false.</returns>
        public bool Equals(Float16 other)
        {
            return (other == this);
        }
        /// <summary>
        /// Returns a value indicating whether this instance and a specified System.Object
        /// represent the same type and value.
        /// </summary>
        /// <param name="obj">An System.Object.</param>
        /// <returns>true if obj is Float16 and its value is equal to this instance; otherwise, false.</returns>
        public override bool Equals(object obj)
        {
            bool result = false;
            if (obj is Float16)
            {
                Float16 fl16 = (Float16)obj;
                result = (fl16 == this);
            }
            return result;
        }
        /// <summary>
        /// Returns the hash code for this instance.
        /// </summary>
        /// <returns>A 32-bit signed integer hash code.</returns>
        public override int GetHashCode()
        {
            return value.GetHashCode();
        }
    }

    /// <summary>
    /// This value type represents A BFloat16 value
    /// it is blittable as defined in https://docs.microsoft.com/en-us/dotnet/framework/interop/blittable-and-non-blittable-types
    /// and as such, represented the same way in managed and native memories. This means that arrays of this type
    /// do not have to be copied to be passed to native memory but simply pinnned and read by native code. Thus,
    /// one can create a Tensor on top of an array of these structures and feed it directly to Onnxruntime library.
    /// Binary wise, it is the same as ushort[] (uint16_t in C++). However, we would like a separate type for type dispatching.
    /// </summary>
    public struct BFloat16
    {
        /// <summary>
        /// bfloat16 representation bits
        /// </summary>
        public ushort value;
        /// <summary>
        /// Ctor
        /// </summary>
        /// <param name="v"></param>
        public BFloat16(ushort v)
        {
            value = v;
        }
        /// <summary>
        /// Converts to ushort
        /// </summary>
        /// <param name="bf">instance of BFloat16</param>
        /// <returns>value member</returns>
        public static implicit operator ushort(BFloat16 bf) { return bf.value; }
        /// <summary>
        /// Converts a 16-bit unsigned integer to a BFloat16.
        /// </summary>
        /// <param name="value">A 16-bit unsigned integer.</param>
        /// <returns>A BFloat16 that represents the converted 16-bit unsigned integer.</returns>
        public static implicit operator BFloat16(ushort value) { return new BFloat16(value); }
        /// <summary>
        /// Compares values of two BFloat16 for binary equality
        /// </summary>
        /// <param name="lhs"></param>
        /// <param name="rhs"></param>
        /// <returns>result of value comparisons</returns>
        public static bool operator ==(BFloat16 lhs, BFloat16 rhs) { return lhs.value == rhs.value; }
        /// <summary>
        /// Compares values of two BFloat16 for binary inequality
        /// </summary>
        /// <param name="lhs"></param>
        /// <param name="rhs"></param>
        /// <returns>result of value comparisons</returns>
        public static bool operator !=(BFloat16 lhs, BFloat16 rhs) { return lhs.value != rhs.value; }

        /// <summary>
        /// Returns a value indicating whether this instance and other BFloat16 represent the same value.
        /// </summary>
        /// <param name="other">A BFloat16 object to compare to this instance.</param>
        /// <returns>true if other.value is equal to this instance; otherwise, false.</returns>
        public bool Equals(BFloat16 other)
        {
            return (other == this);
        }

        /// <summary>
        /// Returns a value indicating whether this instance and a specified System.Object
        /// represent the same type and value.
        /// </summary>
        /// <param name="obj">An System.Object.</param>
       /// <returns>true if obj is BFloat16 its value is equal to this instance; otherwise, false.</returns>
        public override bool Equals(object obj)
        {
            bool result = false;
            if (obj is BFloat16)
            {
                BFloat16 bfl16 = (BFloat16)obj;
                result = (bfl16 == this);
            }
            return result;
        }
        /// <summary>
        /// Returns the hash code for this instance.
        /// </summary>
        /// <returns>A 32-bit signed integer hash code.</returns>
        public override int GetHashCode()
        {
            return value.GetHashCode();
        }
    }

    /// <summary>
    /// Helps typecasting. Holds Tensor element type traits.
    /// </summary>
    public class TensorTypeInfo
    {
        /// <summary>
        /// TensorElementType enum
        /// </summary>
        /// <value>type enum value</value>
        public TensorElementType ElementType { get; private set; }
        /// <summary>
        /// Size of the stored primitive type in bytes
        /// </summary>
        /// <value>size in bytes</value>
        public int TypeSize { get; private set; }
        /// <summary>
        /// Is the type is a string
        /// </summary>
        /// <value>true if Tensor element type is a string</value>
        public bool IsString { get { return ElementType == TensorElementType.String; } }
        /// <summary>
        /// Ctor
        /// </summary>
        /// <param name="elementType">TensorElementType value</param>
        /// <param name="typeSize">size fo the type in bytes</param>
        public TensorTypeInfo(TensorElementType elementType, int typeSize)
        {
            ElementType = elementType;
            TypeSize = typeSize;
        }
    }

    /// <summary>
    /// Holds TensorElement traits
    /// </summary>
    public class TensorElementTypeInfo
    {
        /// <summary>
        /// Tensor element type
        /// </summary>
        /// <value>System.Type</value>
        public Type TensorType { get; private set; }
        /// <summary>
        /// Size of the stored primitive type in bytes
        /// </summary>
        /// <value>size in bytes</value>
        public int TypeSize { get; private set; }
        /// <summary>
        /// Is the type is a string
        /// </summary>
        /// <value>true if Tensor element type is a string</value>
        public bool IsString { get; private set; }
        /// <summary>
        /// Ctor
        /// </summary>
        /// <param name="type">Tensor element type</param>
        /// <param name="typeSize">typesize</param>
        public TensorElementTypeInfo(Type type, int typeSize)
        {
            TensorType = type;
            TypeSize = typeSize;
            IsString = type == typeof(string);
        }
    }

    /// <summary>
    /// This class is a base for all Tensors. It hosts maps with type traits.
    /// </summary>
    public class TensorBase
    {
        private static readonly Dictionary<Type, TensorTypeInfo> typeInfoMap;

        private static readonly Dictionary<TensorElementType, TensorElementTypeInfo> tensorElementTypeInfoMap;

        static TensorBase () {
            typeInfoMap = new Dictionary<Type, TensorTypeInfo>()
            {
                { typeof(float), new TensorTypeInfo( TensorElementType.Float, sizeof(float)) },
                { typeof(byte), new TensorTypeInfo( TensorElementType.UInt8, sizeof(byte)) },
                { typeof(sbyte), new TensorTypeInfo( TensorElementType.Int8, sizeof(sbyte)) },
                { typeof(ushort), new TensorTypeInfo( TensorElementType.UInt16, sizeof(ushort)) },
                { typeof(short), new TensorTypeInfo( TensorElementType.Int16, sizeof(short)) },
                { typeof(int), new TensorTypeInfo( TensorElementType.Int32, sizeof(int)) },
                { typeof(long), new TensorTypeInfo( TensorElementType.Int64, sizeof(long)) },
                { typeof(string), new TensorTypeInfo( TensorElementType.String, -1) },
                { typeof(bool), new TensorTypeInfo( TensorElementType.Bool, sizeof(bool)) },
                { typeof(Float16), new TensorTypeInfo( TensorElementType.Float16, sizeof(ushort)) },
                { typeof(double), new TensorTypeInfo( TensorElementType.Double, sizeof(double)) },
                { typeof(uint), new TensorTypeInfo( TensorElementType.UInt32, sizeof(uint)) },
                { typeof(ulong), new TensorTypeInfo( TensorElementType.UInt64, sizeof(ulong)) },
                { typeof(BFloat16), new TensorTypeInfo( TensorElementType.BFloat16, sizeof(ushort)) }
            };

            tensorElementTypeInfoMap = new Dictionary<TensorElementType, TensorElementTypeInfo>();
            foreach(var info in typeInfoMap)
            {
                tensorElementTypeInfoMap.Add(info.Value.ElementType, new TensorElementTypeInfo(info.Key, info.Value.TypeSize));
            }
         }

        private readonly Type _primitiveType;
        /// <summary>
        /// Constructs TensorBae
        /// </summary>
        /// <param name="primitiveType">primitive type the deriving class is using</param>
        protected TensorBase(Type primitiveType)
        {
            // Should hold as we rely on this to pass arrays of these
            // types to native code
            unsafe
            {
                Debug.Assert(sizeof(ushort) == sizeof(Float16));
                Debug.Assert(sizeof(ushort) == sizeof(BFloat16));
            }
            _primitiveType = primitiveType;
        }

        /// <summary>
        /// Query TensorTypeInfo by one of the supported types
        /// </summary>
        /// <param name="type"></param>
        /// <returns>TensorTypeInfo or null if not supported</returns>
        public static TensorTypeInfo GetTypeInfo(Type type)
        {
            TensorTypeInfo result = null;
            typeInfoMap.TryGetValue(type, out result);
            return result;
        }

        /// <summary>
        /// Query TensorElementTypeInfo by enum
        /// </summary>
        /// <param name="elementType">type enum</param>
        /// <returns>instance of TensorElementTypeInfo or null if not found</returns>
        public static TensorElementTypeInfo GetElementTypeInfo(TensorElementType elementType)
        {
            TensorElementTypeInfo result = null;
            tensorElementTypeInfoMap.TryGetValue(elementType, out result);
            return result;
        }

        /// <summary>
        /// Query TensorTypeInfo using this Tensor type
        /// </summary>
        /// <returns></returns>
        public TensorTypeInfo GetTypeInfo()
        {
            return GetTypeInfo(_primitiveType);
        }
    }

    /// <summary>
    /// Various methods for creating and manipulating Tensor&lt;T&gt;
    /// </summary>
    public static partial class Tensor
    {
        /// <summary>
        /// Creates an identity tensor of the specified size.  An identity tensor is a two dimensional tensor with 1s in the diagonal.
        /// </summary>
        /// <typeparam name="T">type contained within the Tensor.  Typically a value type such as int, double, float, etc.</typeparam>
        /// <param name="size">Width and height of the identity tensor to create.</param>
        /// <returns>a <paramref name="size"/> by <paramref name="size"/> with 1s along the diagonal and zeros elsewhere.</returns>
        public static Tensor<T> CreateIdentity<T>(int size)
        {
            return CreateIdentity(size, false, Tensor<T>.One);
        }

        /// <summary>
        /// Creates an identity tensor of the specified size and layout (row vs column major).  An identity tensor is a two dimensional tensor with 1s in the diagonal.
        /// </summary>
        /// <typeparam name="T">type contained within the Tensor.  Typically a value type such as int, double, float, etc.</typeparam>
        /// <param name="size">Width and height of the identity tensor to create.</param>
        /// <param name="columMajor">>False to indicate that the first dimension is most minor (closest) and the last dimension is most major (farthest): row-major.  True to indicate that the last dimension is most minor (closest together) and the first dimension is most major (farthest apart): column-major.</param>
        /// <returns>a <paramref name="size"/> by <paramref name="size"/> with 1s along the diagonal and zeros elsewhere.</returns>
        public static Tensor<T> CreateIdentity<T>(int size, bool columMajor)
        {
            return CreateIdentity(size, columMajor, Tensor<T>.One);
        }

        /// <summary>
        /// Creates an identity tensor of the specified size and layout (row vs column major) using the specified one value.  An identity tensor is a two dimensional tensor with 1s in the diagonal.  This may be used in case T is a type that doesn't have a known 1 value.
        /// </summary>
        /// <typeparam name="T">type contained within the Tensor.  Typically a value type such as int, double, float, etc.</typeparam>
        /// <param name="size">Width and height of the identity tensor to create.</param>
        /// <param name="columMajor">>False to indicate that the first dimension is most minor (closest) and the last dimension is most major (farthest): row-major.  True to indicate that the last dimension is most minor (closest together) and the first dimension is most major (farthest apart): column-major.</param>
        /// <param name="oneValue">Value of <typeparamref name="T"/> that is used along the diagonal.</param>
        /// <returns>a <paramref name="size"/> by <paramref name="size"/> with 1s along the diagonal and zeros elsewhere.</returns>
        public static Tensor<T> CreateIdentity<T>(int size, bool columMajor, T oneValue)
        {
            unsafe
            {
                Span<int> dimensions = stackalloc int[2];
                dimensions[0] = dimensions[1] = size;

                var result = new DenseTensor<T>(dimensions, columMajor);

                for (int i = 0; i < size; i++)
                {
                    result.SetValue(i * size + i, oneValue);
                }

                return result;
            }
        }

        /// <summary>
        /// Creates a n+1-rank tensor using the specified n-rank diagonal.  Values not on the diagonal will be filled with zeros.
        /// </summary>
        /// <typeparam name="T">type contained within the Tensor.  Typically a value type such as int, double, float, etc.</typeparam>
        /// <param name="diagonal">Tensor representing the diagonal to build the new tensor from.</param>
        /// <returns>A new tensor of the same layout and order as <paramref name="diagonal"/> of one higher rank, with the values of <paramref name="diagonal"/> along the diagonal and zeros elsewhere.</returns>
        public static Tensor<T> CreateFromDiagonal<T>(Tensor<T> diagonal)
        {
            return CreateFromDiagonal(diagonal, 0);
        }

        /// <summary>
        /// Creates a n+1-dimension tensor using the specified n-dimension diagonal at the specified offset from the center.  Values not on the diagonal will be filled with zeros.
        /// </summary>
        /// <typeparam name="T">type contained within the Tensor.  Typically a value type such as int, double, float, etc.</typeparam>
        /// <param name="diagonal">Tensor representing the diagonal to build the new tensor from.</param>
        /// <param name="offset">Offset of diagonal to set in returned tensor.  0 for the main diagonal, less than zero for diagonals below, greater than zero from diagonals above.</param>
        /// <returns>A new tensor of the same layout and order as <paramref name="diagonal"/> of one higher rank, with the values of <paramref name="diagonal"/> along the specified diagonal and zeros elsewhere.</returns>
        public static Tensor<T> CreateFromDiagonal<T>(Tensor<T> diagonal, int offset)
        {
            if (diagonal.Rank < 1)
            {
                throw new ArgumentException($"Tensor {nameof(diagonal)} must have at least one dimension.", nameof(diagonal));
            }

            int diagonalLength = diagonal.dimensions[0];

            // TODO: allow specification of axis1 and axis2?
            var rank = diagonal.dimensions.Length + 1;
            Span<int> dimensions = rank < ArrayUtilities.StackallocMax ? stackalloc int[rank] : new int[rank];

            // assume square
            var axisLength = diagonalLength + Math.Abs(offset);
            dimensions[0] = dimensions[1] = axisLength;

            for (int i = 1; i < diagonal.dimensions.Length; i++)
            {
                dimensions[i + 1] = diagonal.dimensions[i];
            }

            var result = diagonal.CloneEmpty(dimensions);

            var sizePerDiagonal = diagonal.Length / diagonalLength;

            var diagProjectionStride = diagonal.IsReversedStride && diagonal.Rank > 1 ? diagonal.strides[1] : 1;
            var resultProjectionStride = result.IsReversedStride && result.Rank > 2 ? result.strides[2] : 1;

            for (int diagIndex = 0; diagIndex < diagonalLength; diagIndex++)
            {
                var resultIndex0 = offset < 0 ? diagIndex - offset : diagIndex;
                var resultIndex1 = offset > 0 ? diagIndex + offset : diagIndex;

                var resultBase = resultIndex0 * result.strides[0] + resultIndex1 * result.strides[1];
                var diagBase = diagIndex * diagonal.strides[0];

                for (int diagProjectionOffset = 0; diagProjectionOffset < sizePerDiagonal; diagProjectionOffset++)
                {
                    result.SetValue(resultBase + diagProjectionOffset * resultProjectionStride,
                        diagonal.GetValue(diagBase + diagProjectionOffset * diagProjectionStride));
                }
            }

            return result;
        }
    }

    /// <summary>
    /// Represents a multi-dimensional collection of objects of type T that can be accessed by indices.
    /// </summary>
    /// <typeparam name="T">type contained within the Tensor.  Typically a value type such as int, double, float, etc.</typeparam>
    [DebuggerDisplay("{GetArrayString(false)}")]
    // When we cross-compile for frameworks that expose ICloneable this must implement ICloneable as well.
    public abstract class Tensor<T> : TensorBase, IList, IList<T>, IReadOnlyList<T>, IStructuralComparable, IStructuralEquatable
    {
        internal static T Zero
        {
            get
            {
                if (typeof(T) == typeof(bool))
                {
                    return (T)(object)(false);
                }
                else if (typeof(T) == typeof(byte))
                {
                    return (T)(object)(byte)(0);
                }
                else if (typeof(T) == typeof(char))
                {
                    return (T)(object)(char)(0);
                }
                else if (typeof(T) == typeof(decimal))
                {
                    return (T)(object)(decimal)(0);
                }
                else if (typeof(T) == typeof(double))
                {
                    return (T)(object)(double)(0);
                }
                else if (typeof(T) == typeof(float))
                {
                    return (T)(object)(float)(0);
                }
                else if (typeof(T) == typeof(int))
                {
                    return (T)(object)(int)(0);
                }
                else if (typeof(T) == typeof(long))
                {
                    return (T)(object)(long)(0);
                }
                else if (typeof(T) == typeof(sbyte))
                {
                    return (T)(object)(sbyte)(0);
                }
                else if (typeof(T) == typeof(short))
                {
                    return (T)(object)(short)(0);
                }
                else if (typeof(T) == typeof(uint))
                {
                    return (T)(object)(uint)(0);
                }
                else if (typeof(T) == typeof(ulong))
                {
                    return (T)(object)(ulong)(0);
                }
                else if (typeof(T) == typeof(ushort))
                {
                    return (T)(object)(ushort)(0);
                }
                else if (typeof(T) == typeof(Float16))
                {
                    return (T)(object)(ushort)(0);
                }
                else if (typeof(T) == typeof(BFloat16))
                {
                    return (T)(object)(ushort)(0);
                }

                throw new NotSupportedException();
            }
        }

        internal static T One
        {
            get
            {
                if (typeof(T) == typeof(bool))
                {
                    return (T)(object)(true);
                }
                else if (typeof(T) == typeof(byte))
                {
                    return (T)(object)(byte)(1);
                }
                else if (typeof(T) == typeof(char))
                {
                    return (T)(object)(char)(1);
                }
                else if (typeof(T) == typeof(decimal))
                {
                    return (T)(object)(decimal)(1);
                }
                else if (typeof(T) == typeof(double))
                {
                    return (T)(object)(double)(1);
                }
                else if (typeof(T) == typeof(float))
                {
                    return (T)(object)(float)(1);
                }
                else if (typeof(T) == typeof(int))
                {
                    return (T)(object)(int)(1);
                }
                else if (typeof(T) == typeof(long))
                {
                    return (T)(object)(long)(1);
                }
                else if (typeof(T) == typeof(sbyte))
                {
                    return (T)(object)(sbyte)(1);
                }
                else if (typeof(T) == typeof(short))
                {
                    return (T)(object)(short)(1);
                }
                else if (typeof(T) == typeof(uint))
                {
                    return (T)(object)(uint)(1);
                }
                else if (typeof(T) == typeof(ulong))
                {
                    return (T)(object)(ulong)(1);
                }
                else if (typeof(T) == typeof(ushort))
                {
                    return (T)(object)(ushort)(1);
                } 
                else if(typeof(T) == typeof(Float16))
                {
                    return (T)(object)(ushort)(15360);
                }
                else if (typeof(T) == typeof(BFloat16))
                {
                    return (T)(object)(ushort)(16256);
                }

                throw new NotSupportedException();
            }
        }

        internal readonly int[] dimensions;
        internal readonly int[] strides;
        private readonly bool isReversedStride;

        private readonly long length;

        /// <summary>
        /// Initialize a 1-dimensional tensor of the specified length
        /// </summary>
        /// <param name="length">Size of the 1-dimensional tensor</param>
        protected Tensor(int length) : base(typeof(T))
        {
            dimensions = new[] { length };
            strides = new[] { 1 };
            isReversedStride = false;
            this.length = length;
        }

        /// <summary>
        /// Initialize an n-dimensional tensor with the specified dimensions and layout.  ReverseStride=true gives a stride of 1-element width to the first dimension (0).  ReverseStride=false gives a stride of 1-element width to the last dimension (n-1).
        /// </summary>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the Tensor to create.</param>
        /// <param name="reverseStride">False (default) to indicate that the first dimension is most major (farthest apart) and the last dimension is most minor (closest together): akin to row-major in a rank-2 tensor.  True to indicate that the last dimension is most major (farthest apart) and the first dimension is most minor (closest together): akin to column-major in a rank-2 tensor.</param>
        protected Tensor(ReadOnlySpan<int> dimensions, bool reverseStride) : base(typeof(T))
        {
            if (dimensions == null)
            {
                throw new ArgumentNullException(nameof(dimensions));
            }

            this.dimensions = new int[dimensions.Length];
            long size = 1;
            for (int i = 0; i < dimensions.Length; i++)
            {
                if (dimensions[i] < 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(dimensions), "Dimensions must be non-negative");
                }
                this.dimensions[i] = dimensions[i];
                size *= dimensions[i];
            }

            strides = ArrayUtilities.GetStrides(dimensions, reverseStride);
            isReversedStride = reverseStride;

            length = size;
        }

        /// <summary>
        /// Initializes tensor with same dimensions as array, content of array is ignored.  ReverseStride=true gives a stride of 1-element width to the first dimension (0).  ReverseStride=false gives a stride of 1-element width to the last dimension (n-1).
        /// </summary>
        /// <param name="fromArray">Array from which to derive dimensions.</param>
        /// <param name="reverseStride">False (default) to indicate that the first dimension is most major (farthest apart) and the last dimension is most minor (closest together): akin to row-major in a rank-2 tensor.  True to indicate that the last dimension is most major (farthest apart) and the first dimension is most minor (closest together): akin to column-major in a rank-2 tensor.</param>
        protected Tensor(Array fromArray, bool reverseStride) : base(typeof(T))
        {
            if (fromArray == null)
            {
                throw new ArgumentNullException(nameof(fromArray));
            }

            dimensions = new int[fromArray.Rank];
            long size = 1;
            for (int i = 0; i < dimensions.Length; i++)
            {
                dimensions[i] = fromArray.GetLength(i);
                size *= dimensions[i];
            }

            strides = ArrayUtilities.GetStrides(dimensions, reverseStride);
            isReversedStride = reverseStride;

            length = size;
        }

        /// <summary>
        /// Total length of the Tensor.
        /// </summary>
        public long Length => length;

        /// <summary>
        /// Rank of the tensor: number of dimensions.
        /// </summary>
        public int Rank => dimensions.Length;

        /// <summary>
        /// True if strides are reversed (AKA Column-major)
        /// </summary>
        public bool IsReversedStride => isReversedStride;

        /// <summary>
        /// Returns a readonly view of the dimensions of this tensor.
        /// </summary>
        public ReadOnlySpan<int> Dimensions => dimensions;

        /// <summary>
        /// Returns a readonly view of the strides of this tensor.
        /// </summary>
        public ReadOnlySpan<int> Strides => strides;

        /// <summary>
        /// Sets all elements in Tensor to <paramref name="value"/>.
        /// </summary>
        /// <param name="value">Value to fill</param>
        public virtual void Fill(T value)
        {
            for (int i = 0; i < Length; i++)
            {
                SetValue(i, value);
            }
        }

        /// <summary>
        /// Creates a shallow copy of this tensor, with new backing storage.
        /// </summary>
        /// <returns>A shallow copy of this tensor.</returns>
        public abstract Tensor<T> Clone();

        /// <summary>
        /// Creates a new Tensor with the same layout and dimensions as this tensor with elements initialized to their default value.
        /// </summary>
        /// <returns>A new Tensor with the same layout and dimensions as this tensor with elements initialized to their default value.</returns>
        public virtual Tensor<T> CloneEmpty()
        {
            return CloneEmpty<T>(dimensions);
        }

        /// <summary>
        /// Creates a new Tensor with the specified dimensions and the same layout as this tensor with elements initialized to their default value.
        /// </summary>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
        /// <returns>A new Tensor with the same layout as this tensor and specified <paramref name="dimensions"/> with elements initialized to their default value.</returns>
        public virtual Tensor<T> CloneEmpty(ReadOnlySpan<int> dimensions)
        {
            return CloneEmpty<T>(dimensions);
        }

        /// <summary>
        /// Creates a new Tensor of a different type with the same layout and size as this tensor with elements initialized to their default value.
        /// </summary>
        /// <typeparam name="TResult">Type contained within the new Tensor.  Typically a value type such as int, double, float, etc.</typeparam>
        /// <returns>A new Tensor with the same layout and dimensions as this tensor with elements of <typeparamref name="TResult"/> type initialized to their default value.</returns>
        public virtual Tensor<TResult> CloneEmpty<TResult>()
        {
            return CloneEmpty<TResult>(dimensions);
        }

        /// <summary>
        /// Creates a new Tensor of a different type with the specified dimensions and the same layout as this tensor with elements initialized to their default value.
        /// </summary>
        /// <typeparam name="TResult">Type contained within the new Tensor.  Typically a value type such as int, double, float, etc.</typeparam>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
        /// <returns>A new Tensor with the same layout as this tensor of specified <paramref name="dimensions"/> with elements of <typeparamref name="TResult"/> type initialized to their default value.</returns>
        public abstract Tensor<TResult> CloneEmpty<TResult>(ReadOnlySpan<int> dimensions);

        /// <summary>
        /// Gets the n-1 dimension diagonal from the n dimension tensor.
        /// </summary>
        /// <returns>An n-1 dimension tensor with the values from the main diagonal of this tensor.</returns>
        public Tensor<T> GetDiagonal()
        {
            return GetDiagonal(0);
        }

        /// <summary>
        /// Gets the n-1 dimension diagonal from the n dimension tensor at the specified offset from center.
        /// </summary>
        /// <param name="offset">Offset of diagonal to set in returned tensor.  0 for the main diagonal, less than zero for diagonals below, greater than zero from diagonals above.</param>
        /// <returns>An n-1 dimension tensor with the values from the specified diagonal of this tensor.</returns>
        public Tensor<T> GetDiagonal(int offset)
        {
            // Get diagonal of first two dimensions for all remaining dimensions

            // diagnonal is as follows:
            // { 1, 2, 4 }
            // { 8, 3, 9 }
            // { 0, 7, 5 }
            // The diagonal at offset 0 is { 1, 3, 5 }
            // The diagonal at offset 1 is { 2, 9 }
            // The diagonal at offset -1 is { 8, 7 }

            if (Rank < 2)
            {
                throw new InvalidOperationException($"Cannot compute diagonal of {nameof(Tensor<T>)} with Rank less than 2.");
            }

            // TODO: allow specification of axis1 and axis2?
            var axisLength0 = dimensions[0];
            var axisLength1 = dimensions[1];

            // the diagonal will be the length of the smaller axis
            // if offset it positive, the length will shift along the second axis
            // if the offsett is negative, the length will shift along the first axis
            // In that way the length of the diagonal will be 
            //   Min(offset < 0 ? axisLength0 + offset : axisLength0, offset > 0 ? axisLength1 - offset : axisLength1)
            // To illustrate, consider the following
            // { 1, 2, 4, 3, 7 }
            // { 8, 3, 9, 2, 6 }
            // { 0, 7, 5, 2, 9 }
            // The diagonal at offset 0 is { 1, 3, 5 }, Min(3, 5) = 3
            // The diagonal at offset 1 is { 2, 9, 2 }, Min(3, 5 - 1) = 3
            // The diagonal at offset 3 is { 3, 6 }, Min(3, 5 - 3) = 2
            // The diagonal at offset -1 is { 8, 7 }, Min(3 - 1, 5) = 2
            var offsetAxisLength0 = offset < 0 ? axisLength0 + offset : axisLength0;
            var offsetAxisLength1 = offset > 0 ? axisLength1 - offset : axisLength1;

            var diagonalLength = Math.Min(offsetAxisLength0, offsetAxisLength1);

            if (diagonalLength <= 0)
            {
                throw new ArgumentException($"Cannot compute diagonal with offset {offset}", nameof(offset));
            }

            var newTensorRank = Rank - 1;
            var newTensorDimensions = newTensorRank < ArrayUtilities.StackallocMax ? stackalloc int[newTensorRank] : new int[newTensorRank];
            newTensorDimensions[0] = diagonalLength;

            for (int i = 2; i < dimensions.Length; i++)
            {
                newTensorDimensions[i - 1] = dimensions[i];
            }

            var diagonalTensor = CloneEmpty(newTensorDimensions);
            var sizePerDiagonal = diagonalTensor.Length / diagonalTensor.Dimensions[0];

            var diagProjectionStride = diagonalTensor.IsReversedStride && diagonalTensor.Rank > 1 ? diagonalTensor.strides[1] : 1;
            var sourceProjectionStride = IsReversedStride && Rank > 2 ? strides[2] : 1;

            for (int diagIndex = 0; diagIndex < diagonalLength; diagIndex++)
            {
                var sourceIndex0 = offset < 0 ? diagIndex - offset : diagIndex;
                var sourceIndex1 = offset > 0 ? diagIndex + offset : diagIndex;

                var sourceBase = sourceIndex0 * strides[0] + sourceIndex1 * strides[1];
                var diagBase = diagIndex * diagonalTensor.strides[0];

                for (int diagProjectionIndex = 0; diagProjectionIndex < sizePerDiagonal; diagProjectionIndex++)
                {
                    diagonalTensor.SetValue(diagBase + diagProjectionIndex * diagProjectionStride,
                        GetValue(sourceBase + diagProjectionIndex * sourceProjectionStride));
                }
            }

            return diagonalTensor;
        }

        /// <summary>
        /// Gets a tensor representing the elements below and including the diagonal, with the rest of the elements zero-ed.
        /// </summary>
        /// <returns>A tensor with the values from this tensor at and below the main diagonal and zeros elsewhere.</returns>
        public Tensor<T> GetTriangle()
        {
            return GetTriangle(0, upper: false);
        }

        /// <summary>
        /// Gets a tensor representing the elements below and including the specified diagonal, with the rest of the elements zero-ed.
        /// </summary>
        /// <param name="offset">Offset of diagonal to set in returned tensor.  0 for the main diagonal, less than zero for diagonals below, greater than zero from diagonals above.</param>
        /// <returns>A tensor with the values from this tensor at and below the specified diagonal and zeros elsewhere.</returns>
        public Tensor<T> GetTriangle(int offset)
        {
            return GetTriangle(offset, upper: false);
        }

        /// <summary>
        /// Gets a tensor representing the elements above and including the diagonal, with the rest of the elements zero-ed.
        /// </summary>
        /// <returns>A tensor with the values from this tensor at and above the main diagonal and zeros elsewhere.</returns>
        public Tensor<T> GetUpperTriangle()
        {
            return GetTriangle(0, upper: true);
        }

        /// <summary>
        /// Gets a tensor representing the elements above and including the specified diagonal, with the rest of the elements zero-ed.
        /// </summary>
        /// <param name="offset">Offset of diagonal to set in returned tensor.  0 for the main diagonal, less than zero for diagonals below, greater than zero from diagonals above.</param>
        /// <returns>A tensor with the values from this tensor at and above the specified diagonal and zeros elsewhere.</returns>
        public Tensor<T> GetUpperTriangle(int offset)
        {
            return GetTriangle(offset, upper: true);
        }

        /// <summary>
        /// Implementation method for GetTriangle, GetLowerTriangle, GetUpperTriangle
        /// </summary>
        /// <param name="offset">Offset of diagonal to set in returned tensor.</param>
        /// <param name="upper">true for upper triangular and false otherwise</param>
        /// <returns></returns>
        public Tensor<T> GetTriangle(int offset, bool upper)
        {
            if (Rank < 2)
            {
                throw new InvalidOperationException($"Cannot compute triangle of {nameof(Tensor<T>)} with Rank less than 2.");
            }

            // Similar to get diagonal except it gets every element below and including the diagonal.

            // TODO: allow specification of axis1 and axis2?
            var axisLength0 = dimensions[0];
            var axisLength1 = dimensions[1];
            var diagonalLength = Math.Max(axisLength0, axisLength1);

            var result = CloneEmpty();

            var projectionSize = Length / (axisLength0 * axisLength1);
            var projectionStride = IsReversedStride && Rank > 2 ? strides[2] : 1;

            for (int diagIndex = 0; diagIndex < diagonalLength; diagIndex++)
            {
                // starting point for the tri
                var triIndex0 = offset > 0 ? diagIndex - offset : diagIndex;
                var triIndex1 = offset > 0 ? diagIndex : diagIndex + offset;

                // for lower triangle, iterate index0 keeping same index1
                // for upper triangle, iterate index1 keeping same index0

                if (triIndex0 < 0)
                {
                    if (upper)
                    {
                        // out of bounds, ignore this diagIndex.
                        continue;
                    }
                    else
                    {
                        // set index to 0 so that we can iterate on the remaining index0 values.
                        triIndex0 = 0;
                    }
                }

                if (triIndex1 < 0)
                {
                    if (upper)
                    {
                        // set index to 0 so that we can iterate on the remaining index1 values.
                        triIndex1 = 0;
                    }
                    else
                    {
                        // out of bounds, ignore this diagIndex.
                        continue;
                    }
                }

                while ((triIndex1 < axisLength1) && (triIndex0 < axisLength0))
                {
                    var baseIndex = triIndex0 * strides[0] + triIndex1 * result.strides[1];

                    for (int projectionIndex = 0; projectionIndex < projectionSize; projectionIndex++)
                    {
                        var index = baseIndex + projectionIndex * projectionStride;

                        result.SetValue(index, GetValue(index));
                    }

                    if (upper)
                    {
                        triIndex1++;
                    }
                    else
                    {
                        triIndex0++;
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Reshapes the current tensor to new dimensions, using the same backing storage if possible.
        /// </summary>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the Tensor to create.</param>
        /// <returns>A new tensor that reinterprets this tensor with different dimensions.</returns>
        public abstract Tensor<T> Reshape(ReadOnlySpan<int> dimensions);

        /// <summary>
        /// Obtains the value at the specified indices
        /// </summary>
        /// <param name="indices">A one-dimensional array of integers that represent the indices specifying the position of the element to get.</param>
        /// <returns>The value at the specified position in this Tensor.</returns>
        public virtual T this[params int[] indices]
        {
            get
            {
                if (indices == null)
                {
                    throw new ArgumentNullException(nameof(indices));
                }
                var span = new ReadOnlySpan<int>(indices);
                return this[span];
            }

            set
            {
                if (indices == null)
                {
                    throw new ArgumentNullException(nameof(indices));
                }
                var span = new ReadOnlySpan<int>(indices);
                this[span] = value;
            }
        }

        /// <summary>
        /// Obtains the value at the specified indices
        /// </summary>
        /// <param name="indices">A span integers that represent the indices specifying the position of the element to get.</param>
        /// <returns>The value at the specified position in this Tensor.</returns>
        public virtual T this[ReadOnlySpan<int> indices]
        {
            get
            {
                return GetValue(ArrayUtilities.GetIndex(strides, indices));
            }

            set
            {
                SetValue(ArrayUtilities.GetIndex(strides, indices), value);
            }
        }

        /// <summary>
        /// Gets the value at the specied index, where index is a linearized version of n-dimension indices using strides.
        /// </summary>
        /// <param name="index">An integer index computed as a dot-product of indices.</param>
        /// <returns>The value at the specified position in this Tensor.</returns>
        public abstract T GetValue(int index);

        /// <summary>
        /// Sets the value at the specied index, where index is a linearized version of n-dimension indices using strides.
        /// </summary>
        /// <param name="index">An integer index computed as a dot-product of indices.</param>
        /// <param name="value">The new value to set at the specified position in this Tensor.</param>
        public abstract void SetValue(int index, T value);


        #region statics
        /// <summary>
        /// Performs a value comparison of the content and shape of two tensors.  Two tensors are equal if they have the same shape and same value at every set of indices.  If not equal a tensor is greater or less than another tensor based on the first non-equal element when enumerating in linear order.
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns></returns>
        public static int Compare(Tensor<T> left, Tensor<T> right)
        {
            return StructuralComparisons.StructuralComparer.Compare(left, right);
        }

        /// <summary>
        /// Performs a value equality comparison of the content of two tensors. Two tensors are equal if they have the same shape and same value at every set of indices.
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns></returns>
        public static bool Equals(Tensor<T> left, Tensor<T> right)
        {
            return StructuralComparisons.StructuralEqualityComparer.Equals(left, right);
        }
        #endregion

        #region IEnumerable members
        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((IEnumerable<T>)this).GetEnumerator();
        }
        #endregion

        #region ICollection members
        int ICollection.Count => (int)Length;

        bool ICollection.IsSynchronized => false;

        object ICollection.SyncRoot => this; // backingArray.this?

        void ICollection.CopyTo(Array array, int index)
        {
            if (array is T[] destinationArray)
            {
                CopyTo(destinationArray, index);
            }
            else
            {
                if (array == null)
                {
                    throw new ArgumentNullException(nameof(array));
                }
                if (array.Rank != 1)
                {
                    throw new ArgumentException("Only single dimensional arrays are supported for the requested action.", nameof(array));
                }
                if (array.Length < index + Length)
                {
                    throw new ArgumentException("The number of elements in the Tensor is greater than the available space from index to the end of the destination array.", nameof(array));
                }

                for (int i = 0; i < length; i++)
                {
                    array.SetValue(GetValue(i), index + i);
                }
            }
        }
        #endregion

        #region IList members
        object IList.this[int index]
        {
            get
            {
                return GetValue(index);
            }
            set
            {
                try
                {
                    SetValue(index, (T)value);
                }
                catch (InvalidCastException)
                {
                    throw new ArgumentException($"The value \"{value}\" is not of type \"{typeof(T)}\" and cannot be used in this generic collection.");
                }
            }
        }

        /// <summary>
        /// Always fixed size Tensor
        /// </summary>
        /// <value>always true</value>
        public bool IsFixedSize => true;

        /// <summary>
        /// Tensor is not readonly
        /// </summary>
        /// <value>always false</value>
        public bool IsReadOnly => false;

        int IList.Add(object value)
        {
            throw new InvalidOperationException();
        }

        void IList.Clear()
        {
            Fill(default(T));
        }

        bool IList.Contains(object value)
        {
            if (IsCompatibleObject(value))
            {
                return Contains((T)value);
            }
            return false;
        }

        int IList.IndexOf(object value)
        {
            if (IsCompatibleObject(value))
            {
                return IndexOf((T)value);
            }
            return -1;
        }

        void IList.Insert(int index, object value)
        {
            throw new InvalidOperationException();
        }

        void IList.Remove(object value)
        {
            throw new InvalidOperationException();
        }

        void IList.RemoveAt(int index)
        {
            throw new InvalidOperationException();
        }
        #endregion

        #region IEnumerable<T> members
        IEnumerator<T> IEnumerable<T>.GetEnumerator()
        {
            for (int i = 0; i < Length; i++)
            {
                yield return GetValue(i);
            }
        }
        #endregion

        #region ICollection<T> members
        int ICollection<T>.Count => (int)Length;

        void ICollection<T>.Add(T item)
        {
            throw new InvalidOperationException();
        }

        void ICollection<T>.Clear()
        {
            Fill(default(T));
        }

        bool ICollection<T>.Contains(T item)
        {
            return Contains(item);
        }

        /// <summary>
        /// Determines whether an element is in the Tensor&lt;T&gt;.
        /// </summary>
        /// <param name="item">
        /// The object to locate in the Tensor&lt;T&gt;. The value can be null for reference types.
        /// </param>
        /// <returns>
        /// true if item is found in the Tensor&lt;T&gt;; otherwise, false.
        /// </returns>
        protected virtual bool Contains(T item)
        {
            return Length != 0 && IndexOf(item) != -1;
        }

        void ICollection<T>.CopyTo(T[] array, int arrayIndex)
        {
            CopyTo(array, arrayIndex);
        }

        /// <summary>
        /// Copies the elements of the Tensor&lt;T&gt; to an Array, starting at a particular Array index.
        /// </summary>
        /// <param name="array">
        /// The one-dimensional Array that is the destination of the elements copied from Tensor&lt;T&gt;. The Array must have zero-based indexing.
        /// </param>
        /// <param name="arrayIndex">
        /// The zero-based index in array at which copying begins.
        /// </param>
        protected virtual void CopyTo(T[] array, int arrayIndex)
        {
            if (array == null)
            {
                throw new ArgumentNullException(nameof(array));
            }
            if (array.Length < arrayIndex + Length)
            {
                throw new ArgumentException("The number of elements in the Tensor is greater than the available space from index to the end of the destination array.", nameof(array));
            }

            for (int i = 0; i < length; i++)
            {
                array[arrayIndex + i] = GetValue(i);
            }
        }

        bool ICollection<T>.Remove(T item)
        {
            throw new InvalidOperationException();
        }
        #endregion

        #region IReadOnlyCollection<T> members

        int IReadOnlyCollection<T>.Count => (int)Length;

        #endregion

        #region IList<T> members
        T IList<T>.this[int index]
        {
            get { return GetValue(index); }
            set { SetValue(index, value); }
        }

        int IList<T>.IndexOf(T item)
        {
            return IndexOf(item);
        }

        /// <summary>
        /// Determines the index of a specific item in the Tensor&lt;T&gt;.
        /// </summary>
        /// <param name="item">The object to locate in the Tensor&lt;T&gt;.</param>
        /// <returns>The index of item if found in the tensor; otherwise, -1.</returns>
        protected virtual int IndexOf(T item)
        {
            for (int i = 0; i < Length; i++)
            {
                if (GetValue(i).Equals(item))
                {
                    return i;
                }
            }

            return -1;
        }

        void IList<T>.Insert(int index, T item)
        {
            throw new InvalidOperationException();
        }

        void IList<T>.RemoveAt(int index)
        {
            throw new InvalidOperationException();
        }
        #endregion

        #region IReadOnlyList<T> members

        T IReadOnlyList<T>.this[int index] => GetValue(index);

        #endregion

        #region IStructuralComparable members
        int IStructuralComparable.CompareTo(object other, IComparer comparer)
        {
            if (other == null)
            {
                return 1;
            }

            if (other is Tensor<T>)
            {
                return CompareTo((Tensor<T>)other, comparer);
            }

            var otherArray = other as Array;

            if (otherArray != null)
            {
                return CompareTo(otherArray, comparer);
            }

            throw new ArgumentException($"Cannot compare {nameof(Tensor<T>)} to {other.GetType()}.", nameof(other));
        }

        private int CompareTo(Tensor<T> other, IComparer comparer)
        {
            if (Rank != other.Rank)
            {
                throw new ArgumentException($"Cannot compare {nameof(Tensor<T>)} with Rank {Rank} to {nameof(other)} with Rank {other.Rank}.", nameof(other));
            }

            for (int i = 0; i < dimensions.Length; i++)
            {
                if (dimensions[i] != other.dimensions[i])
                {
                    throw new ArgumentException($"Cannot compare {nameof(Tensor<T>)}s with differning dimension {i}, {dimensions[i]} != {other.dimensions[i]}.", nameof(other));
                }
            }

            int result = 0;

            if (IsReversedStride == other.IsReversedStride)
            {
                for (int i = 0; i < Length; i++)
                {
                    result = comparer.Compare(GetValue(i), other.GetValue(i));
                    if (result != 0)
                    {
                        break;
                    }
                }
            }
            else
            {
                var indices = Rank < ArrayUtilities.StackallocMax ? stackalloc int[Rank] : new int[Rank];
                for (int i = 0; i < Length; i++)
                {
                    ArrayUtilities.GetIndices(strides, IsReversedStride, i, indices);
                    result = comparer.Compare(this[indices], other[indices]);
                    if (result != 0)
                    {
                        break;
                    }
                }
            }

            return result;
        }

        private int CompareTo(Array other, IComparer comparer)
        {
            if (Rank != other.Rank)
            {
                throw new ArgumentException($"Cannot compare {nameof(Tensor<T>)} with Rank {Rank} to {nameof(Array)} with rank {other.Rank}.", nameof(other));
            }

            for (int i = 0; i < dimensions.Length; i++)
            {
                var otherDimension = other.GetLength(i);
                if (dimensions[i] != otherDimension)
                {
                    throw new ArgumentException($"Cannot compare {nameof(Tensor<T>)} to {nameof(Array)} with differning dimension {i}, {dimensions[i]} != {otherDimension}.", nameof(other));
                }
            }

            int result = 0;
            var indices = new int[Rank];
            for (int i = 0; i < Length; i++)
            {
                ArrayUtilities.GetIndices(strides, IsReversedStride, i, indices);

                result = comparer.Compare(GetValue(i), other.GetValue(indices));

                if (result != 0)
                {
                    break;
                }
            }

            return result;
        }
        #endregion

        #region IStructuralEquatable members
        bool IStructuralEquatable.Equals(object other, IEqualityComparer comparer)
        {
            if (other == null)
            {
                return false;
            }

            if (other is Tensor<T>)
            {
                return Equals((Tensor<T>)other, comparer);
            }

            var otherArray = other as Array;

            if (otherArray != null)
            {
                return Equals(otherArray, comparer);
            }

            throw new ArgumentException($"Cannot compare {nameof(Tensor<T>)} to {other.GetType()}.", nameof(other));
        }

        private bool Equals(Tensor<T> other, IEqualityComparer comparer)
        {
            if (Rank != other.Rank)
            {
                throw new ArgumentException($"Cannot compare {nameof(Tensor<T>)} with Rank {Rank} to {nameof(other)} with Rank {other.Rank}.", nameof(other));
            }

            for (int i = 0; i < dimensions.Length; i++)
            {
                if (dimensions[i] != other.dimensions[i])
                {
                    throw new ArgumentException($"Cannot compare {nameof(Tensor<T>)}s with differning dimension {i}, {dimensions[i]} != {other.dimensions[i]}.", nameof(other));
                }
            }

            if (IsReversedStride == other.IsReversedStride)
            {
                for (int i = 0; i < Length; i++)
                {
                    if (!comparer.Equals(GetValue(i), other.GetValue(i)))
                    {
                        return false;
                    }
                }
            }
            else
            {
                var indices = Rank < ArrayUtilities.StackallocMax ? stackalloc int[Rank] : new int[Rank];
                for (int i = 0; i < Length; i++)
                {
                    ArrayUtilities.GetIndices(strides, IsReversedStride, i, indices);

                    if (!comparer.Equals(this[indices], other[indices]))
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        private bool Equals(Array other, IEqualityComparer comparer)
        {
            if (Rank != other.Rank)
            {
                throw new ArgumentException($"Cannot compare {nameof(Tensor<T>)} with Rank {Rank} to {nameof(Array)} with rank {other.Rank}.", nameof(other));
            }

            for (int i = 0; i < dimensions.Length; i++)
            {
                var otherDimension = other.GetLength(i);
                if (dimensions[i] != otherDimension)
                {
                    throw new ArgumentException($"Cannot compare {nameof(Tensor<T>)} to {nameof(Array)} with differning dimension {i}, {dimensions[i]} != {otherDimension}.", nameof(other));
                }
            }

            var indices = new int[Rank];
            for (int i = 0; i < Length; i++)
            {
                ArrayUtilities.GetIndices(strides, IsReversedStride, i, indices);

                if (!comparer.Equals(GetValue(i), other.GetValue(indices)))
                {
                    return false;
                }
            }

            return true;
        }
        int IStructuralEquatable.GetHashCode(IEqualityComparer comparer)
        {
            int hashCode = 0;
            // this ignores shape, which is fine  it just means we'll have hash collisions for things 
            // with the same content and different shape.
            for (int i = 0; i < Length; i++)
            {
                hashCode ^= comparer.GetHashCode(GetValue(i));
            }

            return hashCode;
        }
        #endregion

        #region Translations

        /// <summary>
        /// Creates a copy of this tensor as a DenseTensor&lt;T&gt;.  If this tensor is already a DenseTensor&lt;T&gt; calling this method is equivalent to calling Clone().
        /// </summary>
        /// <returns></returns>
        public virtual DenseTensor<T> ToDenseTensor()
        {
            var denseTensor = new DenseTensor<T>(Dimensions, IsReversedStride);
            for (int i = 0; i < Length; i++)
            {
                denseTensor.SetValue(i, GetValue(i));
            }
            return denseTensor;
        }

        #endregion

        /// <summary>
        /// Get a string representation of Tensor
        /// </summary>
        /// <param name="includeWhitespace"></param>
        /// <returns></returns>
        public string GetArrayString(bool includeWhitespace = true)
        {
            var builder = new StringBuilder();

            var strides = ArrayUtilities.GetStrides(dimensions);
            var indices = new int[Rank];
            var innerDimension = Rank - 1;
            var innerLength = dimensions[innerDimension];
            var outerLength = Length / innerLength;

            int indent = 0;
            for (int outerIndex = 0; outerIndex < Length; outerIndex += innerLength)
            {
                ArrayUtilities.GetIndices(strides, false, outerIndex, indices);

                while ((indent < innerDimension) && (indices[indent] == 0))
                {
                    // start up
                    if (includeWhitespace)
                    {
                        Indent(builder, indent);
                    }
                    indent++;
                    builder.Append('{');
                    if (includeWhitespace)
                    {
                        builder.AppendLine();
                    }
                }

                for (int innerIndex = 0; innerIndex < innerLength; innerIndex++)
                {
                    indices[innerDimension] = innerIndex;

                    if ((innerIndex == 0))
                    {
                        if (includeWhitespace)
                        {
                            Indent(builder, indent);
                        }
                        builder.Append('{');
                    }
                    else
                    {
                        builder.Append(',');
                    }
                    builder.Append(this[indices]);
                }
                builder.Append('}');

                for (int i = Rank - 2; i >= 0; i--)
                {
                    var lastIndex = dimensions[i] - 1;
                    if (indices[i] == lastIndex)
                    {
                        // close out
                        --indent;
                        if (includeWhitespace)
                        {
                            builder.AppendLine();
                            Indent(builder, indent);
                        }
                        builder.Append('}');
                    }
                    else
                    {
                        builder.Append(',');
                        if (includeWhitespace)
                        {
                            builder.AppendLine();
                        }
                        break;
                    }
                }
            }

            return builder.ToString();
        }

        private static void Indent(StringBuilder builder, int tabs, int spacesPerTab = 4)
        {
            for (int tab = 0; tab < tabs; tab++)
            {
                for (int space = 0; space < spacesPerTab; space++)
                {
                    builder.Append(' ');
                }
            }
        }

        private static bool IsCompatibleObject(object value)
        {
            // Non-null values are fine.  Only accept nulls if T is a class or Nullable<T>.
            // Note that default(T) is not equal to null for value types except when T is Nullable<T>.
            return ((value is T) || (value == null && default(T) == null));
        }
    }
}
