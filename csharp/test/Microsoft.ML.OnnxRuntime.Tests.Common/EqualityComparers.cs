using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;

namespace Microsoft.ML.OnnxRuntime.Tests
{
    internal class FloatComparer : IEqualityComparer<float>
    {
        private float atol = 1e-3f;
        private float rtol = 1.7e-2f;

        public bool Equals(float x, float y)
        {
            return Math.Abs(x - y) <= (atol + rtol * Math.Abs(y));
        }
        public int GetHashCode(float x)
        {
            return x.GetHashCode();
        }
    }

    internal class DoubleComparer : IEqualityComparer<double>
    {
        private double atol = 1e-3;
        private double rtol = 1.7e-2;

        public bool Equals(double x, double y)
        {
            return Math.Abs(x - y) <= (atol + rtol * Math.Abs(y));
        }
        public int GetHashCode(double x)
        {
            return x.GetHashCode();
        }
    }

    internal class ExactComparer<T> : IEqualityComparer<T>
    {
        public bool Equals(T x, T y)
        {
            return x.Equals(y);
        }
        public int GetHashCode(T x)
        {
            return x.GetHashCode();
        }
    }

    /// <summary>
    /// Use it to compare Float16
    /// </summary>
    internal class Float16Comparer : IEqualityComparer<Float16>
    {
        public ushort tolerance = 0;
        public bool Equals(Float16 x, Float16 y)
        {
            return Math.Abs(x.value - y.value) <= (tolerance + y.value);
        }
        public int GetHashCode(Float16 x)
        {
            return x.GetHashCode();
        }
    }

    /// <summary>
    /// Use it to compare Bloat16
    /// </summary>
    internal class BFloat16Comparer : IEqualityComparer<BFloat16>
    {
        public ushort tolerance = 0;
        public bool Equals(BFloat16 x, BFloat16 y)
        {
            return Math.Abs(x.value - y.value) <= (tolerance + y.value);
        }
        public int GetHashCode(BFloat16 x)
        {
            return x.GetHashCode();
        }
    }
}
