using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime.Tests;
using System.Reflection;

namespace MauiModelTester
{
    internal class Utils
    {
        internal static async Task<byte[]> LoadResource(string name)
        {
            using Stream fileStream = await FileSystem.Current.OpenAppPackageFileAsync(name);
            using MemoryStream memoryStream = new MemoryStream();
            fileStream.CopyTo(memoryStream);
            return memoryStream.ToArray();
        }

        internal static async Task<(Dictionary<string, OrtValue>, Dictionary<string, OrtValue>)> LoadTestData()
        {
            var loadData = async (string prefix) =>
            {
                var data = new Dictionary<string, OrtValue>();
                int idx = 0;

                do
                {
                    var filename = "test_data/test_data_set_0/" + prefix + idx + ".pb";
                    var exists = await FileSystem.Current.AppPackageFileExistsAsync(filename);

                    if (!exists)
                    {
                        // we expect sequentially named files for all inputs so as soon as one is missing we're done
                        break;
                    }

                    var tensorProtoData = await LoadResource(filename);

                    // get name and tensor data and create OrtValue
                    Onnx.TensorProto tensorProto = null;
                    tensorProto = Onnx.TensorProto.Parser.ParseFrom(tensorProtoData);
                    var ortValue = CreateOrtValueFromTensorProto(tensorProto);

                    data[tensorProto.Name] = ortValue;

                    idx++;
                }
                while (true);

                return data;
            };

            var inputData = await loadData("input_");
            var outputData = await loadData("output_");

            return (inputData, outputData);
        }

        internal static OrtValue CreateOrtValueFromTensorProto(Onnx.TensorProto tensorProto)
        {
            Type tensorElementType = GetElementType((TensorElementType)tensorProto.DataType);
            OrtValue ortValue = null;

            // special case for strings
            if (tensorElementType == typeof(string))
            {
                var numElements = tensorProto.Dims.Aggregate(1L, (x, y) => x * y);
                ortValue = OrtValue.CreateTensorWithEmptyStrings(OrtAllocator.DefaultInstance,
                                                                 tensorProto.Dims.ToArray());

                int idx = 0;
                foreach (var str in tensorProto.StringData)
                {
                    ortValue.StringTensorSetElementAt(str.Span, idx++);
                }
            }
            else
            {
                // use reflection to call generic method
                var func = typeof(Utils)
                               .GetMethod(nameof(TensorProtoToOrtValue), BindingFlags.Static | BindingFlags.NonPublic)
                               .MakeGenericMethod(tensorElementType);

                ortValue = (OrtValue)func.Invoke(null, new[] { tensorProto });
            }

            return ortValue;
        }

        internal static Type GetElementType(TensorElementType elemType)
        {
            switch (elemType)
            {
                case TensorElementType.Float:
                    return typeof(float);
                case TensorElementType.Double:
                    return typeof(double);
                case TensorElementType.Int16:
                    return typeof(short);
                case TensorElementType.UInt16:
                    return typeof(ushort);
                case TensorElementType.Int32:
                    return typeof(int);
                case TensorElementType.UInt32:
                    return typeof(uint);
                case TensorElementType.Int64:
                    return typeof(long);
                case TensorElementType.UInt64:
                    return typeof(ulong);
                case TensorElementType.UInt8:
                    return typeof(byte);
                case TensorElementType.Int8:
                    return typeof(sbyte);
                case TensorElementType.String:
                    return typeof(string);
                case TensorElementType.Bool:
                    return typeof(bool);
                default:
                    throw new ArgumentException("Unexpected element type of " + elemType);
            }
        }

        static OrtValue TensorProtoToOrtValue<T>(Onnx.TensorProto tensorProto)
            where T : unmanaged
        {
            unsafe
            {
                var elementSize = sizeof(T);
                T[] data = new T[tensorProto.RawData.Length / elementSize];

                fixed(byte *bytes = tensorProto.RawData.Span) fixed(void *target = data)
                {
                    Buffer.MemoryCopy(bytes, target, tensorProto.RawData.Length, tensorProto.RawData.Length);
                }

                return OrtValue.CreateTensorValueFromMemory(data, tensorProto.Dims.ToArray());
            }
        }

        internal class TensorComparer
        {
            // we need to use a delegate in the checker func to handle string as well as numeric types
            private delegate ReadOnlySpan<T> GetDataFn<T>(OrtValue ortValue);

            private static ReadOnlySpan<T> GetData<T>(OrtValue ortValue)
                where T : unmanaged
            {
                return ortValue.GetTensorDataAsSpan<T>();
            }

            private static ReadOnlySpan<string> GetStringData(OrtValue ortValue)
            {
                return ortValue.GetStringTensorAsArray();
            }

            private static void CheckEqual<T>(string name, OrtValue expected, OrtValue actual,
                                              IEqualityComparer<T> comparer, GetDataFn<T> getDataFn)
            {
                var expectedTypeAndShape = expected.GetTypeInfo().TensorTypeAndShapeInfo;
                var actualTypeAndShape = actual.GetTypeInfo().TensorTypeAndShapeInfo;

                if (expectedTypeAndShape.ElementCount != actualTypeAndShape.ElementCount)
                {
                    throw new ArithmeticException(
                        $"Element count mismatch for {name}. " +
                        $"Expected:{expectedTypeAndShape.ElementCount} Actual:{actualTypeAndShape.ElementCount}");
                }

                var expectedData = getDataFn(expected);
                var actualData = getDataFn(actual);

                List<string> mismatches = new List<string>();

                for (int i = 0; i < expectedData.Length; i++)
                {
                    if (!comparer.Equals(expectedData[i], actualData[i]))
                    {
                        mismatches.Add($"[{i}] {expectedData[i]} != {actualData[i]}");
                    }
                }

                if (mismatches.Count > 0)
                {
                    throw new ArithmeticException(
                        $"Result mismatch for {name}. Mismatched entries:{string.Join(',', mismatches)}");
                }
            }

            private static void CheckEqual<T>(string name, OrtValue expected, OrtValue actual,
                                              IEqualityComparer<T> comparer)
                where T : unmanaged
            {
                CheckEqual(name, expected, actual, comparer, GetData<T>);
            }

            internal static void VerifyTensorResults(string name, OrtValue expected, OrtValue actual)
            {
                var tensorElementType = expected.GetTypeInfo().TensorTypeAndShapeInfo.ElementDataType;
                switch (tensorElementType)
                {
                    case TensorElementType.Float:
                        CheckEqual(name, expected, actual, new FloatComparer());
                        break;
                    case TensorElementType.Double:
                        CheckEqual(name, expected, actual, new DoubleComparer());
                        break;
                    case TensorElementType.Int32:
                        CheckEqual(name, expected, actual, new ExactComparer<int>());
                        break;
                    case TensorElementType.UInt32:
                        CheckEqual(name, expected, actual, new ExactComparer<uint>());
                        break;
                    case TensorElementType.Int16:
                        CheckEqual(name, expected, actual, new ExactComparer<short>());
                        break;
                    case TensorElementType.UInt16:
                        CheckEqual(name, expected, actual, new ExactComparer<ushort>());
                        break;
                    case TensorElementType.Int64:
                        CheckEqual(name, expected, actual, new ExactComparer<long>());
                        break;
                    case TensorElementType.UInt64:
                        CheckEqual(name, expected, actual, new ExactComparer<ulong>());
                        break;
                    case TensorElementType.UInt8:
                        CheckEqual(name, expected, actual, new ExactComparer<byte>());
                        break;
                    case TensorElementType.Int8:
                        CheckEqual(name, expected, actual, new ExactComparer<sbyte>());
                        break;
                    case TensorElementType.Bool:
                        CheckEqual(name, expected, actual, new ExactComparer<bool>());
                        break;
                    case TensorElementType.Float16:
                        CheckEqual(name, expected, actual, new Float16Comparer { tolerance = 2 });
                        break;
                    case TensorElementType.BFloat16:
                        CheckEqual(name, expected, actual, new BFloat16Comparer { tolerance = 2 });
                        break;
                    case TensorElementType.String:
                        CheckEqual<string>(name, expected, actual, new ExactComparer<string>(), GetStringData);
                        break;
                    default:
                        throw new ArgumentException($"Unexpected data type of {tensorElementType}");
                }
            }
        }
    }
}
