using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Tensors;
using Xunit;

namespace Microsoft.ML.OnnxRuntime.Tests
{
    internal static class TestDataLoader
    {
        internal static byte[] LoadModelFromEmbeddedResource(string path)
        {
            var assembly = typeof(TestDataLoader).Assembly;
            byte[] model = null;

            var resourceName = assembly.GetManifestResourceNames().Single(p => p.EndsWith("." + path));
            using (Stream stream = assembly.GetManifestResourceStream(resourceName))
            {
                using (MemoryStream memoryStream = new MemoryStream())
                {
                    stream.CopyTo(memoryStream);
                    model = memoryStream.ToArray();
                }
            }

            return model;
        }


        internal static float[] LoadTensorFromEmbeddedResource(string path)
        {
            var tensorData = new List<float>();
            var assembly = typeof(TestDataLoader).Assembly;

            var resourceName = assembly.GetManifestResourceNames().Single(p => p.EndsWith("." + path));
            using (StreamReader inputFile = new StreamReader(assembly.GetManifestResourceStream(resourceName)))
            {
                inputFile.ReadLine(); //skip the input name
                string[] dataStr = inputFile.ReadLine().Split(new char[] { ',', '[', ']' }, StringSplitOptions.RemoveEmptyEntries);
                for (int i = 0; i < dataStr.Length; i++)
                {
                    tensorData.Add(Single.Parse(dataStr[i]));
                }
            }

            return tensorData.ToArray();
        }

        static NamedOnnxValue LoadTensorPb(Onnx.TensorProto tensor, IReadOnlyDictionary<string, NodeMetadata> nodeMetaDict)
        {
            var intDims = new int[tensor.Dims.Count];
            for (int i = 0; i < tensor.Dims.Count; i++)
            {
                intDims[i] = (int)tensor.Dims[i];
            }

            NodeMetadata nodeMeta = null;
            string nodeName = string.Empty;

            if (nodeMetaDict.Count == 1)
            {
                nodeMeta = nodeMetaDict.Values.First();
                nodeName = nodeMetaDict.Keys.First(); // valid for single node input
            }
            else if (nodeMetaDict.Count > 1)
            {
                if (tensor.Name.Length > 0)
                {
                    nodeMeta = nodeMetaDict[tensor.Name];
                    nodeName = tensor.Name;
                    if (!nodeMeta.IsTensor)
                        throw new Exception("LoadTensorFromFile can load Tensor types only: " + nodeName);
                }
                else
                {
                    bool matchfound = false;
                    // try to find from matching type and shape
                    foreach (var key in nodeMetaDict.Keys)
                    {
                        var meta = nodeMetaDict[key];
                        if (!meta.IsTensor)
                            throw new Exception("LoadTensorFromFile can load Tensor types only");

                        if ((Tensors.TensorElementType)tensor.DataType == meta.ElementDataType && tensor.Dims.Count == meta.Dimensions.Length)
                        {
                            int i = 0;
                            for (; i < meta.Dimensions.Length; i++)
                            {
                                if (meta.Dimensions[i] != -1 && meta.Dimensions[i] != intDims[i])
                                {
                                    break;
                                }
                            }
                            if (i >= meta.Dimensions.Length)
                            {
                                matchfound = true;
                                nodeMeta = meta;
                                nodeName = key;
                                break;
                            }
                        }
                    }
                    if (!matchfound)
                    {
                        // throw error
                        throw new Exception($"No Matching Tensor found in InputOutputMetadata corresponding to the serialized tensor specified");
                    }
                }
            }
            else
            {
                // throw error
                throw new Exception($"While reading the serliazed tensor specified, metaDataDict has 0 elements");
            }

            if (nodeMeta.OnnxValueType != OnnxValueType.ONNX_TYPE_TENSOR)
                throw new Exception("LoadTensorFromFile can load Dense Tensor types only");

            var protoDt = (Tensors.TensorElementType)tensor.DataType;
            if (!((protoDt == nodeMeta.ElementDataType) ||
                (protoDt == TensorElementType.UInt16 && 
                (nodeMeta.ElementDataType == TensorElementType.BFloat16 || nodeMeta.ElementDataType == TensorElementType.Float16))))
                throw new Exception($"{tensor.DataType.ToString()} is expected to be equal to: " + nodeMeta.ElementDataType.ToString());

            if (nodeMeta.Dimensions.Length != tensor.Dims.Count)
                throw new Exception($"{nameof(nodeMeta.Dimensions.Length)} is expected to be equal to {nameof(tensor.Dims.Count)}");

            for (int i = 0; i < nodeMeta.Dimensions.Length; i++)
            {
                if ((nodeMeta.Dimensions[i] != -1) && (nodeMeta.Dimensions[i] != intDims[i]))
                    throw new Exception($"{nameof(nodeMeta.Dimensions)}[{i}] is expected to either be -1 or {nameof(intDims)}[{i}]");
            }

            var elementType = nodeMeta.ElementDataType;
            switch (elementType)
            {
                case TensorElementType.Float:
                    return CreateNamedOnnxValueFromRawData<float>(nodeName, tensor.RawData.ToArray(), sizeof(float), intDims);
                case TensorElementType.Double:
                    return CreateNamedOnnxValueFromRawData<double>(nodeName, tensor.RawData.ToArray(), sizeof(double), intDims);
                case TensorElementType.Int32:
                    return CreateNamedOnnxValueFromRawData<int>(nodeName, tensor.RawData.ToArray(), sizeof(int), intDims);
                case TensorElementType.UInt32:
                    return CreateNamedOnnxValueFromRawData<uint>(nodeName, tensor.RawData.ToArray(), sizeof(uint), intDims);
                case TensorElementType.Int16:
                    return CreateNamedOnnxValueFromRawData<short>(nodeName, tensor.RawData.ToArray(), sizeof(short), intDims);
                case TensorElementType.UInt16:
                    return CreateNamedOnnxValueFromRawData<ushort>(nodeName, tensor.RawData.ToArray(), sizeof(ushort), intDims);
                case TensorElementType.Int64:
                    return CreateNamedOnnxValueFromRawData<long>(nodeName, tensor.RawData.ToArray(), sizeof(long), intDims);
                case TensorElementType.UInt64:
                    return CreateNamedOnnxValueFromRawData<ulong>(nodeName, tensor.RawData.ToArray(), sizeof(ulong), intDims);
                case TensorElementType.UInt8:
                    return CreateNamedOnnxValueFromRawData<byte>(nodeName, tensor.RawData.ToArray(), sizeof(byte), intDims);
                case TensorElementType.Int8:
                    return CreateNamedOnnxValueFromRawData<sbyte>(nodeName, tensor.RawData.ToArray(), sizeof(sbyte), intDims);
                case TensorElementType.Bool:
                    return CreateNamedOnnxValueFromRawData<bool>(nodeName, tensor.RawData.ToArray(), sizeof(bool), intDims);
                case TensorElementType.Float16:
                    return CreateNamedOnnxValueFromRawData<Float16>(nodeName, tensor.RawData.ToArray(), sizeof(ushort), intDims);
                case TensorElementType.BFloat16:
                    return CreateNamedOnnxValueFromRawData<BFloat16>(nodeName, tensor.RawData.ToArray(), sizeof(ushort), intDims);
                case TensorElementType.String:
                    return CreateNamedOnnxValueFromString(tensor, intDims);
                default:
                    throw new Exception($"Tensors of type: " + nodeMeta.ElementType.ToString() + " not currently supported in the LoadTensorFromEmbeddedResource");
            }
        }

        internal static NamedOnnxValue LoadTensorFromEmbeddedResourcePb(string path, IReadOnlyDictionary<string, NodeMetadata> nodeMetaDict)
        {
            Onnx.TensorProto tensor = null;

            var assembly = typeof(TestDataLoader).Assembly;

            using (Stream stream = assembly.GetManifestResourceStream($"{assembly.GetName().Name}.TestData.{path}"))
            {
                tensor = Onnx.TensorProto.Parser.ParseFrom(stream);
            }

            return LoadTensorPb(tensor, nodeMetaDict);
        }

        internal static NamedOnnxValue LoadTensorFromFilePb(string filename, IReadOnlyDictionary<string, NodeMetadata> nodeMetaDict)
        {
            //Set buffer size to 4MB
            int readBufferSize = 4194304;
            Onnx.TensorProto tensor = null;
            using (var file = new FileStream(filename, FileMode.Open, FileAccess.Read, FileShare.Read, readBufferSize))
            {
                tensor = Onnx.TensorProto.Parser.ParseFrom(file);
            }

            return LoadTensorPb(tensor, nodeMetaDict);
        }

        internal static NamedOnnxValue CreateNamedOnnxValueFromRawData<T>(string name, byte[] rawData, int elemWidth, int[] dimensions)
        {
            T[] typedArr = new T[rawData.Length / elemWidth];
            var typeOf = typeof(T);
            if (typeOf == typeof(Float16) || typeOf == typeof(BFloat16))
            {
                using (var memSrcHandle = new Memory<byte>(rawData).Pin())
                using (var memDstHandle = new Memory<T>(typedArr).Pin())
                {
                    unsafe
                    {
                        Buffer.MemoryCopy(memSrcHandle.Pointer, memDstHandle.Pointer, typedArr.Length * elemWidth, rawData.Length);
                    }
                }
            }
            else
            {
                Buffer.BlockCopy(rawData, 0, typedArr, 0, rawData.Length);
            }
            var dt = new DenseTensor<T>(typedArr, dimensions);
            return NamedOnnxValue.CreateFromTensor<T>(name, dt);
        }

        internal static NamedOnnxValue CreateNamedOnnxValueFromString(Onnx.TensorProto tensor, int[] dimensions)
        {   
            if (tensor.DataType != (int)Onnx.TensorProto.Types.DataType.String)
            {
                throw new ArgumentException("Expecting string data");
            }

            string[] strArray = new string[tensor.StringData.Count];
            for (int i = 0; i < tensor.StringData.Count; ++i)
            {
                strArray[i] = System.Text.Encoding.UTF8.GetString(tensor.StringData[i].ToByteArray());
            }

            var dt = new DenseTensor<string>(strArray, dimensions);
            return NamedOnnxValue.CreateFromTensor<string>(tensor.Name, dt);
        }

        internal static float[] LoadTensorFromFile(string filename, bool skipheader = true)
        {
            var tensorData = new List<float>();

            // read data from file
            using (var inputFile = new System.IO.StreamReader(filename))
            {
                if (skipheader)
                    inputFile.ReadLine(); //skip the input name
                string[] dataStr = inputFile.ReadLine().Split(new char[] { ',', '[', ']', ' ' }, StringSplitOptions.RemoveEmptyEntries);
                for (int i = 0; i < dataStr.Length; i++)
                {
                    tensorData.Add(Single.Parse(dataStr[i]));
                }
            }

            return tensorData.ToArray();
        }
    }
}