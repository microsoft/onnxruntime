// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// The class helps to feed the NamedOnnxValue as inference input.
    /// It projects managed classes to OrtValues so they can be consumed
    /// by the native onnxruntime library. if possible, it will avoid copying data.
    /// The NamedOnnxValue can be a tensor, sequence or map.
    /// For recursive structures, create nested NamedOnnxValue instances.
    /// For example, a sequence instance would contain a list of NamedOnnxValue instances
    /// that in turn may represent tensors or other ONNX values.
    /// </summary>
    internal class ManagedTypeProjection
    {
        /// <summary>
        /// Dispatches the creation of the projection
        /// </summary>
        /// <param name="namedOnnxValue"></param>
        /// <param name="metadata"></param>
        /// <returns>OrtValye created accoding to the metadata</returns>
        internal static OrtValue CreateProjection(NamedOnnxValue namedOnnxValue, NodeMetadata metadata)
        {
            OrtValue result;

            NodeMetadata meta = metadata;
            // Use element meta to create types
            if (metadata.OnnxValueType == OnnxValueType.ONNX_TYPE_OPTIONAL)
            {
                meta = metadata.AsOptionalMetadata().ElementMeta;
            }

            if (namedOnnxValue.ValueType != meta.OnnxValueType)
            {
                throw new OnnxRuntimeException(ErrorCode.RuntimeException,
                    $"NamedOnnxValue: {namedOnnxValue.Name} has value type: {namedOnnxValue.ValueType}" +
                    $" expected: {meta.OnnxValueType} after optional type adjustment");
            }

            switch (namedOnnxValue.ValueType)
            {
                case OnnxValueType.ONNX_TYPE_TENSOR:
                    result = CreateTensorProjection(namedOnnxValue, meta);
                    break;
                case OnnxValueType.ONNX_TYPE_SEQUENCE:
                    result = CreateSequenceProjection(namedOnnxValue, meta);
                    break;
                case OnnxValueType.ONNX_TYPE_MAP:
                    result = CreateMapProjection(namedOnnxValue, meta);
                    break;
                default:
                    throw new OnnxRuntimeException(ErrorCode.InvalidArgument, "ManagedTypeProjection can only project tensors, sequences, maps and optional types");
            }
            return result;
        }

        /// <summary>
        /// The function creates OrtValue objects for each element of the sequence
        /// and then creates an OrtValue for the whole sequence.
        /// </summary>
        /// <param name="namedOnnxValue">NamedOnnxValue containing a IEnumerable<NameOnnValue></param>
        /// <param name="metadata">sequence metadata</param>
        /// <returns>OrtValue that represents a sequence</returns>
        /// <exception cref="OnnxRuntimeException"></exception>
        private static OrtValue CreateSequenceProjection(NamedOnnxValue namedOnnxValue, NodeMetadata metadata)
        {
            var elementMeta = metadata.AsSequenceMetadata().ElementMeta;
            var elementOnnxValue = elementMeta.OnnxValueType;
            var seqContainer = namedOnnxValue.AsEnumerable<NamedOnnxValue>() ??
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                                               $"NamedOnnxValue: {namedOnnxValue.Name} sequence does not contain NamedOnnxValue elements");
            int capacity = 0;

            if (seqContainer is ICollection<NamedOnnxValue> collection)
            {
                capacity = collection.Count;
            }

            DisposableList<OrtValue> sequenceOrtValues = new(capacity);
            try
            {
                foreach (var element in seqContainer)
                {
                    if (elementOnnxValue != element.ValueType)
                    {
                        throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                            $"NamedOnnxValue: {namedOnnxValue.Name} sequence element expected to be {elementOnnxValue}, received {element.ValueType}");
                    }

                    sequenceOrtValues.Add(CreateProjection(element, elementMeta));
                }
                return OrtValue.CreateSequence(ref sequenceOrtValues);
            }
            catch(Exception)
            {
                sequenceOrtValues?.Dispose();
                throw;
            }
        }

        /// <summary>
        /// Creates map projection. Since we support only primitive types in maps
        /// we map two tensors (keys and values)
        /// </summary>
        /// <param name="node"></param>
        /// <param name="elementMeta"></param>
        /// <returns>OrtValue</returns>
        /// <exception cref="OnnxRuntimeException"></exception>
        private static OrtValue CreateMapProjection(NamedOnnxValue node, NodeMetadata elementMeta)
        {
            MapMetadata mapMeta = elementMeta.AsMapMetadata();
            Debug.Assert(mapMeta != null);
            // Maps currently support only primitive types expressed as two parallel tensors and not nested Sequences or Maps

            var mapValuesMeta = mapMeta.ValueMetadata;
            if (mapValuesMeta.OnnxValueType != OnnxValueType.ONNX_TYPE_TENSOR)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                    $"Node: {node.Name} onnxruntime only supports maps with primitive types values");
            }

            Span<OrtValue> ortValues = new OrtValue[2];
            var disposableGuard = new DisposableArray<OrtValue>(ortValues);
            try
            {
                TensorBase keys = node.GetDictionaryKeys();
                ortValues[0] = OrtValue.CreateFromTensorObject(keys, out TensorElementType elementTypeKeys);

                if (elementTypeKeys != mapMeta.KeyDataType)
                {
                    throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                                           $"Map key data type supplied: {elementTypeKeys} metadata expected: {mapMeta.KeyDataType}");
                }

                TensorBase values = node.GetDictionaryValues();
                ortValues[1] = OrtValue.CreateFromTensorObject(values, out TensorElementType elementTypeValues);
                if (elementTypeValues != mapValuesMeta.ElementDataType)
                {
                    throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                                                   $"Map value data type supplied: {elementTypeValues} metadata expected: {mapValuesMeta.ElementDataType}");
                }

                // Create Map OrtValue
                return OrtValue.CreateMap(ref ortValues[0], ref ortValues[1]);
            }
            catch (Exception)
            {
                disposableGuard.Dispose();
                throw;
            }
        }

        /// <summary>
        /// This pins memory that is contained within DenseTensor.
        /// </summary>
        /// <param name="node">NodeOnnxValue containing DenseTensor</param>
        /// <param name="elementMeta"></param>
        /// <returns></returns>
        /// <exception cref="OnnxRuntimeException"></exception>
        private static OrtValue CreateTensorProjection(NamedOnnxValue node, NodeMetadata elementMeta)
        {
            if (node.Value is not TensorBase)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                    $"NamedOnnxValue contains: {node.Value.GetType()}, expecting a Tensor<T>");
            }

            OrtValue ortValue = OrtValue.CreateFromTensorObject(node.Value as TensorBase, out TensorElementType elementType);
            try
            {
                if (elementType != elementMeta.ElementDataType)
                {
                    throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                        $"Tensor element data type discovered: {elementType} metadata expected: {elementMeta.ElementDataType}");
                }
            }
            catch (Exception)
            {
                ortValue.Dispose();
                throw;
            }

            return ortValue;
        }
    }
}

