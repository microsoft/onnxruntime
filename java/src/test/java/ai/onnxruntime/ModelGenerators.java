/*
 * Copyright (c) 2023, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import ai.onnxruntime.OnnxMl.StringStringEntryProto;
import ai.onnxruntime.OnnxMl.TensorProto.DataLocation;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

/** Methods to generate test models. */
public final class ModelGenerators {
  private ModelGenerators() {}

  public static OnnxMl.TensorShapeProto getShapeProto(
      long[] dimensions, String[] dimensionOverrides) {
    OnnxMl.TensorShapeProto.Builder builder = OnnxMl.TensorShapeProto.newBuilder();
    for (int i = 0; i < dimensions.length; i++) {
      if (dimensions[i] == -1) {
        builder.addDim(
            OnnxMl.TensorShapeProto.Dimension.newBuilder()
                .setDimParam(dimensionOverrides[i])
                .build());
      } else {
        builder.addDim(
            OnnxMl.TensorShapeProto.Dimension.newBuilder().setDimValue(dimensions[i]).build());
      }
    }
    return builder.build();
  }

  public static OnnxMl.TypeProto buildTensorTypeNode(
      long[] dimensions, String[] dimensionOverrides, OnnxMl.TensorProto.DataType type) {
    OnnxMl.TypeProto.Builder builder = OnnxMl.TypeProto.newBuilder();

    OnnxMl.TypeProto.Tensor.Builder tensorBuilder = OnnxMl.TypeProto.Tensor.newBuilder();
    tensorBuilder.setElemType(type.getNumber());
    tensorBuilder.setShape(getShapeProto(dimensions, dimensionOverrides));
    builder.setTensorType(tensorBuilder.build());

    return builder.build();
  }

  public void generateExternalMatMul() throws IOException {
    OnnxMl.GraphProto.Builder graph = OnnxMl.GraphProto.newBuilder();
    graph.setName("ort-test-matmul");

    // Add placeholders
    OnnxMl.ValueInfoProto.Builder input = OnnxMl.ValueInfoProto.newBuilder();
    input.setName("input");
    OnnxMl.TypeProto inputType =
        buildTensorTypeNode(
            new long[] {-1, 4},
            new String[] {"batch_size", null},
            OnnxMl.TensorProto.DataType.FLOAT);
    input.setType(inputType);
    graph.addInput(input);
    OnnxMl.ValueInfoProto.Builder output = OnnxMl.ValueInfoProto.newBuilder();
    output.setName("output");
    OnnxMl.TypeProto outputType =
        buildTensorTypeNode(
            new long[] {-1, 4},
            new String[] {"batch_size", null},
            OnnxMl.TensorProto.DataType.FLOAT);
    output.setType(outputType);
    graph.addOutput(output);

    // Add initializers
    OnnxMl.TensorProto.Builder tensor = OnnxMl.TensorProto.newBuilder();
    tensor.addDims(4);
    tensor.addDims(4);
    tensor.setDataLocation(DataLocation.EXTERNAL);
    tensor.addExternalData(
        StringStringEntryProto.newBuilder()
            .setKey("location")
            .setValue("external-matmul.out")
            .build());
    tensor.addExternalData(
        StringStringEntryProto.newBuilder().setKey("offset").setValue("0").build());
    tensor.addExternalData(
        StringStringEntryProto.newBuilder().setKey("length").setValue("64").build());
    float[] floats =
        new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f};
    ByteBuffer buf = ByteBuffer.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
    FloatBuffer floatBuf = buf.asFloatBuffer();
    floatBuf.put(floats);
    floatBuf.rewind();
    buf.rewind();
    try (OutputStream os =
        Files.newOutputStream(Paths.get("src", "test", "resources", "external-matmul.out"))) {
      os.write(buf.array());
    }
    tensor.setDataType(OnnxMl.TensorProto.DataType.FLOAT.getNumber());
    tensor.setName("tensor");
    graph.addInitializer(tensor);

    // Add operations
    OnnxMl.NodeProto.Builder matmul = OnnxMl.NodeProto.newBuilder();
    matmul.setName("matmul-0");
    matmul.setOpType("MatMul");
    matmul.addInput("input");
    matmul.addInput("tensor");
    matmul.addOutput("output");
    graph.addNode(matmul);

    // Build model
    OnnxMl.ModelProto.Builder model = OnnxMl.ModelProto.newBuilder();
    model.setGraph(graph);
    model.setDocString("ORT matmul test");
    model.setModelVersion(0);
    model.setIrVersion(8);
    model.setDomain("ai.onnxruntime.test");
    model.addOpsetImport(OnnxMl.OperatorSetIdProto.newBuilder().setVersion(18).build());
    try (OutputStream os =
        Files.newOutputStream(Paths.get("src", "test", "resources", "java-external-matmul.onnx"))) {
      model.build().writeTo(os);
    }
  }

  public void generateMatMul() throws IOException {
    OnnxMl.GraphProto.Builder graph = OnnxMl.GraphProto.newBuilder();
    graph.setName("ort-test-matmul");

    // Add placeholders
    OnnxMl.ValueInfoProto.Builder input = OnnxMl.ValueInfoProto.newBuilder();
    input.setName("input");
    OnnxMl.TypeProto inputType =
        buildTensorTypeNode(
            new long[] {-1, 4},
            new String[] {"batch_size", null},
            OnnxMl.TensorProto.DataType.FLOAT);
    input.setType(inputType);
    graph.addInput(input);
    OnnxMl.ValueInfoProto.Builder output = OnnxMl.ValueInfoProto.newBuilder();
    output.setName("output");
    OnnxMl.TypeProto outputType =
        buildTensorTypeNode(
            new long[] {-1, 4},
            new String[] {"batch_size", null},
            OnnxMl.TensorProto.DataType.FLOAT);
    output.setType(outputType);
    graph.addOutput(output);

    // Add initializers
    OnnxMl.TensorProto.Builder tensor = OnnxMl.TensorProto.newBuilder();
    tensor.addDims(4);
    tensor.addDims(4);
    Float[] floats =
        new Float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f};
    tensor.addAllFloatData(Arrays.asList(floats));
    tensor.setDataType(OnnxMl.TensorProto.DataType.FLOAT.getNumber());
    tensor.setName("tensor");
    graph.addInitializer(tensor);

    // Add operations
    OnnxMl.NodeProto.Builder matmul = OnnxMl.NodeProto.newBuilder();
    matmul.setName("matmul-0");
    matmul.setOpType("MatMul");
    matmul.addInput("input");
    matmul.addInput("tensor");
    matmul.addOutput("output");
    graph.addNode(matmul);

    // Build model
    OnnxMl.ModelProto.Builder model = OnnxMl.ModelProto.newBuilder();
    model.setGraph(graph);
    model.setDocString("ORT matmul test");
    model.setModelVersion(0);
    model.setIrVersion(8);
    model.setDomain("ai.onnxruntime.test");
    model.addOpsetImport(OnnxMl.OperatorSetIdProto.newBuilder().setVersion(18).build());
    try (OutputStream os =
        Files.newOutputStream(Paths.get("src", "test", "resources", "java-matmul.onnx"))) {
      model.build().writeTo(os);
    }
  }

  public void generateThreeOutputMatmul() throws IOException {
    OnnxMl.GraphProto.Builder graph = OnnxMl.GraphProto.newBuilder();
    graph.setName("ort-test-three-matmul");

    // Add placeholders
    OnnxMl.ValueInfoProto.Builder input = OnnxMl.ValueInfoProto.newBuilder();
    input.setName("input");
    OnnxMl.TypeProto inputType =
        buildTensorTypeNode(
            new long[] {-1, 4},
            new String[] {"batch_size", null},
            OnnxMl.TensorProto.DataType.FLOAT);
    input.setType(inputType);
    graph.addInput(input);
    OnnxMl.ValueInfoProto.Builder outputA = OnnxMl.ValueInfoProto.newBuilder();
    outputA.setName("output-0");
    OnnxMl.TypeProto outputType =
        buildTensorTypeNode(
            new long[] {-1, 4},
            new String[] {"batch_size", null},
            OnnxMl.TensorProto.DataType.FLOAT);
    outputA.setType(outputType);
    graph.addOutput(outputA);
    OnnxMl.ValueInfoProto.Builder outputB = OnnxMl.ValueInfoProto.newBuilder();
    outputB.setName("output-1");
    outputB.setType(outputType);
    graph.addOutput(outputB);
    OnnxMl.ValueInfoProto.Builder outputC = OnnxMl.ValueInfoProto.newBuilder();
    outputC.setName("output-2");
    outputC.setType(outputType);
    graph.addOutput(outputC);

    // Add initializers
    OnnxMl.TensorProto.Builder tensor = OnnxMl.TensorProto.newBuilder();
    tensor.addDims(4);
    tensor.addDims(4);
    Float[] floats =
        new Float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f};
    tensor.addAllFloatData(Arrays.asList(floats));
    tensor.setDataType(OnnxMl.TensorProto.DataType.FLOAT.getNumber());
    tensor.setName("tensor");
    graph.addInitializer(tensor);
    OnnxMl.TensorProto.Builder addInit = OnnxMl.TensorProto.newBuilder();
    addInit.addDims(4);
    Float[] addFloats = new Float[] {1f, 2f, 3f, 4f};
    addInit.addAllFloatData(Arrays.asList(addFloats));
    addInit.setDataType(OnnxMl.TensorProto.DataType.FLOAT.getNumber());
    addInit.setName("add-init");
    graph.addInitializer(addInit);

    // Add operations
    OnnxMl.NodeProto.Builder matmul = OnnxMl.NodeProto.newBuilder();
    matmul.setName("matmul-0");
    matmul.setOpType("MatMul");
    matmul.addInput("input");
    matmul.addInput("tensor");
    matmul.addOutput("matmul-output");
    graph.addNode(matmul);

    OnnxMl.NodeProto.Builder id = OnnxMl.NodeProto.newBuilder();
    id.setName("id-1");
    id.setOpType("Identity");
    id.addInput("matmul-output");
    id.addOutput("output-0");
    graph.addNode(id);

    OnnxMl.NodeProto.Builder add = OnnxMl.NodeProto.newBuilder();
    add.setName("add-2");
    add.setOpType("Add");
    add.addInput("matmul-output");
    add.addInput("add-init");
    add.addOutput("output-1");
    graph.addNode(add);

    OnnxMl.NodeProto.Builder log = OnnxMl.NodeProto.newBuilder();
    log.setName("log-3");
    log.setOpType("Log");
    log.addInput("matmul-output");
    log.addOutput("output-2");
    graph.addNode(log);

    // Build model
    OnnxMl.ModelProto.Builder model = OnnxMl.ModelProto.newBuilder();
    model.setGraph(graph);
    model.setDocString("ORT three output matmul test");
    model.setModelVersion(0);
    model.setIrVersion(8);
    model.setDomain("ai.onnxruntime.test");
    model.addOpsetImport(OnnxMl.OperatorSetIdProto.newBuilder().setVersion(18).build());
    try (OutputStream os =
        Files.newOutputStream(
            Paths.get("src", "test", "resources", "java-three-output-matmul.onnx"))) {
      model.build().writeTo(os);
    }
  }

  private static void genCast(
      String name,
      OnnxMl.TensorProto.DataType inputDataType,
      OnnxMl.TensorProto.DataType outputDataType)
      throws IOException {
    OnnxMl.GraphProto.Builder graph = OnnxMl.GraphProto.newBuilder();
    graph.setName("ort-test-" + name);

    // Add placeholders
    OnnxMl.ValueInfoProto.Builder input = OnnxMl.ValueInfoProto.newBuilder();
    input.setName("input");
    OnnxMl.TypeProto inputType =
        buildTensorTypeNode(new long[] {-1, 5}, new String[] {"batch_size", null}, inputDataType);
    input.setType(inputType);
    graph.addInput(input);
    OnnxMl.ValueInfoProto.Builder output = OnnxMl.ValueInfoProto.newBuilder();
    output.setName("output");
    OnnxMl.TypeProto outputType =
        buildTensorTypeNode(new long[] {-1, 5}, new String[] {"batch_size", null}, outputDataType);
    output.setType(outputType);
    graph.addOutput(output);

    // Add operations
    OnnxMl.NodeProto.Builder cast = OnnxMl.NodeProto.newBuilder();
    cast.setName("cast-0");
    cast.setOpType("Cast");
    cast.addInput("input");
    cast.addOutput("output");
    cast.addAttribute(
        OnnxMl.AttributeProto.newBuilder()
            .setName("to")
            .setType(OnnxMl.AttributeProto.AttributeType.INT)
            .setI(outputDataType.getNumber())
            .build());
    graph.addNode(cast);

    // Build model
    OnnxMl.ModelProto.Builder model = OnnxMl.ModelProto.newBuilder();
    model.setGraph(graph);
    model.setDocString("ORT " + name + " test");
    model.setModelVersion(0);
    model.setIrVersion(8);
    model.setDomain("ai.onnxruntime.test");
    model.addOpsetImport(OnnxMl.OperatorSetIdProto.newBuilder().setVersion(18).build());
    try (OutputStream os =
        Files.newOutputStream(
            Paths.get(
                "..", "..", "..", "java", "src", "test", "resources", "java-" + name + ".onnx"))) {
      model.build().writeTo(os);
    }
  }

  public void generateFp16Fp32Cast() throws IOException {
    genCast("fp16-to-fp32", OnnxMl.TensorProto.DataType.FLOAT16, OnnxMl.TensorProto.DataType.FLOAT);
  }

  public void generateFp32Fp16Cast() throws IOException {
    genCast("fp32-to-fp16", OnnxMl.TensorProto.DataType.FLOAT, OnnxMl.TensorProto.DataType.FLOAT16);
  }

  public void generateBf16Fp32Cast() throws IOException {
    genCast(
        "bf16-to-fp32", OnnxMl.TensorProto.DataType.BFLOAT16, OnnxMl.TensorProto.DataType.FLOAT);
  }

  public void generateFp32Bf16Cast() throws IOException {
    genCast(
        "fp32-to-bf16", OnnxMl.TensorProto.DataType.FLOAT, OnnxMl.TensorProto.DataType.BFLOAT16);
  }
}
