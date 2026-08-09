/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

/** The info for an input or output node from an ONNX model. */
public class NodeInfo {

  private final ValueInfo info;

  private final String name;

  /**
   * Creates a node info object from the supplied name and value info.
   *
   * <p>Called from native code.
   *
   * @param name The name of the node.
   * @param info The ValueInfo for this node.
   */
  public NodeInfo(String name, ValueInfo info) {
    this.name = name;
    this.info = info;
  }

  /**
   * The name of the node.
   *
   * @return The name.
   */
  public String getName() {
    return name;
  }

  /**
   * The type and shape information of this node.
   *
   * <p>WARNING: {@link MapInfo} and {@link SequenceInfo} instances returned by this type will have
   * insufficient information in them as it's not available from the model without an example
   * output.
   *
   * @return The information of the value.
   */
  public ValueInfo getInfo() {
    return info;
  }

  @Override
  public String toString() {
    return "NodeInfo(name=" + name + ",info=" + info.toString() + ")";
  }
}
