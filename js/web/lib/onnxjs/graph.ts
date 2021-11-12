// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {onnx} from 'onnx-proto';

import {Attribute} from './attribute';
import {onnxruntime} from './ort-schema/ort-generated';
import ortFbs = onnxruntime.experimental.fbs;
import {Tensor} from './tensor';
import {LongUtil, ProtoUtil, MIN_CLIP, MAX_CLIP} from './util';

export declare namespace Graph {
  export interface Shape {
    readonly dims: readonly number[];
  }
  export interface ValueType {
    readonly tensorType: Tensor.DataType;
    readonly shape: Shape;
  }
  export interface Value {
    // the tensor data. empty for non-initialized inputs
    readonly tensor?: Tensor;

    // index to the Node where the value comes from. -1 for initializer.
    readonly from: number;

    // indices to the Nodes where the values go to.
    readonly to: readonly number[];

    // value type specification. empty for non-input values.
    readonly type?: ValueType;
  }
  export interface Node {
    // name of the node
    readonly name: string;

    // the operator type
    readonly opType: string;

    // indices to the Values where the inputs come from.
    readonly inputs: readonly number[];

    // indices to the Values where the outpus go to.
    readonly outputs: readonly number[];

    // the attributes that used by the operator
    readonly attributes: Attribute;
  }

  /**
   * a Transformer is an instance that allows all possible transformation operations that applied to a graph
   */
  export interface Transformer {
    removeAllIdentityNodes(): void;
    removeAllDropoutNodes(): void;
    fuseConvActivationNodes(): void;
    // TODO: add generic functions to manipulate the graph
  }

  // an initializer can use transformer to transform the graph
  export interface Initializer {
    transformGraph(transformer: Transformer): void;
  }
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
export interface Graph {
  getInputIndices(): readonly number[];
  getInputNames(): readonly string[];
  getOutputIndices(): readonly number[];
  getOutputNames(): readonly string[];
  getValues(): readonly Graph.Value[];
  getNodes(): readonly Graph.Node[];
}

// eslint-disable-next-line @typescript-eslint/naming-convention, @typescript-eslint/no-redeclare
export const Graph = {
  /**
   * construct a graph from a graph protobuf type
   */
  from: (graphProto: onnx.IGraphProto|ortFbs.Graph, initializer?: Graph.Initializer) =>
      new GraphImpl(graphProto, initializer),
};

class Value implements Graph.Value {
  constructor(valueInfo?: onnx.IValueInfoProto) {
    this._from = undefined;
    this._to = [];
    this.tensor = undefined;
    this.type = undefined;

    if (valueInfo) {
      this.type = ProtoUtil.tensorValueTypeFromProto(valueInfo.type!.tensorType!);
    }
  }

  _from?: number;  // -1 represent from initializer
  get from() {
    return this._from!;
  }
  _to: number[];
  get to() {
    return this._to;
  }
  type?: Graph.ValueType;
  tensor?: Tensor;
}

class Node implements Graph.Node {
  constructor(_nodeProto: onnx.INodeProto|ortFbs.Node, name?: string) {
    if (_nodeProto instanceof onnx.NodeProto) {
      this.name = _nodeProto.name;
      this.opType = _nodeProto.opType;
      this.attributes = new Attribute(_nodeProto.attribute);
    } else if (_nodeProto instanceof ortFbs.Node) {
      this.name = name ?? _nodeProto.name()!;
      this.opType = _nodeProto.opType()!;
      this.attributes = new Attribute(ProtoUtil.tensorAttributesFromORTFormat(_nodeProto));
    }

    this.inputs = [];
    this.outputs = [];
    this.executeNode = true;
  }

  name: string;
  opType: string;
  inputs: number[];
  outputs: number[];
  attributes: Attribute;
  executeNode: boolean;
}

class GraphImpl implements Graph, Graph.Transformer {
  private _allData: Value[];

  private _allInputIndices: number[];
  private _allInputNames: string[];

  private _allOutputIndices: number[];
  private _allOutputNames: string[];

  private _nodes: Node[];

  constructor(graph: onnx.IGraphProto|ortFbs.Graph, graphInitializer?: Graph.Initializer) {
    if (!graph) {
      throw new TypeError('graph is empty');
    }

    // build the graph - will throw exceptions if something fatal is detected
    this.buildGraph(graph);

    // execute any transformation logic for the graph (if applicable)
    this.transformGraph(graphInitializer);

    // check for cycles and other inconsistencies - will throw exceptions if something fatal is detected
    this.checkIsAcyclic();
  }

  getInputIndices(): readonly number[] {
    return this._allInputIndices;
  }

  getInputNames(): readonly string[] {
    return this._allInputNames;
  }

  getOutputIndices(): readonly number[] {
    return this._allOutputIndices;
  }

  getOutputNames(): readonly string[] {
    return this._allOutputNames;
  }

  getValues(): readonly Graph.Value[] {
    return this._allData;
  }

  getNodes(): readonly Graph.Node[] {
    return this._nodes;
  }

  private buildGraph(graph: onnx.IGraphProto|ortFbs.Graph) {
    // build the graph - will throw exceptions if something fatal is detected
    if (graph instanceof onnx.GraphProto) {
      this.buildGraphFromOnnxFormat(graph);
    } else if (graph instanceof ortFbs.Graph) {
      this.buildGraphFromOrtFormat(graph);
    } else {
      throw new TypeError('Graph type is not supported.');
    }
  }
  private buildGraphFromOnnxFormat(graph: onnx.IGraphProto) {
    const dataIndices = new Map<string, number>();
    this._allData = [];

    this._allInputIndices = [];
    this._allInputNames = [];

    this._allOutputIndices = [];
    this._allOutputNames = [];

    this._nodes = [];

    const nodesIndices = new Map<string, number>();

    // scan all inputs
    if (!graph.input) {
      throw new Error('missing information in graph: input');
    }
    const inputValueNames = [];
    for (const i of graph.input) {
      if (dataIndices.has(i.name!)) {
        throw new Error(`duplicated input name: ${i.name}`);
      }
      const currentIndex = this._allData.push(new Value(i)) - 1;
      dataIndices.set(i.name!, currentIndex);
      inputValueNames.push(i.name!);
    }

    // scan all initializers
    if (!graph.initializer) {
      throw new Error('missing information in graph: initializer');
    }
    for (const i of graph.initializer) {
      let index = dataIndices.get(i.name!);
      if (index === undefined) {
        const value = new Value();
        value.type = {
          shape: {dims: ProtoUtil.tensorDimsFromProto(i.dims!)},
          tensorType: ProtoUtil.tensorDataTypeFromProto(i.dataType!)
        };
        index = this._allData.push(value) - 1;
        dataIndices.set(i.name!, index);
      }
      this._allData[index]._from = -1;
      this._allData[index].tensor = Tensor.fromProto(i);
    }

    // filter out input indices
    for (let i = 0; i < this._allData.length; i++) {
      if (!this._allData[i].tensor) {
        this._allInputIndices.push(i);
        this._allInputNames.push(inputValueNames[i]);
      }
    }

    // scan all outputs
    if (!graph.output) {
      throw new Error('missing information in graph: output');
    }
    for (const i of graph.output) {
      if (dataIndices.has(i.name!)) {
        throw new Error(`duplicated output name: ${i.name}`);
      }
      const currentIndex = this._allData.push(new Value(i)) - 1;
      dataIndices.set(i.name!, currentIndex);
      this._allOutputIndices.push(currentIndex);
      this._allOutputNames.push(i.name!);
    }

    // scan all nodes
    if (!graph.node) {
      throw new Error('missing information in graph: node');
    }
    for (const nodeProto of graph.node) {
      if (!nodeProto.name) {
        // assign a name to the node if it doesn't have one
        for (let pick = 0;; pick++) {
          const name = `unnamed_${nodeProto.opType}_${pick}`;
          if (!nodesIndices.has(name)) {
            nodeProto.name = name;
            break;
          }
        }
      }

      if (nodesIndices.has(nodeProto.name)) {
        throw new Error(`duplicated node name: ${nodeProto.name}`);
      }
      const currentIndex = this._nodes.push(new Node(nodeProto)) - 1;
      nodesIndices.set(nodeProto.name, currentIndex);
    }

    // scan node's outputs
    for (let i = 0; i < this._nodes.length; i++) {
      const node = this._nodes[i];
      const nodeProto = graph.node[i];
      if (!nodeProto.output) {
        throw new Error(`missing output for node: ${nodeProto.name}`);
      }
      for (const output of nodeProto.output) {
        let dataIndex = dataIndices.get(output);
        if (typeof dataIndex === 'undefined') {
          dataIndex = this._allData.push(new Value()) - 1;
          dataIndices.set(output, dataIndex);
        }
        node.outputs.push(dataIndex);

        if (this._allData[dataIndex]._from !== undefined) {
          throw new Error(`multiple nodes output to one data value: ${dataIndex}`);
        }
        this._allData[dataIndex]._from = i;

        // for the 'Constant' operator, just create a new edge in the graph corresponding to the 'output' of the
        // operator and ignore the node from the graph
        if (nodeProto.opType === 'Constant') {
          if (!nodeProto.attribute || nodeProto.attribute.length !== 1 || !nodeProto.attribute[0].t) {
            throw new Error('missing attributes or missing tensor value in attributes for this Constant operator');
          }
          if (!nodeProto.output || nodeProto.output.length !== 1) {
            throw new Error('missing output or incorrect number of outputs for this Constant operator');
          }
          node.outputs.pop();
          node.executeNode = false;

          this._allData[dataIndex]._from = -1;
          this._allData[dataIndex].tensor = Tensor.fromProto(nodeProto.attribute[0].t);
        }
      }
    }

    // scan node's inputs
    for (let i = 0; i < this._nodes.length; i++) {
      const node = this._nodes[i];
      const nodeProto = graph.node[i];

      if (!nodeProto.input) {
        throw new Error(`missing input for node: ${nodeProto.name}`);
      }
      for (const input of nodeProto.input) {
        const dataIndex = dataIndices.get(input);
        if (typeof dataIndex === 'undefined') {
          throw new Error(`unrecognized input '${input}' for node: ${nodeProto.name}`);
        }
        node.inputs.push(dataIndex);

        this._allData[dataIndex]._to.push(i);
      }
    }

    return true;
  }

  private buildGraphFromOrtFormat(graph: ortFbs.Graph) {
    const dataIndices = new Map<string, number>();
    this._allData = [];

    this._allInputIndices = [];
    this._allInputNames = [];

    this._allOutputIndices = [];
    this._allOutputNames = [];

    this._nodes = [];

    const nodesIndices = new Map<string, number>();

    // scan all inputs
    const inputValueNames = [];
    for (let i = 0; i < graph.inputsLength(); i++) {
      const inputName = graph.inputs(i);
      if (dataIndices.has(inputName)) {
        throw new Error(`duplicated input name: ${inputName}`);
      }
      // Find the input typeInfo from nodeargs
      for (let j = 0; j < graph.nodeArgsLength(); j++) {
        if (graph.nodeArgs(j)?.name() === inputName) {
          const value = new Value();
          const valueType = graph.nodeArgs(j)?.type()?.valueType();
          if (valueType !== ortFbs.TypeInfoValue.tensor_type) {
            throw new Error('Unexpected value type for the nodeArg.');
          }
          const valueInfo = graph.nodeArgs(j)!.type()!.value(new ortFbs.TensorTypeAndShape())!;
          const type = ProtoUtil.tensorDataTypeFromProto(valueInfo.elemType());
          const shape = valueInfo.shape()!;
          const dims = [];
          for (let k = 0; k < shape.dimLength()!; k++) {
            dims.push(LongUtil.longToNumber(shape.dim(k)!.value()!.dimValue()!));
          }
          value.type = {shape: {dims}, tensorType: type};
          const currentIndex = this._allData.push(value) - 1;
          dataIndices.set(inputName, currentIndex);
          inputValueNames.push(inputName);
        }
      }
    }
    // check initializers
    for (let i = 0; i < graph.initializersLength(); i++) {
      const initializer = graph.initializers(i)!;
      let index = dataIndices.get(initializer.name()!);
      if (index === undefined) {
        const value = new Value();
        const dims = ProtoUtil.tensorDimsFromORTFormat(initializer);
        const type = ProtoUtil.tensorDataTypeFromProto(initializer.dataType());
        value.type = {shape: {dims}, tensorType: type};
        index = this._allData.push(value) - 1;
        dataIndices.set(initializer.name()!, index);
      }
      this._allData[index]._from = -1;
      this._allData[index].tensor = Tensor.fromOrtTensor(initializer);
    }

    // filter out input indices
    for (let i = 0; i < this._allData.length; i++) {
      if (!this._allData[i].tensor) {
        this._allInputIndices.push(i);
        this._allInputNames.push(inputValueNames[i]);
      }
    }

    // scan all outputs
    for (let i = 0; i < graph.outputsLength(); i++) {
      const outputName = graph.outputs(i);
      if (dataIndices.has(outputName)) {
        throw new Error(`duplicated output name: ${outputName}`);
      }
      const currentIndex = this._allData.push(new Value()) - 1;
      dataIndices.set(outputName, currentIndex);
      this._allOutputIndices.push(currentIndex);
      this._allOutputNames.push(outputName);
    }

    // scan all nodes
    if (!graph.nodes) {
      throw new Error('missing information in graph: node');
    }
    for (let i = 0; i < graph.nodesLength(); i++) {
      const nodeProto = graph.nodes(i);
      let name = nodeProto!.name();
      if (!name) {
        // assign a name to the node if it doesn't have one
        for (let pick = 0;; pick++) {
          name = `unnamed_${nodeProto!.opType()}_${pick}`;
          if (!nodesIndices.has(name)) {
            // an unique name is found. break.
            break;
          }
        }
      }

      if (nodesIndices.has(name)) {
        throw new Error(`duplicated node name: ${name}`);
      }
      const currentIndex = this._nodes.push(new Node(nodeProto!, name)) - 1;
      nodesIndices.set(name, currentIndex);
    }

    // scan node's outputs
    for (let i = 0; i < this._nodes.length; i++) {
      const node = this._nodes[i];
      const nodeProto = graph.nodes(i);
      if (nodeProto == null) {
        throw new Error(`No node exists at index ${i}`);
      }
      if (nodeProto?.outputsLength() === 0) {
        throw new Error(`missing output for node: ${nodeProto.name}`);
      }
      for (let j = 0; j < nodeProto?.outputsLength(); j++) {
        const output = nodeProto?.outputs(j);
        let dataIndex = dataIndices.get(output);
        if (typeof dataIndex === 'undefined') {
          dataIndex = this._allData.push(new Value()) - 1;
          dataIndices.set(output, dataIndex);
        }
        node.outputs.push(dataIndex);

        if (this._allData[dataIndex]._from !== undefined) {
          throw new Error(`multiple nodes output to one data value: ${dataIndex}`);
        }
        this._allData[dataIndex]._from = i;

        // for the 'Constant' operator, just create a new edge in the graph corresponding to the 'output' of the
        // operator and ignore the node from the graph
        if (nodeProto.opType() === 'Constant') {
          if (nodeProto.attributesLength() !== 1 || !nodeProto.attributes(0)!.t()) {
            throw new Error('missing attributes or missing tensor value in attributes for this Constant operator');
          }
          if (nodeProto.outputsLength() !== 1) {
            throw new Error('missing output or incorrect number of outputs for this Constant operator');
          }
          node.outputs.pop();
          node.executeNode = false;

          this._allData[dataIndex]._from = -1;
          this._allData[dataIndex].tensor = Tensor.fromOrtTensor(nodeProto.attributes(0)!.t()!);
        }
      }
    }

    // scan node's inputs
    for (let i = 0; i < this._nodes.length; i++) {
      const node = this._nodes[i];
      const nodeProto = graph.nodes(i)!;

      if (nodeProto.inputsLength() === 0) {
        throw new Error(`missing input for node: ${nodeProto.name}`);
      }
      for (let j = 0; j < nodeProto.inputsLength()!; j++) {
        const input = nodeProto.inputs(j)!;
        const dataIndex = dataIndices.get(input);
        if (typeof dataIndex === 'undefined') {
          throw new Error(`unrecognized input '${input}' for node: ${nodeProto!.name()}`);
        }
        node.inputs.push(dataIndex);

        this._allData[dataIndex]._to.push(i);
      }
    }
  }

  private checkIsAcyclic() {
    // go through the graph and check for cycles or other fatal inconsistencies
    const starters: Set<number> = new Set<number>();
    this._allInputIndices.forEach(i => {
      const data = this._allData[i];
      data._to.forEach(j => {
        starters.add(j);
      });
    });

    // Iterative DFS to check for cycles
    const nodesStack = Array.from(starters);
    const nodesState = new Array<string>(this._nodes.length).fill('white');

    while (nodesStack.length > 0) {
      const nodeIndex = nodesStack.pop()!;
      // this node has now been processed completely. Mark this node 'black' to denote this.
      if (nodesState[nodeIndex] === 'gray') {
        nodesState[nodeIndex] = 'black';
      } else {
        // this node is under processing stage. mark this node 'gray' to denote this.
        nodesStack.push(nodeIndex);
        nodesState[nodeIndex] = 'gray';

        this._nodes[nodeIndex].outputs.forEach((outgoingEdgeIndex) => {
          const data = this._allData[outgoingEdgeIndex];
          if (typeof data.tensor !== 'undefined') {
            throw new Error('node outputs should not be initialized');
          }
          if (data._from !== nodeIndex) {
            throw new Error('from property of the Value object doesn\'t match index of Node being processed');
          }
          data._to.forEach((downstreamNodeIndex) => {
            // back edge found - cyclic
            if (nodesState[downstreamNodeIndex] === 'gray') {
              throw new Error('model graph is cyclic');
            }
            // tree edge found - continue processing by adding it to stack
            else if (nodesState[downstreamNodeIndex] === 'white') {
              nodesStack.push(downstreamNodeIndex);
            }
          });
        });
      }
    }
  }

  private transformGraph(graphInitializer?: Graph.Initializer): void {
    // apply common transform
    this.removeAllIdentityNodes();
    this.removeAllDropoutNodes();
    this.fuseConvActivationNodes();
    // apply initializer specific transform
    if (graphInitializer) {
      graphInitializer.transformGraph(this);
    }

    // finalize graph
    this.finalizeGraph();
  }

  /**
   * finalize the graph.
   *
   * this function should be called after all the transformation completed.
   * this function removes all unnecessary nodes and values from the graph
   */
  finalizeGraph() {
    let offset = 0;
    // delete all nodes that are not being executed
    for (let i = 0; i < this._nodes.length; i++) {
      if (!this._nodes[i].executeNode) {
        // delete this node and shift all subsequent nodes up
        offset++;
        // delete all output values
        this._nodes[i].outputs.forEach(ind => {
          this._allData[ind]._from = -2;
        });
        this._nodes.splice(i, 1);
        i--;
        continue;
      }
      if (offset > 0) {
        // update the value table
        this._nodes[i].inputs.forEach(value => {
          const ind = this._allData[value]._to.indexOf(i + offset);
          if (ind !== -1) {
            this._allData[value]._to[ind] = i;
          }
        });
        this._nodes[i].outputs.forEach(value => {
          if (this._allData[value]._from && this._allData[value]._from! === i + offset) {
            this._allData[value]._from! = i;
          }
        });
      }
    }
    offset = 0;
    // delete all values that are not being referenced
    for (let i = 0; i < this._allData.length; i++) {
      // if current value is neither linked to next node, nor an output value, remove it.
      if (this._allData[i].from === -2 && this._allOutputIndices.indexOf(i + offset) === -1) {
        offset++;
        this._allData.splice(i, 1);
        i--;
        continue;
      }
      if (offset > 0) {
        let ind = -1;
        // if current value is neither an input value nor an initializer, find the node it's
        // coming from and update the corresponding node output
        if (this._allData[i].from !== undefined && this._allData[i].from !== -1) {
          ind = this._nodes[this._allData[i].from].outputs.indexOf(i + offset);
          if (ind !== -1) {
            this._nodes[this._allData[i].from].outputs[ind] = i;
          }
        } else {
          // if current value is an input value, update its reference in inputIndices
          ind = this._allInputIndices.indexOf(i + offset);
          if (ind !== -1) {
            this._allInputIndices[ind] = i;
          }
        }

        // find the node that the current value is linking to and update its input reference
        this._allData[i].to.forEach(node => {
          ind = this._nodes[node].inputs.indexOf(i + offset);
          if (ind !== -1) {
            this._nodes[node].inputs[ind] = i;
          }
        });
        if (this._allData[i].to.length === 0) {
          // if current value is a graph output, update its reference in outputIndices
          ind = this._allOutputIndices.indexOf(i + offset);
          if (ind !== -1) {
            this._allOutputIndices[ind] = i;
          }
        }
      }
    }
  }

  /**
   * Delete the specifed node. Assume the node has one incoming input and the first output connected to other nodes.
   * An input validation must be done before calling this function.
   * @param nodeIndex The index of node to be deleted
   */
  private deleteNode(nodeIndex: number) {
    const node = this._nodes[nodeIndex];
    if (node.outputs.length > 1) {
      for (let i = 1; i < node.outputs.length; i++) {
        if (this._allData[node.outputs[i]].to.length > 0) {
          throw new Error('Node deletion with more than one output connected to other nodes is not supported. ');
        }
      }
    }

    // this node wil not be executed
    node.executeNode = false;
    const inputValueIndex = node.inputs[0];
    const outputValueIndex = node.outputs[0];
    const nodesConsumingOutput = this._allData[outputValueIndex].to;

    // remove this node from the to property of the input Value
    const delIndex = this._allData[inputValueIndex].to.indexOf(nodeIndex);
    // should not happen
    if (delIndex === -1) {
      throw new Error('The Value object doesn\'t have the current Node in it\'s \'to\' property ');
    }
    this._allData[inputValueIndex].to.splice(delIndex, 1);

    // clear node indices consuming this output Value
    this._allData[outputValueIndex]._to = [];

    // if the output of this node is a graph output, adjust the index appropriately
    const index = this._allOutputIndices.indexOf(outputValueIndex);
    if (index !== -1) {
      this._allOutputIndices[index] = inputValueIndex;
    }

    // override the inputs for nodes consuming this node's output with the input to this node
    if (nodesConsumingOutput && nodesConsumingOutput.length > 0) {
      for (const nodeIndex of nodesConsumingOutput) {
        const replaceIndex = this._nodes[nodeIndex].inputs.indexOf(outputValueIndex);
        // should not happen
        if (replaceIndex === -1) {
          throw new Error('The Node object doesn\'t have the output Value in it\'s \'inputs\' property ');
        }
        this._nodes[nodeIndex].inputs[replaceIndex] = inputValueIndex;
        this._allData[inputValueIndex].to.push(nodeIndex);
      }
    }
  }

  removeAllDropoutNodes() {
    let nodeIndex = 0;
    for (const node of this._nodes) {
      // weed out 'Dropout' nodes so that no time is wasted in execution
      if (node.opType === 'Dropout') {
        // the node should have exactly 1 input and 1 or 2 outputs
        if (node.inputs.length !== 1) {
          throw new Error('Dropout nodes should only contain one input. ');
        }
        if (node.outputs.length !== 1 && node.outputs.length !== 2) {
          throw new Error('Dropout nodes should contain either 1 or 2 output(s)');
        }
        // the second output should not be referenced by any other node
        if (node.outputs.length === 2 && this._allData[node.outputs[1]]._to.length !== 0) {
          throw new Error('Dropout nodes\'s second output should not be referenced by other nodes');
        }
        this.deleteNode(nodeIndex);
      }
      nodeIndex++;
    }
  }

  removeAllIdentityNodes() {
    let nodeIndex = 0;
    for (const node of this._nodes) {
      // weed out 'Identity' nodes so that no time is wasted in execution
      if (node.opType === 'Identity') {
        this.deleteNode(nodeIndex);
      }
      nodeIndex++;
    }
  }

  isActivation(n: Node): boolean {
    switch (n.opType) {
      // TODO: add other activation methods
      case 'Relu':
      case 'Sigmoid':
      case 'Clip':
        return true;
      default:
        return false;
    }
  }

  fuseConvActivationNodes() {
    for (const node of this._nodes) {
      if (node.opType === 'Conv') {
        const next = this._allData[node.outputs[0]]._to;
        if (next.length === 1 && this.isActivation(this._nodes[next[0]])) {
          const child = this._nodes[next[0]];
          if (child.opType === 'Clip') {
            if (child.inputs.length === 1) {
              try {
                node.attributes.set(
                    'activation_params', 'floats',
                    [child.attributes.getFloat('min'), child.attributes.getFloat('max')]);
              } catch (e) {
                node.attributes.set('activation_params', 'floats', [MIN_CLIP, MAX_CLIP]);
              }
            } else if (
                child.inputs.length >= 3 && this._allData[child.inputs[1]].tensor !== undefined &&
                this._allData[child.inputs[2]].tensor !== undefined) {
              node.attributes.set('activation_params', 'floats', [
                this._allData[child.inputs[1]].tensor!.floatData[0], this._allData[child.inputs[2]].tensor!.floatData[0]
              ]);
            } else {
              // Skip fusion with clip node since clip min and clip max are not coming from initializer
              continue;
            }
          }
          node.attributes.set('activation', 'string', (child.opType));
          this.deleteNode(next[0]);
        }
      }
    }
  }
}
