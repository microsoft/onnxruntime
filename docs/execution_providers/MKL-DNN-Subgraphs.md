# Subgraph Optimization

MKL-DNN uses blocked layout (example: nhwc with channels blocked by 16 – nChw16c) to take advantage of vector operations using AVX512.  To get best performance, we avoid reorders (example. Nchw16c to nchw) and propagate blocked layout to next primitive. 

Subgraph optimization achieves this in the following steps.
1.	Parses ONNX Runtime graph and creates an Internal Representation of subgraph..
2.	Subgraph Operator (MklDnnFunKernel) iterates through MKL-DNN nodes and creates a vector MKL-DNN Kernels
3.	Compute Function of MklDnnFunKernel iterates and binds data to MKL-DNN primitives in the vector and submits vector for execution.


## Subgraph (IR) Internal Representation
MklDnnExecutionProvicer::GetCapability() parses ONNX model graph and creates IR (Internal Representation) of subgraphs of MKL-DNN operators.
Each subgraph contains a vector MklDnnNodes, inputs, outputs and attributes for all its MklDnnNodes. There can be attributes of same name. So, we prefix attribute names with Node name and its index. 
Unique id for subgraph is set as an attribute. 

MklDnnNode has an index to its inputs and outputs and pointer to its parent nodes. MklDnnNode directly reads blocked memory from its parent to avoid data reordering.

<p align="left"><img src="images/mkl-dnn_node.png" /></p>


## Subgraph Classes
Primitive like MklDnnConv, MklDnnPool, etc are derived from MklDnnKernel base class.

The following UML diagram captures Subgraph classes.

<p align="left"><img src="images/mkl-dnn_subgraph.png" /></p>


## Subgraph Execution

MklDnnExecutionProvicer::Compute() function creates MklDnnFuncKernel and call it’s Compute Function.


MklDnnFuncKernel::Compute function creates SubgraphPrimitve pool and add the object to a map.

SubgraphPrimitve constructor calls the following member functions
```
SubgraphPrimitve::CreatePrimitives()
    for (auto& mklnode : mklnodes) {
      if (mklnode.name == "Conv") {
        kernel.reset(new MklDnnConv());
        kernels.push_back(kernel);
      } else if (mklnode.name == "BatchNormalization-Relu") {
        kernel.reset(new MklDnnBatchNorm());
        context_.kernels.push_back(kernel);
      } else if (mklnode.name == "MaxPool") {
        kernel.reset(new MklDnnPool());
        context_.kernels.push_back(kernel);
      } 
      .
      .
      .
```      
In CreatePrimitives method, we iterate MklDnnNodes and creates MklDnnKernel objects and add MKL-DNN primitive to a vector. It also reads attributes. This is done only once, at first iteration.

``` 
SubgraphPrimitve::Compute()
   for (auto& kernel : kernels) {
      kernel->Bind(input_tensors, output_tensors);
    }
    stream->submit(net);
```

In SubgraphPrimitve::Compute() method, we iterate thru MklDnn Kernels and bind input data. Then we submit the vector of Primitives to MKL-DNN stream.

