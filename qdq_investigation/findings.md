# CPU Performance Feature Crew Findings 

## 1. MatMulNBits 

### A. DQ -> MatMul fusion Rules 

#### Findings: 
- Currently, only 4-bit block-quantized models are fused;  
- there are no fusion rules for 2-bit, 8-bit. 
- Fp16 models add cast ops around MatMul which block the MatMulNBits fusion 

#### Recommendation: 
- Matmulnbits already supports 2-bit and 4-bit operations. Update the fusion rules to enable support for fusion in 2-bit and 8-bit quantized models.  
= Enable MatMulNBits fusion for fp16 models. 

 
### B. Arm 8 bit MNB kernel fp16 

#### Findings: 
- 8 bit for fp16 scales does not implement accuracy_level=4 kernel 

#### Recommendation:  
- Implement kernel if fp16 cpu models are a priority 

 
### C. Arm:  KleidiAI support for 4 bit asymmetric quantized model

####  Findings:
- KleidiAI's optimized packed kernel (SDOT/I8MM, tiled) is disabled for asymmetric. Asym falls back to generic NEON CompInt8 (~30–46% penalty for ≥1.5b  param models). 
- Fix requires KleidiAI to support per-block zero-points in its `qsi4c32p` packing format (currently hardcodes `rhs_zero_point = 8`). 

#### Recommendation:  
- This is not a top priority. Additional investigation is needed to determine the necessary effort and anticipated performance improvements if support is added. 

## 2. Dequantize Linear (Per-channel, Blockwise) 

### A. Common 

####  Findings 
- 8 bits is faster than 4 bits in general 
- There are separate kernels for 8 bits and sub-byte data types 
- Sub-byte requires nibble extraction leading to overheads 
- Multithreaded/SIMD is not implemented 
  - Per-channel has it for 8bit per-channel but it is not enabled by default 
- DQ kernel does all intermediate compute in fp32 by casting to/from fp16.  

#### Suggestions: 
- Implement Multithreaded/SIMD 
  - Mlas has dequantize blockwise kernels 
- Check if fp16 casts can be avoided. This might not give gains since CPU doesn’t have native fp16 data type and all fp16 compute might do implicit casts to/from fp32 anyways. But it is worth looking at to be sure. 
 

### B. Blockwise Quantization 

#### Findings 
- 4 bit fp32 DQ->MatMul get fused to MatMulNBits 
- These can be expanded to include 2 and 8 bits, so priority of standalone DQ improvements depends on whether fusion to MatMulNBits is enough. 

#### Suggestions: 
- We should also consider the weight layout. MatMulNBits has NxK layout while DQ has KxN layout. For standalone blockwise dequantization, KxN layout is probably better. 

 

### C. Per-channel Quantization 

#### Findings 
- Per-channel slower than blockwise even though it can be faster ideally 
- Incorrect graph transformation gets applied for signed, symmetric models 

####  Suggestions: 
- We should also consider weight layout. DQ has KxN but NxK is probably better for per-channel quantization. 
- Fix broken graph transformation for signed, symmetric models 

 

## 3. FP16 Model 

### Findings 
- FP16 support is available for DQ and Q operators and contrib operators such as MatMulNBits, GQA, etc. 
- However, basic operators such as MatMul, Mul, Add, etc do not have fp16 support. 
- This leads addition of cast operators in the graph as well as blocking of some operator fusion rules 

### Suggestions 
- Consider extending compute operator support for fp16 
- This requires some investigation into the cost of the cast operators since we might still incur cast cost implicitly in the operator implementation if there are no native fp16 kernels. 

 

## 4. Full QDQ Model (per-tensor, activation + weight) 

### A. uint16 activation, uint8 weight 

#### Findings 
- Onnx spec doesn’t have QLinear operators for this combination. It only supports uint8/int8 weight and activation. 
- This combination is commonly used in non-llm models such as stable diffusion and ASG’s perception shell models 
- Without the QLinear operator, there is no operator fusion so the DQ and Q nodes add additional compute overhead to the base operators that still compute in float precision 
- ASG is proposing adding a set of operators that support this combination. 

#### Suggestion 
- Work with ASG to implement QLinear operators for 16-bit activation and 8-bit weights 
- Update ONNX spec as needed. Otherwise, would need to use a contrib op for this 

 
### B. CPU EP’s handling of QDQ models 

#### Findings 
- We have many fully QDQ models that we would like to use as shared models across different EPs. 
- However, fully QDQ models are meant to target EPs such as QNN EP that have integer kernels which fuse the DQ->Op->Q node groups into IntegerOps for native quantized computation 
- CPU EP doesn’t have a limited set of efficient integer operators that only support a limited set of datatypes. 
- The QDQ nodes also block operator fusions for patterns such as MatMulNBits, layer normalizations, etc. 
- These QDQ nodes add computational overhead since they don’t get fused. 

#### Suggestions 
- CPU EP could handle the QDQ models specially 
 - Keep the DQs on the weights 
 - Only keep activation QDQ nodes if they can be fused with the operator. 
- There are some open questions about mismatch between the QDQ eleminated model and the original model 
    - Output mismatch is expected 
    - But overall quality would probably improve unless the model weights were trained to be activation quantization aware. For standard static quantization of activations, this is not a concern. 

## 5. Vision models (float)

### Findings 
- ARM64 can do with optimizations to its NCHWc Conv kernel suite 
- Most Conv models are textbook candidates for NCHWc data-layout based Conv operation acceleration on CPUs but it needs to be backed by optimized and tuned vectorized NCHWc kernels which ARM64 is currently lacking 
- On some platforms, some models (based on various Conv parameters) show slowdowns whilst using the NCHWc data layout (when compared with the default NCHW data layout). Expose a session option for users to opt-out of NCHWc data layout (current turned on by default with the default graph optimization) and cook this logic into OLIVE recipes while presenting the most optimal setting for user workload for their target platform 
- The NCHWc layout transformer doesn’t handle elementwise ops properly (missing coverage) (i.e.) it inserts Reorder nodes (transposes) before and after some elementwise ops when the ops are inherently data layout agnostic (i.e.) we incur transpose costs unnecessarily. These show up in cheap Conv models. 

### Suggestions: 
- Look into compute optimal Conv algorithms (e.g.) Winograd that does well for “vanilla” 3x3 Conv operations. Many other frameworks support this implementation -  https://github.com/microsoft/onnxruntime/issues/12834  
- Explore more activation fusions into the MLAS Conv micro-kernels. Currently, only “ReLU” activation is fused into the assembly micro-kernels. Explore adding more commonly occurring ones (Gelu, SIlu, etc.) but this comes with considerable dev cost as fusions come with more register budget requirement which may lead to re-work of other parts of these micro-kernels 
- For customers who are reporting 2x perf slowdowns of CPU EP when compared with the OpenVINO EP, it is important to understand if we are comparing apple-to-apple. OpenVINO has certain performance hints which may mean that they drop to mixed precision mode whereas the CPU EP honors the provided model as is 
- Explore the threading model in the NCHWc Conv kernel suite (especially for unbatched ungrouped Conv operations (B=1 G=1)) 