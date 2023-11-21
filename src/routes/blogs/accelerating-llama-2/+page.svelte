<script>
	import Header from '../../components/header.svelte';
	import Footer from '../../components/footer.svelte';
	import figure1 from '../../../images/blogs/accelerating-llama-2/Figure1-LLaMA-2-7B-E2E-Throughput.png';
	import figure1b from '../../../images/blogs/accelerating-llama-2/Figure1-LLaMA-2-13B-E2E-Throughput.png';
	import figure2 from '../../../images/blogs/accelerating-llama-2/Figure2-LLaMA-2-7B-Prompt-Latency.png';
	import figure2b from '../../../images/blogs/accelerating-llama-2/Figure2-LLaMA-2-13B-Prompt-Latency.png';
	import figure3 from '../../../images/blogs/accelerating-llama-2/Figure3-LLaMA-2-7B-Tokens-Generated-Throughput.png';
	import figure3b from '../../../images/blogs/accelerating-llama-2/Figure3-LLaMA-2-13B-Tokens-Generated-Throughput.png';
	import figure4 from '../../../images/blogs/accelerating-llama-2/Figure4-LLaMA-2-70B-Model-Throughput.png';
	import figure5 from '../../../images/blogs/accelerating-llama-2/LLaMA-2OptimizationDiagram-5.png';
	import figure6 from '../../../images/blogs/accelerating-llama-2/LLaMAWindowsExportRotaryEmbeddingSubgraph-6.jpg';
	import figure7 from '../../../images/blogs/accelerating-llama-2/RotaryEmbeddingFunctionExample-7.png';
</script>

<svelte:head>
	<meta
		name="description"
		content="Explore how ONNX Runtime can propel your Llama2 variants for faster inference."
	/>
	<meta
		name="keywords"
		content="Accelerating LLaMA-2, ONNX Runtime, Inference, AI, Microsoft, Meta, Llama2, Performance Optimization, Multi-GPU Inference"
	/>
	<meta name="author" content="Kunal Vaishnavi, Parinita Rahi" />
	<meta name="date" content="2023-11-14" />
	<meta property="og:title" content="Accelerating LLaMA-2 Inference with ONNX Runtime" />
	<meta
		property="og:description"
		content="Explore how ONNX Runtime accelerates LLaMA-2 inference, achieving up to 3.8X faster performance for models ranging from 7B to 70B parameters. Learn about graph fusions, kernel optimizations, multi-GPU inference support, and more."
	/>
	<meta property="og:type" content="article" />
	<meta property="og:url" content="https://onnxruntime.ai/blogs/accelerating-llama-2" />
	<meta property="og:image" content={figure5} />
	<meta property="og:site_name" content="ONNX Runtime" />
	<meta name="twitter:card" content={figure5} />
	<meta name="twitter:title" content="Accelerating LLaMA-2 Inference with ONNX Runtime" />
	<meta
		name="twitter:description"
		content="Explore how ONNX Runtime can propel your Llama2 variants for faster inference."
	/>
	<meta name="twitter:image" content={figure5} />
</svelte:head>
<Header pathvar="" />
<div class="container mx-auto px-4 md:px-8 lg:px-48 pt-8">
	<h1 class="text-5xl pb-2">Accelerating LLaMA-2 Inference with ONNX Runtime</h1>
	<p class="text-neutral">
		By: <a href="https://www.linkedin.com/in/kunal-v-16315b94" class="text-blue-500"
			>Kunal Vaishnavi</a
		>
		and
		<a href="https://www.linkedin.com/in/parinitaparinita/" class="text-blue-500">Parinita Rahi</a>
	</p>
	<p class="text-neutral">14TH NOVEMBER, 2023 <span class="italic text-stone-500">(Updated 22nd November)</span></p>
	<div class="py-4">
		<p class="mb-4">
			Interested in running Llama2 faster? Let us explore how ONNX Runtime can propel your Llama2
			variants for faster inference!
		</p>

		<p class="mb-4">
			You can now experience significant inference gains—up to 3.8X faster—for the 7B, 13B, and 70B
			models, thanks to state-of-the-art fusion and kernel optimizations with ONNX Runtime. This
			blog details performance enhancements, dives into ONNX Runtime fusion optimizations, multi-GPU
			inferencing support, and guides you on how to leverage the cross-platform prowess of ONNX
			Runtime for seamless inferencing across platforms. This is the first in a series of upcoming
			blogs that will cover additional aspects for efficient memory usage with ONNX Runtime
			quantization updates, and cross-platform usage scenarios.
		</p>

		<h2 class="text-blue-500 text-3xl mb-4">Background: Llama2 and Microsoft</h2>

		<p class="mb-4">
			Llama2 is a state-of-the-art open source LLM from Meta ranging in scale from 7B to 70B
			parameters (7B, 13B, 70B). Microsoft and Meta <a
				href="https://blogs.microsoft.com/blog/2023/07/18/microsoft-and-meta-expand-their-ai-partnership-with-llama-2-on-azure-and-windows/"
				class="text-blue-500">announced</a
			> their AI on Azure and Windows collaboration in July 2023. As part of the announcement, Llama2
			was added to the Azure AI model catalog, which serves as a hub of foundation models that empower
			developers and machine learning (ML) professionals to easily discover, evaluate, customize, and
			deploy pre-built large AI models at scale.
		</p>

		<p class="mb-4">
			ONNX Runtime allows users to easily integrate the power of this generative AI model into your
			apps and services with improved optimizations that yield faster inferencing speeds and lower
			your costs.
		</p>

		<h2 class="text-blue-500 text-3xl mb-4">
			Faster Inferencing with New ONNX Runtime Optimizations
		</h2>

		<p class="mb-4">
			As part of the new 1.16.2 release, ONNX Runtime now has several built-in optimizations for
			Llama2, including graph fusions and kernel optimizations. The inference speedups, when
			compared to Hugging Face (HF) variants of Llama2 in PyTorch compile mode for prompt latency of
			CUDA FP16, are mentioned below. The end-to-end throughput or wall-clock throughput shown below
			is defined as <i
				>batch size * (prompt length + token generation length) / wall-clock latency</i
			> where wall-clock latency = the latency from running end-to-end and token generation length =
			256 generated tokens. The E2E throughput is 2.4X more (13B) and 1.8X more (7B) when compared to
			PyTorch compile. For higher batch size, sequence length like 16, 2048 pytorch eager times out,
			while ORT shows better performance than compile mode.
		</p>
		<div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
			<figure class="px-10 pt-4">
				<img src={figure1} alt="E2E Throughput Comparisons - Llama-2-7b" />
			</figure>
			<figure class="px-10 pt-4 my-auto">
				<img src={figure1b} alt="E2E Throughput Comparisons - Llama-2-13b" />
			</figure>
		</div>
		<div class="mt-2 mb-4 text-center">Figure 1: E2E Throughput Comparisons</div>

		<h2 class="text-blue-500 text-3xl mb-4">Latency and Throughput</h2>

		<p class="mb-4">
			The graphs below show latency comparisons between the ONNX Runtime and PyTorch variants of the
			Llama2 7B model on CUDA FP16. Latency here is defined as the time it takes to complete one
			pass through the model to produce the logits and synchronize the outputs.
		</p>

		<div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
			<figure class="px-10 pt-4">
				<img src={figure2} alt="Prompt Latency Comparisons - Llama-2-7b" />
			</figure>
			<figure class="px-10 pt-4 my-auto">
				<img src={figure2b} alt="Prompt Latency Comparisons - Llama-2-13b" />
			</figure>
		</div>
		<div class="mt-2 mb-4 text-center">Figure 2: Prompt Latency Comparisons</div>

		<p class="mb-4">
			Token generation throughput below is the average throughput of the first 256 tokens generated.
			We see up to ~1.4X (7B) and ~1.7X (13B) gains in token generation throughput when compared to
			PyTorch compile mode.
		</p>

		<div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
			<figure class="px-10 pt-4">
				<img src={figure3} alt="Tokens Generated Throughput Comparisons - Llama-2-7b" />
			</figure>
			<figure class="px-10 pt-4 my-auto">
				<img src={figure3b} alt="Tokens Generated Throughput Comparisons - Llama-2-13b" />
			</figure>
		</div>
		<div class="mt-2 mb-4 text-center">Figure 3: Tokens Generated Throughput Comparisons</div>

		<p class="mb-4">
			More details on these metrics can be found <a
				href="https://github.com/microsoft/onnxruntime-inference-examples/blob/main/python/models/llama2/README.md"
				class="text-blue-500">here</a
			>.
		</p>

		<h2 class="text-blue-500 text-3xl mb-4">ONNX Runtime with Multi-GPU Inference</h2>

		<p class="mb-4">
			ONNX Runtime supports multi-GPU inference to enable serving large models. Even in FP16
			precision, the LLaMA-2 70B model requires 140GB. Loading the model requires multiple GPUs for
			inference, even with a powerful NVIDIA A100 80GB GPU.
		</p>

		<p class="mb-4">
			ONNX Runtime applied <a href="https://arxiv.org/pdf/1909.08053.pdf" class="text-blue-500"
				>Megatron-LM</a
			>
			Tensor Parallelism on the 70B model to split the original model weight onto different GPUs. Megatron
			sharding on the 70B model shards the PyTorch model with FP16 precision into 4 partitions, converts
			each partition into ONNX format, and then applies a new ONNX Runtime graph fusion on the converted
			ONNX model. The 70B model has ~30 tokens per second throughput for token generation at batch size
			1, and end-to-end throughput starts at 30 tps for smaller sequence lengths with these optimizations.
			You can find additional example scripts
			<a
				href="https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/models/llama/"
				class="text-blue-500">here</a
			>.
		</p>

		<figure class="px-10 pt-4">
			<img src={figure4} alt="70B Llama2 Model Throughput" class="w-3/5 mx-auto" />
			<figcaption class="mt-2 mb-4 text-center">Figure 4: 70B Llama2 Model Throughput</figcaption>
		</figure>

		<h2 class="text-blue-500 text-3xl mb-4">ONNX Runtime Optimizations</h2>
		<figure class="px-10 pt-4">
			<img src={figure5} alt="LLaMA-2 Optimization Diagram" />
			<figcaption class="mt-2 mb-4 text-center">Figure 5: LLaMA-2 Optimization Diagram</figcaption>
		</figure>

		<p class="mb-4">
			The techniques that ONNX Runtime uses for optimizations, such as graph fusions, are applicable
			to state-of-the-art models. As these models become more complex, the techniques used to apply
			the graph fusions are adapted to accommodate the extra complexity. For example, instead of
			manually matching fusion patterns in the graph, ONNX Runtime now supports automated pattern
			matching. Rather than detect large subgraphs by hand and match the many paths they form,
			fusion opportunities can instead be identified by exporting a large module as a function and
			then pattern matching against a function's spec.
		</p>
		<figure class="px-10 pt-4">
			<img src={figure6} alt="Example of Rotary Embedding Function" />
			<figcaption class="mt-2 mb-4 text-center">
				Figure 6: Example of Rotary Embedding Function
			</figcaption>
		</figure>

		<p class="mb-4">
			As a concrete example, Figure 6 is an example of the nodes that comprise rotary embedding
			computations. Pattern matching against this subgraph is cumbersome because of the number of
			paths to verify. By exporting this as a function, the parent view of the graph will only show
			the inputs and outputs and represent all these nodes as a single operator.
		</p>

		<figure class="px-10 pt-4">
			<img src={figure7} alt="Example of Rotary Embedding Function in Parent Graph" />
			<figcaption class="mt-2 mb-4 text-center">
				Figure 7: Example of Rotary Embedding Function in Parent Graph
			</figcaption>
		</figure>

		<p class="mb-4">
			This approach makes it easier to maintain and support future versions of the rotary embedding
			computations because the pattern matching is only dependent on the operator's inputs and
			outputs instead of its internal semantic representation. It also allows other existing
			implementations of rotary embeddings in similar models such as GPT-NeoX, Falcon, Mistral,
			Zephyr, etc. to be pattern matched and fused with minimal or no changes.
		</p>

		<p class="mb-4">
			ONNX Runtime also adds support for the GroupQueryAttention (GQA) operator, which leverages the
			new Flash Attention V2 algorithm and its optimized kernels to efficiently compute attention.
			The GQA operator supports past-present buffer sharing between the past key/value cache (past
			KV cache) and the present key/value cache (present KV cache). By binding the present KV caches
			to the past KV caches, there is no need to allocate separate on-device memory for both caches.
			Instead, the past KV caches can be pre-allocated with enough on-device memory so that no new
			on-device memory needs to be requested during inference. This reduces memory usage when the KV
			caches become large during compute-intensive workloads and lowers latency by eliminating
			on-device memory allocation requests. The past-present buffer sharing can be enabled or
			disabled without needing to change the ONNX model, allowing greater flexibility for end users
			to decide which approach is best for them.
		</p>

		<p class="mb-4">
			In addition to these fusions and kernel optimizations, ONNX Runtime reduces the model’s memory
			usage. Besides quantization improvements (which will be covered in a future post), ONNX
			Runtime compresses the size of the cosine and sine caches used in each of the rotary
			embeddings by 50%. The compute kernels in ONNX Runtime that run the rotary embedding
			computations can then recognize this format and use their parallelized implementations to
			calculate the rotary embeddings more efficiently with less memory usage. The rotary embedding
			compute kernels also support interleaved and non-interleaved formats to support both the <a
				href="https://github.com/microsoft/Llama-2-Onnx"
				class="text-blue-500">Microsoft version of LLaMA-2</a
			>
			and the Hugging Face version of LLaMA-2 respectively while sharing the same calculations.
		</p>

		<p class="mb-4">
			The optimizations work for the <a
				href="https://huggingface.co/meta-llama"
				class="text-blue-500">Hugging Face versions</a
			>
			(models ending with <i>-hf</i>) and the Microsoft versions. You can download the optimized HF
			versions from
			<a href="https://github.com/microsoft/Llama-2-Onnx/tree/main-CUDA_CPU" class="text-blue-500"
				>Microsoft's LLaMA-2 ONNX repository</a
			>. Stay tuned for newer Microsoft versions coming soon!
		</p>

		<h2 class="text-blue-500 text-3xl mb-4">Optimize your own model using Olive</h2>

		<p class="mb-4">
			Olive is a hardware-aware model optimization tool that incorporates advanced techniques such
			as model compression, optimization, and compilation. We have made ONNX Runtime optimizations
			available through Olive so you can streamline the entire optimization process for a given
			hardware with simple experience.
		</p>

		<p class="mb-4">
			Here is an example of <a
				href="https://github.com/microsoft/Olive/tree/main/examples/llama2"
				class="text-blue-500">Llama2 optimization with Olive</a
			>, which harnesses ONNX Runtime optimizations highlighted in this blog. Distinct optimization
			flows cater to various requirements. For instance, you have the flexibility to choose
			different data types for quantization in CPU and GPU inference, based on your accuracy
			tolerance. Additionally, you can fine-tune your own Llama2 model with Olive-QLoRa on client
			GPUs and perform inference with ONNX Runtime optimizations.
		</p>

		<h2 class="text-blue-500 text-3xl mb-4">Usage Example</h2>

		<p class="mb-4">
			Here is a <a
				href="https://github.com/microsoft/onnxruntime-inference-examples/blob/main/python/models/llama2/LLaMA-2%20E2E%20Notebook.ipynb"
				class="text-blue-500">sample notebook</a
			> that shows you an end-to-end example of how you can use the above ONNX Runtime optimizations
			in your application.
		</p>

		<h2 class="text-blue-500 text-3xl mb-4">Conclusion</h2>

		<p class="mb-4">
			The advancements discussed in this blog provide faster Llama2 inferencing with ONNX Runtime,
			offering exciting possibilities for AI applications and research. With improved performance
			and efficiency, the horizon is wide open for innovation, and we eagerly await new applications
			built with Llama2 and ONNX Runtime by its vibrant community of developers. Stay tuned for more
			updates!
		</p>
	</div>
</div>
<Footer pathvar="" />
