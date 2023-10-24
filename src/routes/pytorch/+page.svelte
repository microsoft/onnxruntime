<svelte:head>
	<meta
		name="description"
		content="Learn about the benefits of using ONNX Runtime with your PyTorch models."
	/>
</svelte:head>
<script lang="ts">
	import Highlight from 'svelte-highlight';
	import python from 'svelte-highlight/languages/python';
	import LandingHero from '../components/landing-hero.svelte';
	import ImagePt1 from '../../images/undraw/image_PT1.svelte';
	import ImagePt2 from '../../images/undraw/image_PT2.svelte';
	import ImagePt3 from '../../images/undraw/image_PT3.svelte';
	import ImagePt4 from '../../images/undraw/image_PT4.svelte';
	import ImagePt5 from '../../images/undraw/image_PT5.svelte';
	import prodready from '../../images/prod-ready.png';
	const title = 'PyTorch + ONNX Runtime';
	const description =
		'PyTorch leads the deep learning landscape with its readily digestible and flexible API; the large number of ready-made models available, particularly in the natural language (NLP) domain; as well as its domain specific libraries.';
	const imgsrc = 'onnxruntimelogo';
	const imgalt = 'ONNX Runtime Logo';
	const pythonCode = 'import torch\n\ntorch.onnx.export(\nmodel,\ninputs,\n"model.onnx")';
</script>

<LandingHero {title} {description} {imgsrc} {imgalt} />
<div class="container mx-auto px-10 my-10">
	<div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
		<div class="card bg-base-300">
			<figure class="px-10 pt-10">
				<ImagePt1 />
			</figure>
			<div class="card-body items-center text-center">
				<h2 class="card-title">Deploy anywhere</h2>
				<p>Run PyTorch models on cloud, desktop, mobile, IoT, and even in the browser</p>
			</div>
		</div>
		<div class="card bg-base-300">
			<figure class="px-10 pt-10">
				<ImagePt2 />
			</figure>
			<div class="card-body items-center text-center">
				<h2 class="card-title">Boost performance</h2>
				<p>Accelerate PyTorch models to improve user experience and reduce costs</p>
			</div>
		</div>
		<div class="card bg-base-300">
			<figure class="px-10 pt-10">
				<ImagePt3 />
			</figure>
			<div class="card-body items-center text-center">
				<h2 class="card-title">Improve time to market</h2>
				<p>Used by Microsoft and many others for their production PyTorch workloads</p>
			</div>
		</div>
	</div>
</div>
<div class="container mx-auto px-10 my-10">
	<h1 class="text-4xl pb-4">Why PyTorch + ONNX Runtime?</h1>
	<div class="container mx-auto">
		<div class="lg:grid pb-10 gap-10 grid-cols-2">
			<article class="card bg-base-300 lg:mb-48 mb-10 grid md:grid-cols-3">
				<!-- Order-first and order-last classes makes for cleaner small device viewing -->
				<div class="card-body col-span-2 order-last md:order-first">
					<h2 class="card-title">Native support in PyTorch</h2>
					<p>
						PyTorch includes support for ONNX through the torch.onnx APIs to simplify exporting your
						PyTorch model to the portable ONNX format. The ONNX Runtime team maintains these
						exporter APIs to ensure a high level of compatibility with PyTorch models.
					</p>
					<div class="card-actions">
						<a href="https://onnxruntime.ai/docs/api/" class="btn btn-primary"
							><span class="hidden md:grid">Get your PyTorch Models ready for</span> optimized deployment
							→</a
						>
					</div>
				</div>
				<pre class="md:col-span-1 col-span-2 order-first sm:order-last">
					<Highlight language={python} code={pythonCode} />
				</pre>
			</article>
			<article class="card bg-base-300 lg:mt-48 grid md:grid-cols-3">
				<div class="card-body col-span-2 order-last md:order-first">
					<h2 class="card-title">Production Ready</h2>
					<p>
						Train and deploy models reliably and at scale using a built-in PyTorch environment
						within Azure Machine Learning to ensure that the latest PyTorch version is fully
						supported through a lightweight, standalone environment that includes needed components
						like ONNX Runtime for Training to effectively run optimized training for large models.
					</p>
					<div class="card-actions">
						<a
							href="https://techcommunity.microsoft.com/t5/ai-machine-learning-blog/azure-container-for-pytorch-is-now-generally-available-in-azure/ba-p/3774616"
							class="btn btn-primary">Learn about ACPT →</a
						>
					</div>
				</div>
				<figure class="md:col-span-1 col-span-2 order-first sm:order-last">
					<img src={prodready} alt="" />
				</figure>
			</article>
		</div>
	</div>
	<div class="container mx-auto">
		<div class="lg:grid pb-10 gap-10 grid-cols-2">
			<article class="card bg-base-300 lg:mb-48 mb-10 grid md:grid-cols-3">
				<!-- Order-first and order-last classes makes for cleaner small device viewing -->
				<div class="card-body col-span-2 order-last md:order-first">
					<h2 class="card-title">Lower latency, higher throughput</h2>
					<p>
						Better performance can help improve your user experience and lower your operating costs.
						A wide range of models from computer vision (ResNet, MobileNet, Inception, YOLO, super
						resolution, etc) to speech and NLP (BERT, RoBERTa, GPT-2, T5, etc) can benefit from ONNX
						Runtime's optimized performance. The ONNX Runtime team regularly benchmarks and
						optimizes top models for performance. ONNX Runtime also integrates with top hardware
						accelerator libraries like TensorRT and OpenVINO so you can get the best performance on
						the hardware available while using the same common APIs across all your target
						platforms.
					</p>
					<div class="card-actions">
						<a class="btn btn-primary hidden md:grid" href="https://cloudblogs.microsoft.com/opensource/tag/onnx"
							>Check the latest on performance enhancements →</a
						>
						<a class="btn btn-primary md:hidden" href="https://cloudblogs.microsoft.com/opensource/tag/onnx">Performance enhancements →</a>
					</div>
				</div>
				<figure class="md:col-span-1 col-span-2 order-first sm:order-last">
					<ImagePt4 />
				</figure>
			</article>
			<article class="card bg-base-300 lg:mt-48 grid md:grid-cols-3">
				<div class="card-body col-span-2 order-last md:order-first">
					<h2 class="card-title">Get innovations into production faster</h2>
					<p>
						Development agility is a key factor in overall costs. ONNX Runtime was built on the
						experience of taking PyTorch models to production in high scale services like Microsoft
						Office, Bing, and Azure. It used to take weeks and months to take a model from R&D to
						production. With ONNX Runtime, models can be ready to be deployed at scale in hours or
						days.
					</p>
					<div class="card-actions">
						<a href="./testimonials" class="btn btn-primary hidden md:grid"
							>See what ONNX Runtime users are saying →</a
						>
						<a href="./testimonials" class="btn btn-primary md:hidden">Testimonials →</a>
					</div>
				</div>
				<figure class="md:col-span-1 col-span-2 order-first sm:order-last">
					<ImagePt5 />
				</figure>
			</article>
		</div>
	</div>
</div>
