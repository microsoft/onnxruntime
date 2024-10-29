<script>
	let modelhubs = [
		{
			title: 'ONNX Community (hosted on Hugging Face)',
			description:
				'Access, share, and discuss ONNX models published to Hugging Face by members of the ONNX Community.',
			url: 'https://huggingface.co/onnx-community',
			image: 'https://huggingface.co/front/assets/huggingface_logo-noborder.svg',
			imagealt: 'Hugging Face Logo'
		},
		{
			title: 'Model Zoo (hosted on ONNX website)',
			description:
				'Explore a wide variety of pre-trained ONNX models curated by the ONNX community.',
			url: 'https://onnx.ai/models/',
			image: 'https://onnx.ai/models/assets/logos/onnx.png',
			imagealt: 'ONNX Model Zoo'
		},
		{
			title: 'AMD Models (hosted on Hugging Face)',
			description:
				'Various quark-quantized models are graciously provided by AMD, ready to run using the ONNX Runtime generate() api!',
			url: 'https://huggingface.co/amd',
			image: 'https://cdn-avatars.huggingface.co/v1/production/uploads/647fcd2cdd29d0de8120db74/Og-6kJn0Hkjy5s0TnUWt_.png',
			imagealt: 'AMD Logo on Hugging Face'
		},
		{
			title: 'Qualcomm ONNX Models (hosted on Hugging Face)',
			description:
				'Discover ONNX-compatible models optimized for Qualcomm hardware on Hugging Face.',
			url: 'https://huggingface.co/qualcomm',
			image: 'https://logosandtypes.com/wp-content/uploads/2020/08/qualcomm.svg',
			imagealt: 'Qualcomm AI Hub Logo'
		},
		{
			title: 'All ONNX Models on Hugging Face',
			description:
				'Access all ONNX models available on the Hugging Face Model Hub here!',
			url: 'https://huggingface.co/models?library=onnx&sort=trending',
			image: 'https://huggingface.co/front/assets/huggingface_logo-noborder.svg',
			imagealt: 'Hugging Face Logo'
		}
	];

    let description = 'ONNX Models - find ONNX models for natural language processing, computer vision, and more.';
    let keywords = 'onnx models, onnx model zoo, onnx community, onnx models huggingface, onnx models qualcomm';
</script>
<svelte:head>
	<!-- Dynamic meta tags -->
	<meta name="description" content={description} />
	<meta name="keywords" content={keywords} />
	<!-- Open Graph / Facebook -->
	<meta property="og:description" content={description}/>
	
	<!-- Twitter -->
	<meta property="twitter:description" content={description} />
</svelte:head>
<div class="container mx-auto px-8">
	<h1 class="text-3xl">ONNX Models</h1>
	<p>
		ONNX is the Open Neural Network Exchange, and we take that name to heart! Many members of the
		community upload their ONNX models to various repositories, and we want to make it easy for you
		to find them. Below are some of the most popular repositories where you can find ONNX models:
	</p>

	<div class="my-8 grid grid-cols-1 md:grid-cols-2 gap-8">
		{#each modelhubs as modelhub}
			<div class="card bg-success image-full transition hover:scale-105">
				<a rel="noopener noreferrer" target="_blank" href={modelhub.url} class="card-body">
					<div class="grid grid-cols-5">
                        <div class="h-full">
                            <img src={modelhub.image} alt={modelhub.imagealt} class="w-24" />
                        </div>
						<div class="col-span-4">
							<h2 class="card-title text-white">{modelhub.title}</h2>
							<p class="text-white">{modelhub.description}</p>
						</div>
					</div>
				</a>
			</div>
		{/each}
	</div>

	<h2 class="text-2xl">Can't find what you're looking for?</h2>
	<p>
		Convert to ONNX, optimize, and quantize your own models quickly and easily with <a
			class="text-blue-700 underline"
			href="https://github.com/microsoft/Olive/tree/main">Olive</a
		>. Here's a quick snippet showing you how easy it can be done:
	</p>
	<div class="my-4 mockup-code bg-slate-300 dark:bg-primary">
		<pre data-prefix="$" class="text-black"><code>pip install olive-ai onnxruntime optimum</code></pre>
		<pre data-prefix=">" class="text-success dark:text-info"><code>olive auto-opt -m microsoft/Phi-3-mini-4k-instruct -o models/phi3-mini-4k</code></pre>
	</div>
</div>
