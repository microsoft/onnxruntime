<script lang="ts">
	import LandingHero from '../components/landing-hero.svelte';
	import ImageLargemodeltraining from '../../images/undraw/image_largemodeltraining.svelte';
	import ImageOndevtraining from '../../images/undraw/image_ondevtraining.svelte';
	import ImageTraining1 from '../../images/undraw/image_training1.svelte';
	import ImageTraining2 from '../../images/undraw/image_training2.svelte';
	import foundationmodels from '../../images/foundationmodels.png';
	const title = 'ONNX Runtime for Training';
	const description =
		'ONNX Runtime can be used to accelerate both large model training and on-device training.';
	const imgsrc = 'onnxruntimelogo';
	const imgalt = 'ONNX Runtime Logo';
	let image = 'https://i.ibb.co/0YBy62j/ORT-icon-for-light-bg.png'
	let imageSquare = 'https://i.ibb.co/0YBy62j/ORT-icon-for-light-bg.png'
	let authors = ['']
	let keywords = 'onnx runtime, onnx, onnxruntime, onnx runtime performance, onnx runtime training, onnx runtime cross-platform, onnx runtime cross platform, onnx runtime crossplatform, onnx runtime training, onnx runtime cross-platform, onnx runtime cross platform, onnx runtime crossplatform'
</script>
<svelte:head>
	<!-- Dynamic meta tags -->
	<meta name="description" content={description} />
	<meta name="image" content={image} />
	<meta name="author" content={authors.join(', ')} />
	<meta name="keywords" content={keywords} />
	<!-- Open Graph / Facebook -->
	<meta property="og:description" content={description}/>
	<meta property="og:image" content={image} />
	
	<!-- Twitter -->
	<meta property="twitter:description" content={description} />
	<meta property="twitter:image" content={image} />
	<meta property="twitter:card" content={imageSquare} />
</svelte:head>

<LandingHero {title} {description} {imgsrc} {imgalt} />
<div class="container mx-auto px-10 my-10">
	<div class="grid grid-cols-1 md:grid-cols-3 gap-4 lg:gap-10">
		<div class="col-span-2">
			<h1 class="text-3xl">Large Model Training</h1>
			<br /><br />
			<p class="text-xl">
				ORTModule accelerates training of large transformer based PyTorch models. The training time
				and training cost is reduced with a few lines of code change. It is built on top of highly
				successful and proven technologies of ONNX Runtime and ONNX format. It is composable with
				technologies like DeepSpeed and accelerates pre-training and finetuning for state of the art
				LLMs. It is integrated in the Hugging Face Optimum library which provides an ORTTrainer API
				to use ONNX Runtime as the backend for training acceleration.
			</p>
			<br />
			<div class="bg-white w-100 md:w-1/2 p-4">
				<code>
					<span class="text-red-600">- model = build_model() # User's PyTorch model</span><br />
					<span class="text-green-700">+ model = ORTModule(build_model())</span>
				</code>
			</div>
			<br /><br />
			<a
				href="https://onnxruntime.ai/docs/get-started/training-pytorch.html"
				class="btn btn-primary">Get started with large model training →</a
			>
		</div>
		<div class="mx-auto md:pt-10">
			<ImageLargemodeltraining widthscale={1} heightscale={1} />
		</div>
	</div>
</div>
<div class="container mx-auto px-10 my-10">
	<h1 class="text-4xl pb-4">Benefits</h1>
	<div class="grid gap-10 grid-cols-1 md:grid-cols-2 lg:grid-cols-3 pb-10">
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">Faster training</h2>
				<p>Optimized kernels and memory optimizations provides >1.5X speed up in training time.</p>
			</div>
		</div>
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">Flexible & extensible hardware support</h2>
				<p>
					The same model and API works with NVIDIA and AMD GPUs, and the extensible "execution
					provider" architecture allow you to plug-in custom operators, optimizer and hardware
					accelerators.
				</p>
			</div>
		</div>
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">Part of the PyTorch ecosystem</h2>
				<p>
					ONNX Runtime Training is available via the <a
						class="dark:text-blue-300 text-blue-800 underline"
						href="https://pytorch.org/ort/">torch-ort</a
					>
					package as part of the
					<a
						class="dark:text-blue-300 text-blue-800 underline"
						href="https://learn.microsoft.com/en-us/azure/machine-learning/resource-azure-container-for-pytorch?view=azureml-api-2"
						>Azure Container for PyTorch (ACPT)</a
					> and seamlessly integrates with existing training pipelines for PyTorch models.
				</p>
			</div>
		</div>
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">Composable with popular acceleration systems</h2>
				<p>
					Compose with <a href="https://github.com/microsoft/DeepSpeed" class="dark:text-blue-300 text-blue-800 underline"
						>DeepSpeed</a
					>,
					<a href="https://github.com/facebookresearch/fairscale" class="dark:text-blue-300 text-blue-800 underline">FairScale</a
					>, <a href="https://github.com/NVIDIA/Megatron-LM" class="dark:text-blue-300 text-blue-800 underline">Megatron</a>, and
					more for even faster and more efficient training.
				</p>
			</div>
		</div>
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">Works with Azure AI curated models</h2>
				<p>
					ORT Training is turned on for curated models in the <a
						href="https://ml.azure.com/"
						class="dark:text-blue-300 text-blue-800 underline">Azure AI | Machine Learning Studio</a
					> model catalog.
				</p>
			</div>
		</div>
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">Can be used to accelerate popular models like Llama-2-7b</h2>
				<p>
					ORT Training can be used to accelerate Hugging Face models like Llama-2-7b through <a
						href="https://github.com/huggingface/optimum/blob/main/examples/onnxruntime/training/text-classification/README.md#onnx-runtime-training"
						class="dark:text-blue-300 text-blue-800 underline">these scripts</a
					>.
				</p>
			</div>
		</div>
	</div>
</div>
<div class="container mx-auto px-10 my-10">
	<h1 class="text-4xl pb-4">Improved Foundation Model Performance with ORT Training</h1>
	<div class="grid lg:grid-cols-3 grid-cols-1 lg:gap-4">
		<img class="col-span-2" src={foundationmodels} alt="Foundation Model Throughput chart" />
		<div class="my-auto flex flex-col gap-4 mt-4 lg:mt-auto">
			<div class="card bg-base-300">
				<div class="card-body items-center text-center">
					<div class="card-title">Average throughput improvement:</div>
					<div class="stat-value">2.7x</div>
				</div>
			</div>
			<div class="card bg-base-300">
				<div class="card-body items-center text-center">
					<div class="card-title">Median throughput improvement:</div>
					<div class="stat-value">1.7x</div>
				</div>
			</div>
		</div>
	</div>
</div>
<div class="container mx-auto px-10 my-10">
	<div class="grid grid-cols-1 md:grid-cols-3 gap-4 lg:gap-10">
		<div class="col-span-2">
			<h1 class="text-4xl" id="on-device-training">On-Device Training</h1>
			<br /><br />
			<p class="text-xl">
				On-Device Training refers to the process of training a model on an edge device, such as
				mobile phones, embedded devices, gaming consoles, web browsers, etc. This is in contrast to
				training a model on a server or a cloud. On-Device Training extends the Inference ecosystem
				to leverage data on the device for providing customized user experiences on the edge. Once
				the model is trained on the device, it can be used to get an Inference model for deployment,
				update global weights for federated learning or create a checkpoint for future use. It also
				preserves user privacy by training on the device.
			</p>
			<br />
			<a
				href="https://onnxruntime.ai/docs/get-started/training-on-device.html"
				class="btn btn-primary">Get started with on-device training →</a
			>
		</div>
		<div class="m-auto">
			<ImageOndevtraining />
		</div>
	</div>
</div>
<div class="container mx-auto px-10 my-10">
	<h1 class="text-4xl pb-4">Benefits</h1>
	<div class="grid gap-10 grid-cols-1 md:grid-cols-2 pb-10">
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">Memory and performance efficiency</h2>
				<p>for lower resource consumption on device</p>
			</div>
		</div>
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">Simple APIs and multiple language bindings</h2>
				<p>make it easy to scale across multiple platform targets</p>
			</div>
		</div>
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">Improves data privacy & security</h2>
				<p>
					especially when working with sensitive data that cannot be shared with a server or a cloud
				</p>
			</div>
		</div>
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">Same solution runs cross-platform</h2>
				<p>on cloud, desktop, edge, and mobile</p>
			</div>
		</div>
	</div>
</div>
<div class="container mx-auto px-10 my-10">
	<h1 class="text-4xl pb-4">Use Cases</h1>
	<div class="grid gap-10 grid-cols-1 lg:grid-cols-2">
		<div class="card bg-base-300">
			<div class="card-body">
				<h2 class="text-xl">
					<span class="font-bold">Personalization tasks</span> where the model needs to be trained on
					the user's data
				</h2>
				Examples:
				<ul class="list-disc list-inside">
					<li>Image / Audio classification</li>
					<li>Text Prediction</li>
				</ul>
			</div>
			<figure>
				<ImageTraining1 />
			</figure>
		</div>
		<div class="card bg-base-300">
			<div class="card-body">
				<h2 class="text-xl">
					<span class="font-bold">Federated learning tasks</span> where the model is locally trained
					on data distributed across multiple devices to build a more robust aggregated global model
				</h2>
				Examples:
				<ul class="list-disc list-inside">
					<li>Medical research</li>
					<li>Autonomous vehicles</li>
					<li>Robotics</li>
				</ul>
			</div>
			<figure>
				<ImageTraining2 />
			</figure>
		</div>
	</div>
</div>
