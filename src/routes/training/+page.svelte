<script lang="ts">
	import LandingHero from '../components/landing-hero.svelte';
	import ImageLargemodeltraining from '../../images/undraw/image_largemodeltraining.svelte';
	import ImageOndevtraining from '../../images/undraw/image_ondevtraining.svelte';
	import ImageTraining1 from '../../images/undraw/image_training1.svelte';
	import ImageTraining2 from '../../images/undraw/image_training2.svelte';
	const title = 'ONNX Runtime for Training';
	const description =
		'ONNX Runtime can be used to accelerate both large model training and on-device training.';
	const imgsrc = 'onnxruntimelogo';
	const imgalt = 'ONNX Runtime Logo';
</script>

<LandingHero {title} {description} {imgsrc} {imgalt} />
<div class="container mx-auto px-10 my-10">
	<div class="grid grid-cols-1 md:grid-cols-3 gap-4 lg:gap-10">
		<div class="col-span-2">
			<h1 class="text-3xl">Large Model Training</h1>
			<br /><br />
			<p class="text-xl">
				ORTModule accelerates training of large transformer PyTorch models. The training time and
				cost are reduced with a single line code change. It is built on top of highly successful and
				proven technologies of ONNX Runtime and ONNX format.
			</p>
			<br />
			<div class="bg-white w-100 md:w-1/2 p-4">
				<code>
					<span class="text-red-500">- model = build_model() # User's PyTorch model</span><br />
					<span class="text-green-500">+ model = ORTModule(build_model())</span>
				</code>
			</div>
			<br /><br />
			<a
				href="https://onnxruntime.ai/docs/get-started/training-pytorch.html"
				class="btn btn-primary">Get started with large model training →</a
			>
		</div>
		<div class="mx-auto">
			<ImageLargemodeltraining widthscale={1.8} heightscale={1.8} />
		</div>
	</div>
</div>
<div class="container mx-auto px-10 my-10">
	<h1 class="text-4xl pb-4">Benefits</h1>
	<div class="grid gap-10 grid-cols-1 md:grid-cols-2 lg:grid-cols-4 pb-10">
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">Part of the PyTorch ecosystem</h2>
				<p>
					ONNX Runtime Training is available via the <a
						class="text-blue-500"
						href="https://pytorch.org/ort/">torch-ort</a
					>
					package as part of the
					<a
						class="text-blue-500"
						href="https://learn.microsoft.com/en-us/azure/machine-learning/resource-azure-container-for-pytorch?view=azureml-api-2"
						>Azure Container for PyTorch (ACPT)</a
					> and seamlessly integrates with existing training pipelines for PyTorch models​.
				</p>
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
				<h2 class="card-title">Faster training</h2>
				<p>Optimized kernels and memory optimizations provides >1.5X speed up in training time.</p>
			</div>
		</div>
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">Composable with popular acceleration systems</h2>
				<p>
					Compose with Deepspeed, Fairscale, Megatron, and more for even faster and more efficient
					training.
				</p>
			</div>
		</div>
	</div>
</div>
<div class="container mx-auto px-10 my-10">
	<div class="bg-secondary rounded p-4">
		<h1 class="text-3xl pb-4">Examples</h1>
		<div class="overflow-x-auto p-4">
			<table class="table">
				<thead>
					<tr>
						<th>Example</th>
						<th>Performance Comparison</th>
						<th>Model Change</th>
					</tr>
				</thead>
				<tbody>
					<tr>
						<td>HuggingFace BART</td>
						<td
							>See <a
								class="text-blue-200"
								href="https://github.com/microsoft/onnxruntime-training-examples/blob/master/huggingface/BART.md"
								>BART</a
							></td
						>
						<td>No model change required</td>
					</tr>
					<tr>
						<td>HuggingFace BERT</td>
						<td
							>See <a
								class="text-blue-200"
								href="https://github.com/microsoft/onnxruntime-training-examples/blob/master/huggingface/BERT.md"
								>BERT</a
							></td
						>
						<td>No model change required</td>
					</tr>
					<tr>
						<td>HuggingFace DistilBERT</td>
						<td
							>See <a
								class="text-blue-200"
								href="https://github.com/microsoft/onnxruntime-training-examples/blob/master/huggingface/DistilBERT.md"
								>DistilBERT</a
							></td
						>
						<td>No model change required</td>
					</tr>
					<tr>
						<td>HuggingFace GPT2</td>
						<td
							>See <a
								class="text-blue-200"
								href="https://github.com/microsoft/onnxruntime-training-examples/blob/master/huggingface/GPT2.md"
								>GPT2</a
							></td
						>
						<td>No model change required</td>
					</tr>
				</tbody>
			</table>
		</div>
	</div>
</div>
<div class="container mx-auto px-10 my-10">
	<div class="grid grid-cols-1 md:grid-cols-3 gap-4 lg:gap-10">
		<div class="col-span-2">
			<h1 class="text-4xl">On-Device Training</h1>
			<br /><br />
			<p class="text-xl">
				On-Device Training refers to the process of training a model on an edge device, such as
				mobile phones, embedded devices, gaming consoles, web browsers, etc. This is in contrast to
				training a model on a server or a cloud.
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
	<div class="grid gap-10 grid-cols-1 md:grid-cols-3 pb-10">
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">Extends the ONNX Runtime Inference solution</h2>
			</div>
		</div>
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
	</div>
	<div class="grid gap-10 grid-cols-1 md:grid-cols-2 mx-auto">
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">Improving data privacy & security</h2>
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
				<p class="list-disc">
					Examples:
					<li>Image / Audio classification</li>
					<li>Text Prediction</li>
				</p>
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
				<p class="list-disc">
					Examples:
					<li>Medical research</li>
					<li>Autonomous vehicles</li>
					<li>Robotics</li>
				</p>
			</div>
			<figure>
				<ImageTraining2 />
			</figure>
		</div>
	</div>
</div>
