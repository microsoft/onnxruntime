<script>
	import { base } from '$app/paths';
	import Largemodeltraining from '../../images/undraw/image_largemodeltraining.svelte';
	import Ondevtraining from '../../images/undraw/image_ondevtraining.svelte';
	import Ortmobile from '../../images/undraw/image_ortmobile.svelte';
	import Ortweb from '../../images/undraw/image_ortweb.svelte';
</script>

<div class="container mx-auto px-10">
	<h1 class="text-4xl pb-2">ONNX Runtime Training</h1>
	<p class="text-xl pb-4">
		ONNX Runtime can be used to accelerate large model training and enable on-device training.
	</p>
	<a href="./training" class="btn btn-primary">Learn more about ONNX Runtime Training →</a>
	<div class="grid grid-cols-1 md:grid-cols-2 gap-10 mt-10 my-4 md:mx-10">
		<div class="bg-slate-300 p-4 rounded">
			<div class="grid xl:grid-cols-4 place-items-center">
				<div class="col-span-3 text-black">
					<h1 class="text-2xl pb-2">Large Model Training</h1>
					<p class="text-lg">
						ORT Training can be used to accelerate training for a large number of popular models,
						including <a href="https://huggingface.co/" class="text-blue-500">Hugging Face</a>
						models like Llama-2-7b and curated models from the
						<a href="https://ml.azure.com/" class="text-blue-500"
							>Azure AI | Machine Learning Studio</a
						> model catalog.
					</p>
				</div>
				<div class="hidden xl:grid">
					<Largemodeltraining />
				</div>
			</div>
		</div>
		<div class="bg-slate-300 p-4 rounded">
			<div class="grid md:grid-cols-4 place-items-center">
				<div class="col-span-3 text-black">
					<h1 class="text-2xl pb-2">On-Device Training</h1>
					<p class="text-lg">
						On-device training extends ORT Mobile inferencing to enable training on edge devices.
						Developers can take an inference model and train it locally on-device to provide an
						improved user experience for end customers.
					</p>
				</div>
				<div class="hidden xl:grid">
					<Ondevtraining />
				</div>
			</div>
		</div>
	</div>
</div>
<div class="container mx-auto px-10 my-10">
	<h1 class="text-4xl pb-2">ONNX Runtime Inferencing</h1>
	<p class="text-xl pb-4">
		ONNX Runtime Inference powers machine learning models in key Microsoft products and services
		across Office, Azure, Bing, as well as thousands of community projects.
	</p>
	<a href="./inference" class="btn btn-primary">Learn more about ONNX Runtime Inferencing →</a>
	<div class="grid grid-cols-1 md:grid-cols-2 gap-10 mt-10 my-4 md:mx-10">
		<div class="bg-slate-300 p-4 rounded">
			<div class="grid xl:grid-cols-4 place-items-center">
				<div class="col-span-3 text-black">
					<h1 class="text-2xl pb-2">ONNX Runtime Web</h1>
					<p class="text-lg">
						ONNX Runtime Web allows JavaScript developers to run and deploy machine learning models
						in browsers.
					</p>
				</div>
				<div class="hidden xl:grid">
					<Ortweb />
				</div>
			</div>
		</div>
		<div class="bg-slate-300 p-4 rounded">
			<div class="grid md:grid-cols-4 place-items-center">
				<div class="col-span-3 text-black">
					<h1 class="text-2xl pb-2">ONNX Runtime Mobile</h1>
					<p class="text-lg">
						ONNX Runtime Mobile allows you to run model inferencing on mobile devices.
					</p>
				</div>
				<div class="hidden xl:grid">
					<Ortmobile />
				</div>
			</div>
		</div>
	</div>
</div>
