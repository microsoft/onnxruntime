<script lang="ts">
	import LandingHero from '../components/landing-hero.svelte';

	let title = 'ONNX Runtime for Inferencing';
	let description =
		'ONNX Runtime Inference powers machine learning models in key Microsoft products and services across Office, Azure, Bing, as well as thousands of community projects.';
	let imgsrc = 'https://placehold.co/600x300';
	let imgalt = '';
</script>

<div class="container mx-auto px-10 my-10">
	<div class="grid grid-cols-1 md:grid-cols-3 gap-4 lg:gap-10">
		<div class="col-span-2">
			<h1 class="text-4xl">{title}</h1>
			<br /><br />
			<p class="text-xl">
				{description}
			</p>
			<br />
			<a href="https://onnxruntime.ai/docs/build/inferencing.html" class="btn btn-primary"
				>Learn how to build ONNX Runtime for inferencing →</a
			>
		</div>
		<img class="m-auto" src={imgsrc} alt={imgalt} />
	</div>
</div>
<div class="container mx-auto px-10 my-10">
	<h1 class="text-4xl pb-4">Benefits</h1>
	<div class="grid gap-10 grid-cols-1 md:grid-cols-3 pb-10">
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">Cost savings vs. running models in the cloud</h2>
			</div>
		</div>
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">Better latency and availability than request in the cloud</h2>
			</div>
		</div>
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">More privacy since data stays on device</h2>
			</div>
		</div>
	</div>
	<div class="grid gap-10 grid-cols-1 md:grid-cols-2 mx-auto">
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">
					Easily enable cross-platform portability with the same implementation through the browser
				</h2>
			</div>
		</div>
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">
					Simplify the distribution experience without needing any additional libraries and driver
					installations
				</h2>
			</div>
		</div>
	</div>
</div>
<div class="container mx-auto px-10 my-10">
	<h1 class="text-4xl pb-4">Use Cases</h1>
	<div class="grid gap-10 grid-cols-1 md:grid-cols-2 lg:grid-cols-4 mx-auto">
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">Improve inference performance for a wide variety of ML models</h2>
			</div>
		</div>
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">Run on different hardware and operating systems</h2>
			</div>
		</div>
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">Train in Python but deploy into a C#/C++/Java app</h2>
			</div>
		</div>
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">
					Train and perform inference with models created in different frameworks
				</h2>
			</div>
		</div>
	</div>
</div>
<LandingHero
	title="ONNX Runtime Mobile"
	description="ONNX Runtime Mobile allows you to run model inferencing on mobile devices (iOS and Android)."
	imgsrc="https://placehold.co/600x300"
	imgalt=""
/>
<div class="container mx-auto px-10 my-10">
	<div class="grid grid-cols-1 md:grid-cols-3 gap-4 lg:gap-10">
		<div class="col-span-2">
			<h1 class="text-4xl">ONNX Runtime Web</h1>
			<br /><br />
			<p class="text-xl">
				ONNX Runtime Web allows JavaScript developers to run and deploy machine learning models in
				browsers.
			</p>
			<br />
			<a href="https://www.youtube.com/watch?v=vYzWrT3A7wQ" class="btn btn-primary"
				>Inference in JavaScript with ONNX Runtime Web YouTube Tutorial →</a
			>
		</div>
		<img class="m-auto" src={imgsrc} alt={imgalt} />
	</div>
</div>
<div class="container mx-auto px-10 my-10">
	<div class="grid grid-cols-1 md:grid-cols-3 gap-4 lg:gap-10">
		<div class="col-span-2">
			<h1 class="text-4xl">On-Device Training</h1>
			<br /><br />
			<p class="text-xl">ONNX Runtime on-device training is an extension of ORT Inferencing.</p>
			<br />
			<a href="https://www.youtube.com/watch?v=vYzWrT3A7wQ" class="btn btn-primary"
				>Learn more about on-device training →</a
			>
		</div>
		<img class="m-auto" src={imgsrc} alt={imgalt} />
	</div>
</div>
