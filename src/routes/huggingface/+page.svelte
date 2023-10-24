<script lang="ts">
	import LandingHero from '../components/landing-hero.svelte';
	import ImagesHf1 from '../../images/undraw/image_HF1.svelte';
	import ImageHf2 from '../../images/undraw/image_HF2.svelte';
	import ImageHf3 from '../../images/undraw/image_HF3.svelte';
	const title = 'Hugging Face + ONNX Runtime';
	const description =
		'ONNX Runtime can be used to accelerate well over 100,000 of the models available on Hugging Face.';
	const imgsrc = 'onnxruntimelogo';
	const imgalt = 'ONNX Runtime Logo';
</script>

<svelte:head>
	<meta
		name="description"
		content="Learn about how ONNX Runtime works with hundreds of thousands of models on Hugging Face."
	/>
</svelte:head>

<LandingHero {title} {description} {imgsrc} {imgalt} />
<div class="container mx-auto px-10 my-10">
	<div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
		<div>
			<h1 class="text-4xl pb-4">Supported Models</h1>
			<p>
				The top 30 most popular model families on Hugging Face are all supported by ONNX Runtime,
				and over 80 Hugging Face model families in total boast ORT support. <span
					class="hidden md:inline">The table outlines the models:</span
				>
			</p>
			<br />
			<p class="pb-4">
				ONNX models can be found directly from the Hugging Face Model Hub in its ONNX model library.
			</p>
			<a href="https://huggingface.co/models?library=onnx" class="btn btn-primary"
				>Check out Hugging Face's ONNX model library →</a
			>
			<br />
			<br />
			<p class="pb-4">
				Hugging Face also provides ONNX support for a variety of other models not listed in the ONNX
				model library. With Hugging Face Optimum, you can easily convert pretrained models to ONNX,
				and Transformers.js lets you run Hugging Face Transformers directly from your browser!
			</p>
			<a
				href="https://huggingface.co/docs/optimum/exporters/onnx/overview"
				class="btn btn-primary mr-0 mb-4 md:mr-4">Learn more about Hugging Face Optimum →</a
			>
			<a href="https://huggingface.co/docs/transformers.js/index" class="btn btn-primary"
				>Learn more about Transformers.js →</a
			>
		</div>
		<div class="hidden md:grid">
			<table class="table">
				<thead>
					<tr>
						<th>Model Family</th>
						<th>Approx. No. of Models</th>
						<th>ORT Support</th>
					</tr>
				</thead>
				<tbody>
					<tr>
						<td>bert</td>
						<td>24900</td>
						<td>Convertible to ONNX using Optimum API</td>
					</tr>
					<tr>
						<td>gpt2</td>
						<td>12700</td>
						<td>Convertible to ONNX using Optimum API</td>
					</tr>
					<tr>
						<td>distilbert</td>
						<td>10000</td>
						<td>Convertible to ONNX using Optimum API</td>
					</tr>
					<tr>
						<td>roberta</td>
						<td>9400</td>
						<td>Convertible to ONNX using Optimum API</td>
					</tr>
					<tr>
						<td>t5</td>
						<td>8800</td>
						<td>Convertible to ONNX using Optimum API</td>
					</tr>
					<tr>
						<td>wav2vec2</td>
						<td>5800</td>
						<td>Convertible to ONNX using Optimum API</td>
					</tr>
					<tr>
						<td>stable-diffusion</td>
						<td>4400</td>
						<td>Convertible to ONNX using Optimum API</td>
					</tr>
					<tr>
						<td>xlm-roberta</td>
						<td>4300</td>
						<td>Convertible to ONNX using Optimum API</td>
					</tr>
					<tr>
						<td>bart</td>
						<td>3200</td>
						<td>Convertible to ONNX using Optimum API</td>
					</tr>
					<tr>
						<td>whisper</td>
						<td>2900</td>
						<td>Convertible to ONNX using Optimum API</td>
					</tr>
					<tr>
						<td>marian</td>
						<td>2500</td>
						<td>Convertible to ONNX using Optimum API</td>
					</tr>
				</tbody>
			</table>
		</div>
	</div>
</div>

<div class="container mx-auto px-10 my-10">
	<div class="divider" />
	<div class="flex">
		<h1 class="text-4xl pb-4">Export Hugging Face Models to ONNX</h1>
		<div class="ml-5 hidden md:flex">
			<ImagesHf1 />
		</div>
	</div>
	<h2 class="text-2xl">PyTorch Models</h2>
	<p class="pb-4">
		Hugging Face's ONNX Export Space allows users to easily export PyTorch models from the Hugging
		Face Model Hub to ONNX.
	</p>
	<a href="https://huggingface.co/spaces/onnx/export" class="btn btn-primary"
		>Try out the Hugging Face ONNX Export Space for yourself →</a
	>
	<br />
	<br />
	<h2 class="text-2xl">Other Models</h2>
	<p class="pb-4">Hugging Face provides many alternatives for exporting models to ONNX.</p>
	<a href="https://huggingface.co/docs/transformers/serialization" class="btn btn-primary"
		>Other ONNX export options →</a
	>
</div>
<div class="container mx-auto px-10 my-10">
	<div class="divider" />
	<div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
		<div class="col-span-1">
			<div class="flex">
				<h1 class="text-4xl pb-4">Large Language Models</h1>
				<div class="ml-5 hidden md:flex">
					<ImageHf2 />
				</div>
			</div>
			<p>
				ONNX Runtime also supports many increasingly popular large language model (LLM) families,
				including most of those available in the HF Model Hub. <span class="hidden md:inline"
					>These model families are showcased in the table.</span
				>
			</p>
			<br />
			<p>
				Hugging Face also provides a leaderboard with more detailed tracking and evaluation of
				recently releases LLMs from the community.
			</p>
			<br />
			<a
				class="btn btn-primary"
				href="https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard"
				>Hugging Face Open LLM Leaderboard →</a
			>
		</div>
		<div class="overflow-x-auto hidden md:grid">
			<table class="table">
				<thead>
					<tr>
						<th>LLM Model Family</th>
						<th>Approx. No. of Models</th>
						<th>ORT Support</th>
					</tr>
				</thead>
				<tbody>
					<tr>
						<td>llama</td>
						<td>3785</td>
						<td>Convertible to ONNX using Optimum API</td>
					</tr>
					<tr>
						<td>gpt_neo</td>
						<td>892</td>
						<td>Convertible to ONNX using Optimum API</td>
					</tr>
					<tr>
						<td>bloom</td>
						<td>552</td>
						<td>Convertible to ONNX using Optimum API</td>
					</tr>
					<tr>
						<td>opt</td>
						<td>543</td>
						<td>Convertible to ONNX using Optimum API</td>
					</tr>
					<tr>
						<td>gpt-j</td>
						<td>17</td>
						<td>Convertible to ONNX using Optimum API</td>
					</tr>
					<tr>
						<td>flan-t5</td>
						<td>4</td>
						<td>Convertible to ONNX using Optimum API</td>
					</tr>
				</tbody>
			</table>
		</div>
	</div>
</div>
<div class="container mx-auto px-10 my-10">
	<div class="divider" />
	<div class="flex">
		<h1 class="text-4xl pb-4">Cloud Models</h1>
		<div class="ml-5 hidden md:flex">
			<ImageHf3 />
		</div>
	</div>
	<p>
		Models accelerated by ONNX Runtime can be easily deployed to the cloud through Azure ML, which
		improves time to value, streamlines MLOps, and provides built-in security.
	</p>
	<br />
	<p>
		Azure ML also publishes a curated model list that is updated regularly and includes some of the
		most popular models at the moment. Of the models on this list that are available on Hugging
		Face, there is currently Optimum ONNX support for over 85%.
	</p>
	<br />
	<a
		class="btn btn-primary"
		href="https://ml.azure.com/model/catalog?wsid=/subscriptions/48bbc269-ce89-4f6f-9a12-c6f91fcb772d/resourceGroups/aml1p-rg/providers/Microsoft.MachineLearningServices/workspaces/aml1p-ml-wus2&tid=72f988bf-86f1-41af-91ab-2d7cd011db47"
		>Azure ML Curated Model List →</a
	>
</div>
