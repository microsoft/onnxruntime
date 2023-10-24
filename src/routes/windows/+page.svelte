<svelte:head>
	<meta
		name="description"
		content="Learn about using ONNX Runtime to create AI experiences on Windows."
	/>
</svelte:head>
<script lang="ts">
	import OnnxLight from '../../images/ONNX-Light.svelte';
	import LandingHero from '../components/landing-hero.svelte';
	const title = 'Windows + ONNX Runtime';
	const description =
		'ONNX Runtime makes it easier for you to create amazing AI experiences on Windows with less engineering effort and better performance.';
	const imgsrc = 'onnxruntimelogo';
	const imgalt = 'ONNX Runtime Logo';
</script>

<div class="container mx-auto px-10 my-10">
	<div class="grid grid-cols-1 md:grid-cols-3 gap-10">
		<div class="col-span-2">
			<h1 class="text-4xl">{title}</h1>
			<br /><br />
			<p class="text-xl">
				{description}
			</p>
			<br />
			<a
				href="https://onnxruntime.ai/docs/get-started/with-windows.html#windows-os-integration"
				class="btn btn-primary">Get started with ONNX Runtime for Windows →</a
			>
		</div>
		<div class="m-auto">
			<OnnxLight height={250} width={250}/>
		</div>
	</div>
</div>
<div class="container mx-auto px-10 my-10">
	<h1 class="text-4xl pb-4">Benefits</h1>
	<div class="grid gap-10 grid-cols-1 md:grid-cols-3 pb-10">
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">
					Run AI models on Windows with the same tools used by Microsoft internally
				</h2>
			</div>
		</div>
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">Extend your Windows AI investment to all your app platforms</h2>
			</div>
		</div>
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">
					Enable hybrid inference scenarios that switch between local resources and the cloud
				</h2>
			</div>
		</div>
	</div>
	<p>
		To learn more about the benefits of using ONNX Runtime with Windows, check out some of our
		recent blogs:
	</p>
	<a
		class="btn btn-primary mb-4 md:mb:0"
		href="https://blogs.windows.com/windowsdeveloper/2023/05/23/unlocking-the-end-to-end-windows-ai-developer-experience-using-onnx-runtime-and-olive/"
		>Unlocking the end-to-end Windows AI developer experience using ONNX Runtime and Olive →</a
	>
	<a
		class="btn btn-primary"
		href="https://blogs.windows.com/windowsdeveloper/2023/05/23/bringing-the-power-of-ai-to-windows-11-unlocking-a-new-era-of-productivity-for-customers-and-developers-with-windows-copilot-and-dev-home/"
		>Bringing the power of AI to Windows 11 →</a
	>
</div>
<div class="container mx-auto px-10 my-10">
	<h1 class="text-4xl pb-4">Examples</h1>
	<div class="grid gap-10 grid-cols-1 md:grid-cols-2">
		<div class="card bg-base-300">
			<div class="card-body">
				<h2 class="card-title">Windows ML Samples Gallery</h2>
				<p>
					This gallery demonstrates different machine learning scenarios and features using <a
						class="text-blue-500"
						href="https://docs.microsoft.com/en-us/windows/ai/windows-ml/">Windows ML</a
					>
					in an interactive format. The app is an interactive companion that shows the integration
					of
					<a
						class="text-blue-500"
						href="https://docs.microsoft.com/en-us/uwp/api/windows.ai.machinelearning"
						>Windows Machine Learning Library APIs</a
					>
					into a desktop
					<a
						class="text-blue-500"
						href="https://docs.microsoft.com/en-us/uwp/api/windows.ai.machinelearning">WinUI 3</a
					> application.
				</p>
				<div class="card-actions">
					<a
						href="https://github.com/microsoft/Windows-Machine-Learning/tree/master/Samples/WinMLSamplesGallery"
						class="btn btn-primary hidden md:grid">Check out the Windows ML Sample Gallery →</a
					>
					<a
						href="https://github.com/microsoft/Windows-Machine-Learning/tree/master/Samples/WinMLSamplesGallery"
						class="btn btn-primary md:hidden">Windows ML Sample Gallery →</a
					>
				</div>
			</div>
		</div>
		<div class="card bg-base-300">
			<div class="card-body">
				<h2 class="card-title">SqueezeNet Object Detection</h2>
				<p>
					This is a desktop application that uses SqueezeNet, a pre-trained machine learning model,
					to detect the predominant object in an image selected by the user from a file.
				</p>
				<div class="card-actions">
					<a
						href="https://github.com/microsoft/Windows-Machine-Learning/tree/master/Samples/WinMLSamplesGallery"
						class="btn btn-primary">Check out SqueezeNet →</a
					>
				</div>
			</div>
		</div>
	</div>
</div>
