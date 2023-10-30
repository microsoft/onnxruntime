<script>
	import Ortperf from '../../images/undraw/image_ortperf.svelte';
</script>

<div class="container mx-auto px-10 mb-10">
	<div class="grid grid-cols-1 md:grid-cols-2 gap-10">
		<div class="">
			<h1 class="text-4xl">Performance</h1>
			<br /><br />
			<p class="text-xl">
				CPU, GPU, NPU - no matter what hardware you run on, ONNX Runtime optimizes for latency, throughput, memory utilization, and binary size. In addition to excellent out-of-the-box performance for common usage patterns, additional
				<a href="https://onnxruntime.ai/docs/performance/" class="text-blue-500">model optimization techniques</a> and runtime configurations are available to further improve performance for specific use cases and models.
			</p>
		</div>
		<div class="m-auto overflow:hidden">
			<Ortperf />
		</div>
	</div>
</div>
