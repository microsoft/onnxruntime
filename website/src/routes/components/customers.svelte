<script>
	import anime from 'animejs';
  
	import { onMount } from 'svelte';
	// anime ({
	//   targets: '.partner',
	//   translateY: 10,
	//   direction: 'alternate',
	//   delay: function(el, i , l) {
	//     return i * 100
	//   },
	// })
	onMount(() => {
		anime({
			targets: '.partner',
			translateY: -20,
			direction: 'alternate',
			loop: false,
			delay: function (el, i, l) {
				return 3000 + i * 50;
			},
			endDelay: function (el, i, l) {
				return (l - i) * 50;
			}
		});
	});
</script>

<div class="container mx-auto my-20">
	<h1 class="text-3xl my-5">Trusted by:</h1>
	<div class="grid grid-cols-8 gap-2 mx-auto justify-items-center">
		<div class="partner pt-10">
			<img src="https://onnxruntime.ai/images/logos/amd-logo.png" alt="AMD" />
		</div>
		<div class="partner pt-10">
			<img src="https://onnxruntime.ai/images/logos/huggingface-logo.png" alt="HuggingFace" />
		</div>
		<div class="partner pt-10">
			<img src="https://onnxruntime.ai/images/logos/intel-logo.png" alt="Intel" />
		</div>
		<div class="partner pt-10">
			<img src="https://onnxruntime.ai/images/logos/nvidia.png" alt="Nvidia" />
		</div>
		<div class="partner pt-10">
			<img src="https://onnxruntime.ai/images/logos/adobe-logo.png" alt="Adobe" />
		</div>
		<div class="partner pt-10">
			<img src="https://onnxruntime.ai/images/logos/visual-studio-logo.png" alt="Microsoft" />
		</div>
		<div class="partner pt-10">
			<img src="https://onnxruntime.ai/images/logos/oracle-logo.png" alt="Oracle" />
		</div>
		<div class="partner pt-10">
			<img src="https://onnxruntime.ai/images/logos/redis-logo.png" alt="Redis" />
		</div>
	</div>
</div>

<style>
	.partner {
		width: 5vw;
	}
</style>
