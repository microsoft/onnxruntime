<script lang="ts">
	import anime from 'animejs';

	import { onMount } from 'svelte';
	onMount(() => {
		anime({
			targets: '.partner',
			translateY: -20,
			direction: 'alternate',
			loop: false,
			delay: function (el, i, l) {
				return 500 + i * 50;
			},
			endDelay: function (el, i, l) {
				return (l - i) * 50;
			}
		});
	});

	let handleEnter = (e:any) => {
		anime({
			targets: e.target,
			scale: 1.1,
			duration: 1500,
		})
	}
	let handleLeave = (e:any) => {
		anime({
			targets: e.target,
			scale: 1,
			duration: 1500,
		})
	}
</script>

<div class="container mx-auto lg:my-20">
	<h1 class="text-3xl my-5 text-center">Trusted By </h1>
	<div class="divider" />
	<div
		class="grid lg:grid-cols-8 md:grid-cols-4 grid-cols-2 gap-2 mx-auto justify-items-center pt-10"
	>
		<button on:mouseenter={handleEnter} on:mouseleave={handleLeave} class="partner">
			<img src="https://onnxruntime.ai/images/logos/amd-logo.png" alt="AMD" />
		</button>
		<button on:mouseenter={handleEnter} on:mouseleave={handleLeave} class="partner">
			<img src="https://onnxruntime.ai/images/logos/huggingface-logo.png" alt="HuggingFace" />
		</button>
		<button on:mouseenter={handleEnter} on:mouseleave={handleLeave} class="partner">
			<img src="https://onnxruntime.ai/images/logos/intel-logo.png" alt="Intel" />
		</button>
		<button on:mouseenter={handleEnter} on:mouseleave={handleLeave} class="partner">
			<img src="https://onnxruntime.ai/images/logos/nvidia.png" alt="Nvidia" />
		</button>
		<button on:mouseenter={handleEnter} on:mouseleave={handleLeave} class="partner">
			<img src="https://onnxruntime.ai/images/logos/adobe-logo.png" alt="Adobe" />
		</button>
		<button on:mouseenter={handleEnter} on:mouseleave={handleLeave} class="partner">
			<img src="https://onnxruntime.ai/images/logos/visual-studio-logo.png" alt="Microsoft" />
		</button>
		<button on:mouseenter={handleEnter} on:mouseleave={handleLeave} class="partner">
			<img src="https://onnxruntime.ai/images/logos/oracle-logo.png" alt="Oracle" />
		</button>
		<a on:mouseenter={handleEnter} on:mouseleave={handleLeave} href="/testimonials" class="partner">
			<p class="underline text-center pt-20">...and many more</p>
		</a>
	</div>
</div>
