<script>
	import onnximage from '../../images/ONNX-Icon.svg';
	import { onMount, beforeUpdate } from 'svelte';
	import anime from 'animejs';
	import { text } from '@sveltejs/kit';
	onMount(() => {
		anime
			.timeline({ loop: false })
			.add({
				targets: '.img',
				easing: 'spring',
        rotate: '1turn',
				duration: 1000,
			})
	});
</script>

<div class="hero bg-primary">
	<div class="hero-content my-20">
		<div class="flex">
			<div class="basis-2/3 self-center mr-20">
				<h1 class="text-5xl ">Cross-platform accelerated <br/>machine learning</h1>
				<p class="py-6">
					Built-in optimizations speed up training and inferencing with your existing technology
					stack.
				</p>
				<button class="btn btn-success">Get Started →</button>
			</div>
			<img src={onnximage} class="img mx-auto basis-1/3" alt="ONNX Runtime logo" />
		</div>
	</div>
</div>

<style>
	.img {
		transform: scale(0.75);
	}
</style>
