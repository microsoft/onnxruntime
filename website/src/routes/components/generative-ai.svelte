<script lang="ts">
	import anime from "animejs";
	let handleEnter = (e: any) => {
		anime({
			targets: e.target,
			scale: 1.05,
			duration: 1500,
		})
	}
	let handleLeave = (e: any) => {
		anime({
			targets: e.target,
			scale: 1,
			duration: 1500,
		})
	}
</script>
<div class="container mx-auto px-10 mb-10">
	<div class="grid grid-cols-1 md:grid-cols-2 gap-10">
		<div class="">
			<h1 class="text-4xl">Generative AI</h1><br/><br/>
			<p class="text-xl">
				Integrate the power of generative AI in your apps and services with ONNX Runtime. Broad platform support and deep optimizations enable you to 
benefit from state of the art models for image synthesis, text generation, and more
			</p>
		</div>
    <img on:mouseenter={handleEnter} on:mouseleave={handleLeave} class="mx-auto" src="https://placehold.co/600x200" alt="" />
	</div>
</div>
