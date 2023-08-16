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
    <img on:mouseenter={handleEnter} on:mouseleave={handleLeave} class="mx-auto" src="https://placehold.co/600x300" alt="" />
		<div class="">
			<h1 class="text-4xl">Cross-Platform</h1><br/><br/>
			<p class="text-xl">
				Take advantage of the benefits of ONNX Runtime without changing your technology stack. Access ONNX RunTime using your preferred API - C#, C++, C, Python, or Java. Support for Linux, Windows, and Mac allows you to build and deploy applications without worry.
			</p>
		</div>
	</div>
</div>
