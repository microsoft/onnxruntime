<script lang="ts">
	export let videos: { title: string; link: string }[];
	export let px = 10;
	export let my = 10;
	export let title = true;
</script>

<div class="container mx-auto px-{px} my-{my}">
	{#if title == true}
		<h1 class="text-4xl">Videos</h1>
		<h1 class="text-2xl">Check out some of our videos to help you get started!</h1>
	{/if}
	<div class="grid grid-cols-1 lg:grid-cols-3 gap-8 mx-auto my-{my}">
		{#each videos as video}
			<div>
				<iframe
					class="w-full h-64 pb-2"
					src={video.link}
					title="YouTube video player"
					frameborder="0"
					allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
					allowfullscreen
				/>
				<div class="text-center">
					<a class="text-xl" href={video.link}>
						{video.title}
					</a>
				</div>
			</div>
		{/each}
	</div>
    {#if title == true}
	<a href="https://www.youtube.com/@ONNXRuntime" class="btn btn-primary mr-4">
		ONNX Runtime YouTube channel →
	</a>
    {/if}
</div>
