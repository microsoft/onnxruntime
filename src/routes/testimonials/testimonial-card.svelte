<script lang="ts">
	export let title: string;
	export let description: string;
	export let imgsrc: string;
	export let imgalt: string;
	export let author: string;
	import anime from 'animejs';
	let handleEnter = (e: any) => {
		anime({
			targets: e.target,
			scale: 1.04,
			duration: 1500
		});
	};
	let handleLeave = (e: any) => {
		anime({
			targets: e.target,
			scale: 1,
			duration: 1500
		});
	};
</script>

<article
	on:mouseenter={handleEnter}
	on:mouseleave={handleLeave}
	class="max-w-md mx-auto bg-blue-100 rounded-sm overflow-hidden md:max-w-2xl"
	id={title}
>
	<div class="md:flex">
		<div class="md:shrink-0 my-auto">
			<img class="md:h-48 w-full hidden md:flex" src={imgsrc} alt={imgalt} />
		</div>
		<div class="p-8">
			<p class="block mt-1 leading-tight font-bold text-neutral text-lg">{title}</p>
			<p class="mt-2 text-neutral">{description}</p>
			<br />
			<p class="text-blue-500 text-right">-{author}</p>
		</div>
	</div>
</article>
