<script lang="ts">
	import anime from 'animejs';
	export let title: string;
	export let description: string;
	export let date: string;
	export let link: string;
	let handleEnter = (e: any) => {
		anime({
			targets: e.target,
			scale: 1.05,
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
	class="mb-4 bg-base-300 border rounded-md border-primary"
>
	<a href={link}>
		<div class="card h-full">
			<div class="card-body">
				<h2 class="card-title">{title}</h2>
				<p>{description}</p>
				<img src={image}/>
				<div class="text-right text-blue-500">
					{date}
				</div>
			</div>
		</div>
	</a>
</article>
