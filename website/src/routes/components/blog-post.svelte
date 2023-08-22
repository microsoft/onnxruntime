<script lang="ts">
	import anime from 'animejs';
	export let title: string;
	export let description: string;
	export let badges: Array<Array<string>>;
	export let imgsrc: string;
	export let imgalt: string;
	let handleEnter = (e: any) => {
		anime({
			targets: e.target,
			scale: 1.02,
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

<article on:mouseenter={handleEnter} on:mouseleave={handleLeave} class="m-4 md:m-10">
	<a href="http://blog.com">
		<div class="card card-side bg-base-100 border border-primary">
			<div class="card-body">
				<h2 class="card-title">{title}</h2>
				<p>{description}</p>
				<div class="card-actions">
					{#each badges as badge}
						<div class="badge badge-{badge[1]}">{badge[0]}</div>
					{/each}
				</div>
			</div>
			<figure >
				<img class="hidden md:inline" src={imgsrc} alt={imgalt} />
			</figure>
		</div>
	</a>
</article>
