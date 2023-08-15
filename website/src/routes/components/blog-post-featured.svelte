<script>
	import anime from "animejs";
	/**
	 * @type {String}
	 */
	 export let title;
	/**
	 * @type {String}
	 */
	 export let description;
	/**
	 * @type {Array<Array<String>>}
	 */
	 export let badges;
	/**
	 * @type {String}
	 */
	 export let imgsrc;
	/**
	 * @type {String}
	 */
	 export let imgalt;
	let handleEnter = (e) => {
		anime({
			targets: e.target,
			scale: 1.05,
			duration: 1500,
		})
	}
	let handleLeave = (e) => {
		anime({
			targets: e.target,
			scale: 1,
			duration: 1500,
		})
	}
</script>

<article on:mouseenter={handleEnter} on:mouseleave={handleLeave} class="m-10">
	<a href="http://blog.com">
		<div class="card bg-base-100 border border-primary" data-svelte-h="svelte-1oufy6g">
			<figure>
				<img src={imgsrc} alt={imgalt} />
			</figure>
			<div class="card-body">
				<h2 class="card-title">{title}</h2>
				<p>{description}</p>
				<div class="card-actions">
					<div>
						{#each badges as badge}
							<div class="badge badge-{badge[1]} mr-1">{badge[0]}</div>
						{/each}
					</div>
				</div>
			</div>
		</div>
	</a>
</article>
