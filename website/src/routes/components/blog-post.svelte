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
	let handleEnter = (/** @type {{ target: any; }} */ e) => {
		anime({
			targets: e.target,
			scale: 1.02,
			duration: 1500,
		})
	}
	let handleLeave = (/** @type {{ target: any; }} */ e) => {
		anime({
			targets: e.target,
			scale: 1,
			duration: 1500,
		})
	}
</script>

<article on:mouseenter={handleEnter} on:mouseleave={handleLeave}  class="m-4 md:m-10">
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
			<figure>
				<img src={imgsrc} alt={imgalt} />
			</figure>
		</div>
	</a>
</article>
