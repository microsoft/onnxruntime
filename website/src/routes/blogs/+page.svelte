<script>
	import Blog from '../components/blog-post.svelte';
	import FeaturedBlog from '../components/blog-post-featured.svelte';
	import anime from 'animejs';
	import { onMount } from 'svelte';
	// TODO: Add props for blog
	onMount(() => {
		anime({
			targets: '.card',
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
</script>

<div class="container px-20 mx-auto">
	<h1 class="text-5xl">Blogs & Announcements</h1>

	<div class="pt-5">
		<h3 class="text-3xl">Featured posts</h3>
		<div class="grid grid-cols-3">
			<FeaturedBlog />
			<FeaturedBlog />
			<FeaturedBlog />
		</div>
	</div>
	<h3 class="text-3xl">Recent posts</h3>
	<Blog />
</div>
