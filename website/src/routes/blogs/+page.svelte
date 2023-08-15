<script>
	// @ts-nocheck

	import Blog from '../components/blog-post.svelte';
	import FeaturedBlog from '../components/blog-post-featured.svelte';
	import anime from 'animejs';
	import { onMount } from 'svelte';
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
	let badges = [
		['ONNX', 'primary'],
		['ML', 'primary']
	];
	let imgsrc = "https://source.unsplash.com/random/400x300"
	let imgalt = "image alt"
</script>

<div class="container px-20 mx-auto">
	<h1 class="text-5xl">Blogs & Announcements</h1>

	<div class="pt-5">
		<h3 class="text-3xl">Featured posts</h3>
		<div class="grid grid-cols-3">
			<FeaturedBlog title="test" description="test" badges={badges} {imgsrc} {imgalt}/>
			<FeaturedBlog title="test" description="test" badges={badges} {imgsrc} {imgalt}/>
			<FeaturedBlog title="test" description="test" badges={badges} {imgsrc} {imgalt}/>
		</div>
	</div>
	<h3 class="text-3xl">Recent posts</h3>
	<Blog title="test" description="test" badges={badges} {imgsrc} {imgalt} />
</div>
