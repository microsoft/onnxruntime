<script>
	import Blog from '../components/blog-post.svelte';
	import FeaturedBlog from '../components/blog-post-featured.svelte';
	// TODO: Add props for blog
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
