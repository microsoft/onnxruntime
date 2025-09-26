import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/kit/vite';
import { mdsvex } from 'mdsvex';
import relativeImages from 'mdsvex-relative-images';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	// Consult https://kit.svelte.dev/docs/integrations#preprocessors
	// for more information about preprocessors
	extensions: ['.svelte', '.md', '.svx'],
	preprocess: [
		vitePreprocess(),
		mdsvex({
			extensions: ['.md', '.svx'],
			layout: {
				blogs: 'src/routes/blogs/post.svelte'
			},
			remarkPlugins: [relativeImages]
		})
	],

	kit: {
		// Use adapter-static for GitHub Pages deployment
		adapter: adapter({
			// Output to build directory for GitHub Pages
			pages: 'build',
			assets: 'build',
			fallback: '404.html', // GitHub Pages SPA fallback
			precompress: false,
			strict: false // Allow dynamic routes to be skipped
		}),
		paths: {
			base: process.env.NODE_ENV === 'production' ? '' : ''
		},
		prerender: {
			// Handle missing IDs gracefully for dynamic routes
			handleMissingId: 'warn',
			handleHttpError: 'warn',
			// Crawl all static pages but handle dynamic routes client-side
			crawl: true
		}
	}
};

export default config;
