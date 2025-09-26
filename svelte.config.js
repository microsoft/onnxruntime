import adapter from '@sveltejs/adapter-auto';
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
		// adapter-auto supports multiple environments and can handle both static and dynamic routes
		adapter: adapter(),
		paths: {
			base: process.env.NODE_ENV === 'production' ? '' : ''
		},
		prerender: {
			// Enable prerendering for most of the site
			handleMissingId: 'warn',
			handleHttpError: 'warn'
		}
	}
};

export default config;
