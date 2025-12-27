export const prerender = true; // Fully static site
export const load = ({ url }) => {
	const { pathname } = url;

	return {
		pathname
	};
};
