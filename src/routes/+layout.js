export const prerender = 'auto'; // Allow some pages to be dynamic
export const load = ({ url }) => {
	const { pathname } = url;

	return {
		pathname
	};
};
