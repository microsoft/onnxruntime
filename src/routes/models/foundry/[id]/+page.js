// Configure the dynamic route to disable prerendering
// This route fetches data dynamically and can't be statically prerendered
// since we don't know all possible model IDs at build time
export const prerender = false;
export const ssr = false; // Client-side rendering only for this dynamic route
