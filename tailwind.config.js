/** @type {import('tailwindcss').Config} */
export default {
	content: ['./src/**/*.{html,svelte,js,ts}'],
	theme: {
		extend: {}
	},
	plugins: [require('daisyui')],
	daisyui: {
		themes: [
			{
				darkmode: {
					...require('daisyui/src/theming/themes')['[data-theme=business]'],
					primary: '#0099cc',
					'base-100': '#212933',
					info: '#d1d1d1',
				},
				lightmode: {
					...require('daisyui/src/theming/themes')['[data-theme=corporate]'],
					primary: '#80dfff',
					'base-100': '#f3f4f6',
					info: '#d1d1d1',
				}
			}
		],
		base: true,
		styled: true,
		utils: true
	}
};
