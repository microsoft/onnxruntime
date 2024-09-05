import { d } from 'svelte-highlight/languages';
const defaultTheme = require('tailwindcss/defaultTheme')
import flattenColorPalette from 'tailwindcss/lib/util/flattenColorPalette';

/** @type {import('tailwindcss').Config} */
export default {
	darkMode: ['selector', '[data-theme=" darkmode"]'],
	content: ['./src/**/*.{html,svelte,js,ts}'],
	theme: {
		extend: {
			animation:{
				scroll: 
				'scroll var(--animation-duration, 40s) var(--animation-direction, forwards) linear infinite'
			},
			keyframes:{
				scroll: {
					to: {
						transform: 'translate(calc(-50% - 0.5rem))'
					}
				}
			}
		},
		screens: {
			'xs': '360px',
			...defaultTheme.screens
		}
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

function addVariablesForColors({ addBase, theme }) {
	let allColors = flattenColorPalette(theme('colors'));
	let newVars = Object.fromEntries(
		Object.entries(allColors).map(([key, val]) => [`--${key}`, val])
	);

	addBase({
		':root': newVars
	});
}