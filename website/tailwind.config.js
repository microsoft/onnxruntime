/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{html,svelte,js,ts}'],
  theme: {
    extend: {},
  },
  plugins: [require("daisyui")],
  daisyui: {
    themes: ["corporate", "business", {
      darkmode: {
        ...require("daisyui/src/theming/themes")["[data-theme=business]"],
        "primary": "#b2b2b2",
        "secondary": "#fcfcfc",
        "accent": "#64c0fe",
        "neutral": "#d1d1d1",
        "base-100": "#212933",
        "info": "#1d4ed8",
        "success": "#818cf8",
        "warning": "#facc15",
        "error": "#ef4444",
       },
       lightmode: {
        ...require("daisyui/src/theming/themes")["[data-theme=corporate]"],
        "primary": "#b2b2b2",
        "secondary": "#fcfcfc",
        "accent": "#64c0fe",
        "neutral": "#d1d1d1",
        "base-100": "#212933",
        "info": "#1d4ed8",
        "success": "#818cf8",
        "warning": "#facc15",
        "error": "#ef4444",
       },
    }],
    base: true,
    styled: true,
    utils: true,
  },
}

