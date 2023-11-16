---
nav_exclude: true
---
## Developing

Once you've installed dependencies with `npm install` (or  `yarn`), start a development server with hot-reload enabled:

```bash
npm run dev

# or start the server and open the app in a new browser tab
npm run dev -- --open
```
All working pages are in `src/routes/[page url]/+page.svelte`, which is where you can make your edits.

### Technologies & relevant docs
Please use the docs pages below to aid in your development process. As a general target, we should be using zero CSS, as daisyUI (framework with components) and tailwindcss (css classes) should be able to handle all of our styling needs.
- [Svelte](https://svelte.dev/)
- daisyUI [docs](https://daisyui.com/)
- tailwindcss [docs](https://tailwindcss.com/docs)


## Building

To create a production version of your app:

```bash
npm run build
```

You can preview the production build with `npm run preview`.

> To deploy your app, you may need to install an [adapter](https://kit.svelte.dev/docs/adapters) for your target environment.
