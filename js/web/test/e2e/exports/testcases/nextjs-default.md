## nextjs-default

### Summary

This is a Next.js application created by `npx create-next-app@latest` using the following config:

```
<ORT_ROOT>\js\web\test\e2e\exports\testcases>npx create-next-app@latest

√ What is your project named? ... nextjs-default
√ Would you like to use TypeScript? ... No
√ Would you like to use ESLint? ... No
√ Would you like to use Tailwind CSS? ... No
√ Would you like your code inside a `src/` directory? ... No
√ Would you like to use App Router? (recommended) ... No
√ Would you like to use Turbopack for `next dev`? ... No
√ Would you like to customize the import alias (`@/*` by default)? ... No
Creating a new Next.js app in <ORT_ROOT>\js\web\test\e2e\exports\testcases\nextjs-default.

Using npm.

Initializing project with template: app


Installing dependencies:
- react
- react-dom
- next
```

Small changes were made based on the application template, including:

- Remove default Logos, images, CSS and SVG
- Add a client side rendering (CSR) component which contains:
  - a checkbox for multi-thread
  - a checkbox for proxy
  - a "Load Model" button
  - a "Run Model" button
  - a state DIV
  - a log DIV
- Add a helper module for creating ORT session and run

### Tests

Uses puppeteer to simulate the following tests:

- Tests on `npm run dev` (dev server)
  - multi-thread OFF, proxy OFF
  - multi-thread OFF, proxy ON
  - multi-thread ON, proxy OFF
  - multi-thread ON, proxy ON
- Tests on `npm run dev -- --turbopack` (dev server using TurboPack)
  - multi-thread OFF, proxy OFF
  - multi-thread OFF, proxy ON
  - multi-thread ON, proxy OFF
  - multi-thread ON, proxy ON
- Tests on `npm run build` + `npm run start` (prod)
  - multi-thread OFF, proxy OFF
  - multi-thread OFF, proxy ON
  - multi-thread ON, proxy OFF
  - multi-thread ON, proxy ON
