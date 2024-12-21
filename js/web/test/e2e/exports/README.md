This folder includes test data, scripts and source code that used to test the export functionality of onnxruntime-web package.

## nextjs-default

### Summary

This is a Next.js application created by `npx create-next-app@latest` using the following config:

```
<ORT_ROOT>\js\web\test\e2e\exports\testcases>npx create-next-app@latest

√ What is your project named? ... nextjs-default
√ Would you like to use TypeScript? ... No / Yes
√ Would you like to use ESLint? ... No / Yes
√ Would you like to use Tailwind CSS? ... No / Yes
√ Would you like your code inside a `src/` directory? ... No / Yes
√ Would you like to use App Router? (recommended) ... No / Yes
√ Would you like to use Turbopack for `next dev`? ... No / Yes
√ Would you like to customize the import alias (`@/*` by default)? ... No / Yes
Creating a new Next.js app in <ORT_ROOT>\js\web\test\e2e\exports\testcases\nextjs-default.

Using npm.

Initializing project with template: app


Installing dependencies:
- react
- react-dom
- next
```

Small changes are made based on the application template, including:

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
- Tests on `npm run build` + `npm run serve` (prod)
  - multi-thread OFF, proxy OFF
  - multi-thread OFF, proxy ON
  - multi-thread ON, proxy OFF
  - multi-thread ON, proxy ON
