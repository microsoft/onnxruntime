# ONNX Runtime Website

This directory contains website source files that are maintained in the main repository.

## Accessibility Fix

Fixed dark mode link visibility issue in `src/routes/components/hero.svelte`:
- Changed hardcoded `text-blue-800` to responsive `text-blue-700 dark:text-blue-300`
- Added hover states: `hover:text-blue-600 dark:hover:text-blue-200`
- Added smooth transitions: `transition-colors`

## Color Contrast Analysis

### Before (failing):
- Dark mode: `#1e40af` (text-blue-800) on `#212933` background
- Contrast ratio: 1.37:1 (fails WCAG 4.5:1 requirement)

### After (fixed):
- Light mode: `#1d4ed8` (text-blue-700) on `#f3f4f6` background
- Dark mode: `#93c5fd` (text-blue-300) on `#212933` background
- Both should exceed WCAG 4.5:1 contrast requirement

## Deployment

The website is built from the `gh-pages` branch. This fix should be applied there as well.