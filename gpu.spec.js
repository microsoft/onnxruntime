import { test, expect } from '@playwright/test';

test('It should take a snapshot of the GPU Chrome page', async ({ page }) => {
    await page.goto('chrome://gpu', { waitUntil: 'domcontentloaded' });
    await page.screenshot({ path: 'gpu.png', fullPage: true });
    await expect(page.locator('text=Graphics Feature Status').first()).toBeVisible();
});