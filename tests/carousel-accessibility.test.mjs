import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const expectedCustomerNames = [
	'Adobe',
	'AMD',
	'Ant Group',
	'Algoriddim',
	'ATLAS',
	'Autodesk',
	'Bazaarvoice',
	'Camo',
	'Cephable',
	'ClearBlade',
	'Deezer',
	'GoodNotes',
	'Graiphic',
	'Hugging Face',
	'Hypefactors',
	'InFarm',
	'Intel',
	'Intelligenza Etica',
	'Navitaire',
	'NVIDIA',
	'Apache OpenNLP',
	'Oracle',
	'Peakspeed',
	'Pieces',
	'PTW Dosimetry',
	'Redis',
	'Rockchip',
	'Samtec',
	'SAS',
	'Teradata',
	'Topaz Labs',
	'Unreal Engine',
	'USDA',
	'Vespa',
	'Writer',
	'Xilinx'
];

const html = await readFile(new URL('../build/index.html', import.meta.url), 'utf8');
const carousel = html.match(/<div[^>]*data-customer-carousel[^>]*>([\s\S]*?)<\/ul>/)?.[1];

test('homepage carousel renders one semantic customer-link set', () => {
	assert.ok(carousel, 'customer carousel should be present in the rendered homepage');

	const originals = [...carousel.matchAll(/<li[^>]*data-carousel-original[^>]*>([\s\S]*?)<\/li>/g)];
	const names = originals.map(([, item]) => item.match(/<img[^>]*alt="([^"]+)"/)?.[1]);
	const tabbableLinks = originals.filter(
		([, item]) => /<a href=/.test(item) && !/tabindex="-1"/.test(item)
	);

	assert.deepEqual(names, expectedCustomerNames);
	assert.equal(tabbableLinks.length, expectedCustomerNames.length);
});

test('customer focus rings render inside their clipped border boxes', () => {
	assert.ok(carousel, 'customer carousel should be present in the rendered homepage');

	const originals = [...carousel.matchAll(/<li[^>]*data-carousel-original[^>]*>([\s\S]*?)<\/li>/g)];

	for (const [, item] of originals) {
		assert.match(item, /focus-visible:ring-inset/);
	}
});

test('visual carousel copies are hidden and excluded from sequential focus', () => {
	assert.ok(carousel, 'customer carousel should be present in the rendered homepage');

	const copies = [
		...carousel.matchAll(/<li[^>]*aria-hidden="true"[^>]*data-carousel-copy[^>]*>([\s\S]*?)<\/li>/g)
	];

	assert.equal(copies.length, expectedCustomerNames.length);
	for (const [, item] of copies) {
		assert.match(item, /<a[^>]*tabindex="-1"/);
		assert.match(item, /<img[^>]*alt=""/);
	}
});
