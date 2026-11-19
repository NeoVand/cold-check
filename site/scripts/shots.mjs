// Screenshots of the page at desktop and phone widths, with each demo run to its end.
// node scripts/shots.mjs [url]   (needs Playwright; writes shots/*.png)
import { createRequire } from 'node:module';
import { mkdirSync } from 'node:fs';
// Playwright is not a dependency of the page: install it anywhere and point PLAYWRIGHT_FROM at that package.json.
const require = createRequire(process.env.PLAYWRIGHT_FROM ?? import.meta.url);
const { chromium } = require('playwright');

const url = process.argv[2] ?? 'http://127.0.0.1:5195/';
const out = new URL('../shots/', import.meta.url).pathname;
mkdirSync(out, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await chromium.launch();
for (const [name, vp, dsf] of [
	['desktop', { width: 1440, height: 900 }, 2],
	['phone', { width: 390, height: 844 }, 3]
]) {
	const page = await browser.newPage({ viewport: vp, deviceScaleFactor: dsf });
	const errors = [];
	page.on('pageerror', (e) => errors.push(String(e)));
	page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
	await page.goto(url, { waitUntil: 'networkidle' });
	await sleep(7200); // the hero's boards filled
	await page.screenshot({ path: `${out}${name}-01-top.png` });
	const cards = page.locator('figure.card');
	const run = async (k, choice, wait, file) => {
		const card = cards.nth(k);
		await card.scrollIntoViewIfNeeded();
		if (choice) await card.getByRole('button', { name: choice, exact: true }).click();
		const b = card.getByRole('button', { name: 'Play' });
		if (await b.count()) await b.first().click();
		await sleep(wait);
		await card.screenshot({ path: `${out}${name}-${file}.png` });
	};
	await run(1, null, 9000, '02-sudoku-9');
	await run(1, '16 × 16', 9000, '03-sudoku-16');
	await run(1, '25 × 25', 11000, '04-sudoku-25');
	await run(2, null, 5500, '05-maze-hard-106');
	await run(2, 'Room', 5500, '06-maze-room');
	await run(2, 'Edge', 5500, '07-maze-edge');
	await run(3, null, 9000, '08-bfs');
	await cards.nth(4).scrollIntoViewIfNeeded();
	await cards.nth(4).screenshot({ path: `${out}${name}-09-results.png` });
	await cards.nth(5).screenshot({ path: `${out}${name}-10-sizes.png` });
	if ((await cards.count()) > 6) await cards.nth(6).screenshot({ path: `${out}${name}-11-clrs.png` });
	await page.screenshot({ path: `${out}${name}-00-full.png`, fullPage: true });
	console.log(name, errors.length ? errors : 'no page errors');
	await page.close();
}
await browser.close();
