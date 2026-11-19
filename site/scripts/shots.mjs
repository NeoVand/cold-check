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
	// figures in order: 0 hero, 1 architecture, 2 Sudoku, 3 maze, 4 BFS figure, 5 BFS live, 6 ledger, 7 table 1, 8 table 2, 9 CLRS-30
	const figs = page.locator('figure.fig');
	const shot = async (k, file, choice = null, wait = 0) => {
		const f = figs.nth(k);
		await f.scrollIntoViewIfNeeded();
		await sleep(900); // revealed; a live demo starts as it comes into view
		if (choice) {
			await f.getByRole('button', { name: choice, exact: true }).click();
			await f.getByRole('button', { name: 'Run', exact: true }).click();
		}
		await sleep(wait);
		await f.screenshot({ path: `${out}${name}-${file}.png` });
	};
	await shot(1, '02-architecture');
	await shot(2, '03-sudoku-9', null, 9000);
	await shot(2, '04-sudoku-16', '16 × 16', 9000);
	await shot(2, '05-sudoku-25', '25 × 25', 11000);
	await shot(3, '06-maze-hard-106', null, 5500);
	await shot(3, '07-maze-room', 'Room', 5500);
	await shot(3, '08-maze-edge', 'Edge', 5500);
	await shot(4, '09-bfs-figure');
	await shot(5, '10-bfs', null, 9000);
	const ledger = figs.nth(6);
	await shot(6, '11-ledger-sudoku', null, 8000);
	if (name === 'desktop') {
		await ledger.getByRole('button', { name: /^Tree mazes, no ring/ }).click();
		await sleep(6000);
		await ledger.screenshot({ path: `${out}${name}-12-ledger-maze.png` });
	}
	await shot(7, '13-results');
	await shot(8, '14-sizes');
	if ((await figs.count()) > 9) await shot(9, '15-clrs');
	await page.evaluate(() => window.scrollTo(0, 0));
	await page.screenshot({ path: `${out}${name}-00-full.png`, fullPage: true });
	console.log(name, errors.length ? errors : 'no page errors');
	await page.close();
}
await browser.close();
