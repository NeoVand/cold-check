// One card at desktop and phone widths: node scripts/shot_card.mjs <card index> <name> [url]
import { createRequire } from 'node:module';
// Playwright is not a dependency of the page: install it anywhere and point PLAYWRIGHT_FROM at that package.json.
const require = createRequire(process.env.PLAYWRIGHT_FROM ?? import.meta.url);
const { chromium } = require('playwright');
const [k, name, url = 'http://127.0.0.1:5195/'] = [Number(process.argv[2]), process.argv[3], process.argv[4]];
const out = new URL('../shots/', import.meta.url).pathname;
const b = await chromium.launch();
for (const [dev, vp, dsf] of [
	['desktop', { width: 1440, height: 900 }, 2],
	['phone', { width: 390, height: 844 }, 3]
]) {
	const p = await b.newPage({ viewport: vp, deviceScaleFactor: dsf });
	const errors = [];
	p.on('pageerror', (e) => errors.push(String(e)));
	await p.goto(url, { waitUntil: 'networkidle' });
	const c = p.locator('figure.card').nth(k);
	await c.scrollIntoViewIfNeeded();
	await new Promise((r) => setTimeout(r, 800));
	await c.screenshot({ path: `${out}${dev}-${name}.png` });
	const over = await p.evaluate(() => [...document.querySelectorAll('.scroll')].map((e) => e.scrollWidth - e.clientWidth));
	console.log(dev, errors.length ? errors : 'no page errors', 'table overflow px', JSON.stringify(over));
	await p.close();
}
await b.close();
