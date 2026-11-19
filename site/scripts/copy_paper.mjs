// Copies the built paper (../paper/main.pdf) and the paper figures the page shows into static/; runs before every build.
import { copyFileSync, existsSync, mkdirSync } from 'node:fs';
const src = new URL('../../paper/main.pdf', import.meta.url);
const dest = new URL('../static/paper/', import.meta.url);
if (!existsSync(src)) throw new Error('paper/main.pdf is missing: build the paper first');
mkdirSync(dest, { recursive: true });
copyFileSync(src, new URL('main.pdf', dest));
const figs = new URL('../static/figures/', import.meta.url);
mkdirSync(figs, { recursive: true });
for (const f of ['fig_arch.svg', 'fig1.svg']) {
	const s = new URL(`../../paper/figures/v1/out/${f}`, import.meta.url);
	if (!existsSync(s)) throw new Error(`paper/figures/v1/out/${f} is missing: build the figures first`);
	copyFileSync(s, new URL(f, figs));
}
console.log('copied paper/main.pdf and the figures into static/');
