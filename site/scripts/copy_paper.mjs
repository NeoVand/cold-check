// Copies the built paper (../paper/main.pdf) into static/paper/ so the page serves it; runs before every build.
import { copyFileSync, existsSync, mkdirSync } from 'node:fs';
const src = new URL('../../paper/main.pdf', import.meta.url);
const dest = new URL('../static/paper/', import.meta.url);
if (!existsSync(src)) throw new Error('paper/main.pdf is missing: build the paper first');
mkdirSync(dest, { recursive: true });
copyFileSync(src, new URL('main.pdf', dest));
console.log('copied paper/main.pdf into static/paper/');
