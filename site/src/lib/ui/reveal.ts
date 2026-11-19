/** Things build up as they enter the view: a rise and fade, once. Without JavaScript, or with reduced motion, nothing hides. */
import type { Attachment } from 'svelte/attachments';
import { reducedMotion } from '#lib/theme.ts';

export const reveal: Attachment<HTMLElement> = (el) => {
	if (reducedMotion() || typeof IntersectionObserver === 'undefined') return;
	const r = el.getBoundingClientRect();
	if (r.top < window.innerHeight * 0.92) return; // already on the first screen: no hide-and-show
	el.classList.add('rv');
	const io = new IntersectionObserver(
		(es) => {
			for (const e of es)
				if (e.isIntersecting) {
					el.classList.add('rv-in');
					io.disconnect();
				}
		},
		{ rootMargin: '0px 0px -8% 0px' }
	);
	io.observe(el);
	return () => io.disconnect();
};

/** Calls `fn` once, when `el` first comes into view (at once if it is already there). Returns the cleanup. */
export function whenVisible(el: HTMLElement, fn: () => void): () => void {
	if (typeof IntersectionObserver === 'undefined') {
		fn();
		return () => {};
	}
	const io = new IntersectionObserver(
		(es) => {
			if (es.some((e) => e.isIntersecting)) {
				io.disconnect();
				fn();
			}
		},
		{ threshold: 0.35 }
	);
	io.observe(el);
	return () => io.disconnect();
}
