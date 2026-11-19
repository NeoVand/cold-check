<script lang="ts">
	import type { Snippet } from 'svelte';
	import { GLOSSARY } from '#lib/glossary.ts';

	let { term, children }: { term: keyof typeof GLOSSARY; children: Snippet } = $props();
	const entry = $derived(GLOSSARY[term]);
	let open = $state(false);
	let shift = $state(0);
	function place(anchor: HTMLElement) {
		const r = anchor.getBoundingClientRect();
		const w = Math.min(300, window.innerWidth - 24);
		const centre = r.left + r.width / 2;
		const left = Math.max(12, Math.min(window.innerWidth - 12 - w, centre - w / 2));
		shift = left - r.left;
	}
	function show(e: Event) {
		place(e.currentTarget as HTMLElement);
		open = true;
	}
</script>

<span
	class="gloss"
	role="button"
	tabindex="0"
	aria-expanded={open}
	onmouseenter={show}
	onmouseleave={() => (open = false)}
	onfocus={show}
	onblur={() => (open = false)}
	onclick={(e) => (open ? (open = false) : show(e))}
	onkeydown={(e) => e.key === 'Escape' && (open = false)}
	>{@render children()}{#if open}<span class="pop" role="tooltip" style:left="{shift}px"
			><span class="t">{entry.title}</span>{entry.text}</span
		>{/if}</span
>

<style>
	.gloss {
		position: relative;
		text-decoration: underline dotted;
		text-decoration-color: var(--color-blue-grey);
		text-underline-offset: 3px;
		text-decoration-thickness: 1px;
		cursor: help;
	}
	.pop {
		position: absolute;
		top: calc(100% + 8px);
		z-index: 20;
		width: min(300px, calc(100vw - 24px));
		padding: 12px 14px;
		border: 1px solid var(--color-edge);
		border-radius: 12px;
		background: var(--color-paper);
		box-shadow: 0 8px 24px rgb(31 42 68 / 0.08);
		color: var(--color-ink-soft);
		font-size: 13.5px;
		line-height: 1.5;
		text-align: left;
		hyphens: none;
		font-weight: 400;
		animation: rise 180ms var(--ease);
	}
	.t {
		display: block;
		margin-bottom: 3px;
		color: var(--color-ink);
		font-weight: 600;
	}
	@keyframes rise {
		from {
			opacity: 0;
			transform: translateY(-3px);
		}
	}
</style>
