<script lang="ts">
	import type { Snippet } from 'svelte';
	let { title, children }: { title: string; children: Snippet } = $props();
	let open = $state(false);
	const id = $props.id();
</script>

<div class={['fold', { open }]}>
	<button class="head" aria-expanded={open} aria-controls={id} onclick={() => (open = !open)}>
		<span>{title}</span>
		<svg class="chev" width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"
			><path d="M3.5 5.5 7 9l3.5-3.5" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round" /></svg
		>
	</button>
	<div class="body" {id}>
		<div class="inner">
			<div class="pad">{@render children()}</div>
		</div>
	</div>
</div>

<style>
	.fold {
		margin: 12px 0;
		border-radius: 16px;
		background: var(--color-tint);
	}
	.head {
		display: flex;
		width: 100%;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
		padding: 14px 20px;
		border: 0;
		background: none;
		color: var(--color-ink-soft);
		font: inherit;
		font-size: 15.5px;
		text-align: left;
		cursor: pointer;
		transition: color var(--hover) ease;
	}
	.head:hover,
	.open .head {
		color: var(--color-ink);
	}
	.chev {
		flex: none;
		transition: transform 300ms var(--ease);
	}
	.open .chev {
		transform: rotate(180deg);
	}
	.body {
		display: grid;
		grid-template-rows: 0fr;
		transition: grid-template-rows 380ms var(--ease);
	}
	.open .body {
		grid-template-rows: 1fr;
	}
	.inner {
		overflow: hidden;
	}
	.pad {
		padding: 0 20px 18px;
	}
	.pad :global(p) {
		font-size: 15.5px !important;
		line-height: 1.6 !important;
	}
</style>
