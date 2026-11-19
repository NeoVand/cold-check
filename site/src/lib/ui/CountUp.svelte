<script lang="ts">
	/** A number that counts up once from zero, eased, when first shown. */
	import { onMount } from 'svelte';
	import { ease, reducedMotion, stopFrame } from '#lib/theme.ts';

	let { value, format, ms = 1100 }: { value: number; format: (x: number) => string; ms?: number } = $props();
	// svelte-ignore state_referenced_locally
	let shown = $state(value);
	onMount(() => {
		if (reducedMotion()) return;
		shown = 0;
		let raf = 0;
		const t0 = performance.now() + 250;
		const step = () => {
			const t = Math.max(0, Math.min(1, (performance.now() - t0) / ms));
			shown = value * ease(t);
			if (t < 1) raf = requestAnimationFrame(step);
		};
		raf = requestAnimationFrame(step);
		return () => stopFrame(raf);
	});
</script>

<span class="num">{format(shown)}</span>
