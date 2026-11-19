<script lang="ts">
	let { phase, onclick }: { phase: 'idle' | 'running' | 'done'; onclick: () => void } = $props();
</script>

<button class="pill small" {onclick} aria-label={phase === 'running' ? 'Pause' : phase === 'done' ? 'Replay' : 'Play'}>
	{#if phase === 'running'}
		<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"
			><rect x="2.5" y="2" width="2.4" height="8" rx="0.6" fill="currentColor" /><rect x="7.1" y="2" width="2.4" height="8" rx="0.6" fill="currentColor" /></svg
		>
		Pause
	{:else if phase === 'done'}
		<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"
			><path d="M9.6 6a3.6 3.6 0 1 1-1.05-2.55M8.9 1.6v2.2H6.7" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round" /></svg
		>
		Replay
	{:else}
		<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 1.8v8.4L10 6z" fill="currentColor" /></svg>
		Play
	{/if}
</button>
