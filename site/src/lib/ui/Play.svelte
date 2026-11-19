<script lang="ts">
	import IconButton from './IconButton.svelte';
	import { PauseIcon, PlayIcon, RepeatIcon } from '@hugeicons/core-free-icons';
	let { phase, onclick }: { phase: 'idle' | 'running' | 'done'; onclick: () => void } = $props();
</script>

{#if phase === 'running'}
	<IconButton icon={PauseIcon} label="Pause" show solid {onclick} />
{:else if phase === 'done'}
	<IconButton icon={RepeatIcon} label="Run again" show {onclick} />
{:else}
	<IconButton icon={PlayIcon} label="Run" show solid {onclick} />
{/if}
