<script lang="ts">
	import './layout.css';
	import type { Snippet } from 'svelte';
	let { children }: { children: Snippet } = $props();
</script>

<svelte:head>
	<meta name="theme-color" content="#ffffff" />
</svelte:head>

{@render children()}
