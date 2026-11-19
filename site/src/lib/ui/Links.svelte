<script lang="ts">
	/** The page's links: the paper, the code, and a citation that copies itself. HugeIcons, as on the cold-attention page. */
	import { HugeiconsIcon } from '@hugeicons/svelte';
	import { File02Icon, GithubIcon, QuoteDownIcon, Tick02Icon } from '@hugeicons/core-free-icons';
	import { asset } from '$app/paths';

	let { size = 16 }: { size?: number } = $props();
	const BIB = `@misc{mohsenvand2026check,
  title  = {When the Check Is the Algorithm},
  author = {Mohsenvand, Mostafa and Jahangiri, Pouyan},
  year   = {2026},
  url    = {https://github.com/NeoVand/cold-check}
}`;
	let copied = $state(false);
	function cite() {
		navigator.clipboard?.writeText(BIB);
		copied = true;
		setTimeout(() => (copied = false), 1600);
	}
</script>

<nav class="links" aria-label="Links">
	<a class="ilink" href={asset('paper/main.pdf' as Parameters<typeof asset>[0])}><HugeiconsIcon icon={File02Icon} {size} strokeWidth={1.6} />Paper</a>
	<a class="ilink" href="https://github.com/NeoVand/cold-check"><HugeiconsIcon icon={GithubIcon} {size} strokeWidth={1.6} />Code</a>
	<button class="ilink cite" onclick={cite} title="Copy the BibTeX entry"
		><HugeiconsIcon icon={copied ? Tick02Icon : QuoteDownIcon} {size} strokeWidth={1.6} />{copied ? 'Copied' : 'Cite'}</button
	>
</nav>

<style>
	.links {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 18px;
		font-size: 13px;
	}
	.cite {
		border: 0;
		padding: 0;
		background: none;
		font: inherit;
		cursor: pointer;
	}
</style>
