export interface PruningCounts {
  colors: number
  swaps: number
  layers: number
}

/** Starting values for the pruning dialog: exactly what the result has now.
 *
 * The dialog used to pre-fill a reduction of its own (one color fewer, a
 * third fewer swaps). That made the numbers a suggestion the user hadn't
 * asked for — pressing Start accepted a target nobody chose, and there was
 * no way to tell what the result actually had without reading the "now"
 * labels. Starting from the current counts means the limits are a floor the
 * user raises or lowers deliberately; pruning still improves the result at
 * equal limits, because it also merges same-color bands and moves swaps.
 *
 * (The old fixed 100/100/75 defaults are a separate, still-fixed problem:
 * they were above any real result, so pruning with them changed nothing.) */
export function suggestPruningLimits(current: PruningCounts): PruningCounts {
  return {
    colors: Math.max(1, current.colors || 1),
    swaps: Math.max(0, current.swaps || 0),
    layers: Math.max(1, current.layers || 1),
  }
}

/** The counts the pruner limits. Max colors includes the base color (the
 * pruner subtracts it), so it's one more than the layer bands use. */
export function resultCounts(plan: { colors: number; swaps: number; topLayer: number }): PruningCounts {
  return { colors: plan.colors + 1, swaps: plan.swaps, layers: plan.topLayer }
}

/** The counts to show while pruning runs.
 *
 * The dialog's own numbers come from the color sliders, and those only
 * change when the backend pushes the finished stack — so they sat frozen at
 * the pre-pruning values for the whole run. The pruning job reports the live
 * counts of the solution it is working on; use those whenever they're
 * present, and fall back to the sliders before the first report arrives. */
export function liveCounts(
  fallback: PruningCounts,
  job: { result_colors?: number | null; result_swaps?: number | null; result_layers?: number | null } | null,
): PruningCounts {
  if (!job) return fallback
  const pick = (reported: number | null | undefined, current: number) =>
    typeof reported === 'number' && Number.isFinite(reported) ? reported : current
  return {
    colors: pick(job.result_colors, fallback.colors),
    swaps: pick(job.result_swaps, fallback.swaps),
    layers: pick(job.result_layers, fallback.layers),
  }
}

/** "26 → 12 colors, 25 → 11 swaps, 60 layers" — unchanged counts stay single. */
export function describePruningChange(before: PruningCounts, after: PruningCounts): string {
  const part = (key: keyof PruningCounts, noun: string) =>
    before[key] === after[key] ? `${after[key]} ${noun}` : `${before[key]} → ${after[key]} ${noun}`
  return [part('colors', 'colors'), part('swaps', 'swaps'), part('layers', 'layers')].join(', ')
}
