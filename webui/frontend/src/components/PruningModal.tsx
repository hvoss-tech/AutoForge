import React, { useEffect, useMemo, useState } from 'react'
import { Info, Loader2, Pause, Play, Scissors, Square, X } from 'lucide-react'
import { useAppStore } from '../store/appStore'
import { NumberInput } from './ui/number-input'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from './ui/dialog'
import { buildPrintPlan } from '../lib/printPlan'
import { describePruningChange, liveCounts, resultCounts, suggestPruningLimits } from '../lib/pruning'

const Current: React.FC<{ label: string; value: number; testId: string; from?: number }> = ({ label, value, testId, from }) => (
  <div className="flex-1 px-3 py-2 rounded bg-gray-800 text-center" data-testid={testId} data-value={value}>
    <div className="text-lg font-semibold text-gray-100 tabular-nums">
      {from !== undefined && from !== value && (
        <span className="text-xs font-normal text-gray-500 line-through mr-1 tabular-nums" data-testid={`${testId}-before`}>{from}</span>
      )}
      {value}
    </div>
    <div className="text-[11px] text-gray-400">{label}</div>
  </div>
)

/** Safety net for auto-repeat: matches PruningSettings.max_passes. */
const MAX_AUTO_PASSES = 25

/** The two polish passes that run before the reduction phases. Both can only
 * improve the result — each keeps its work solely if the real loss went down
 * — so the only reason to turn one off is the time it takes. */
const POLISH_FIELDS = [
  {
    key: 'seedSearch',
    limitKey: 'seedSearchCount',
    testId: 'pruning-seed-search',
    limitTestId: 'pruning-seed-search-count',
    label: 'Look for better colors first',
    help:
      'The per-layer colors are read out of the solution with a seeded random draw, so a different seed can give a ' +
      'noticeably better set of them for free. Only a seed that beats the current result is kept.',
    limitLabel: 'Seeds to try',
    limitHint: 'more seeds, better odds, longer wait',
    min: 1,
    max: 20000,
  },
  {
    key: 'fineTuneHeight',
    limitKey: 'fineTuneSteps',
    testId: 'pruning-fine-tune-height',
    limitTestId: 'pruning-fine-tune-steps',
    label: 'Polish the heights first',
    help:
      'Nudges each layer band\'s height to fit the picture better, then cleans up any spikes that introduces. Kept ' +
      'only if the combined result is better.',
    limitLabel: 'Steps',
    limitHint: 'stops early once it stops improving',
    min: 1,
    max: 2000,
  },
] as const

const LIMIT_FIELDS = [
  { key: 'colors', label: 'Max colors', help: 'Including the base color. Fewer colors means fewer filaments to have on hand.', min: 1, testId: 'pruning-max-colors' },
  { key: 'swaps', label: 'Max swaps', help: 'Each swap is a pause to change filament during the print.', min: 0, testId: 'pruning-max-swaps' },
  { key: 'layers', label: 'Max layers', help: 'Fewer layers print faster and make a thinner print.', min: 1, testId: 'pruning-max-layer' },
] as const

export const PruningModal: React.FC = () => {
  const open = useAppStore((s) => s.pruningModalOpen)
  const setOpen = useAppStore((s) => s.setPruningModalOpen)
  const setPruningSettings = useAppStore((s) => s.setPruningSettings)
  const startPruning = useAppStore((s) => s.startPruning)
  const pruningJob = useAppStore((s) => s.pruningJob)
  const pausePruning = useAppStore((s) => s.pausePruning)
  const resumePruning = useAppStore((s) => s.resumePruning)
  const cancelPruning = useAppStore((s) => s.cancelPruning)
  const colorSliders = useAppStore((s) => s.colorSliders)
  const settings = useAppStore((s) => s.settings)

  const pruningBaseline = useAppStore((s) => s.pruningBaseline)
  const plan = useMemo(() => buildPrintPlan(colorSliders, [], settings), [colorSliders, settings])
  // Max colors counts the base/background color too (the pruner subtracts it).
  const fromSliders = resultCounts(plan)
  const running = !!pruningJob && ['pending', 'running', 'paused'].includes(pruningJob.status)
  // While pruning runs the sliders still hold the *pre-pruning* stack (the
  // backend only pushes the new one at the end), so the job's own live
  // counts are what the "Current result" tiles follow. Once it has finished,
  // the sliders are authoritative again — and they keep following the user's
  // own edits, which the finished job's counts would not.
  const current = running ? liveCounts(fromSliders, pruningJob) : fromSliders

  const [limits, setLimits] = useState(() => suggestPruningLimits(current))
  const [error, setError] = useState<string | null>(null)
  const [startedHere, setStartedHere] = useState(false)
  // Pruning is a greedy search that restarts from whatever it is given, so
  // repeating it keeps gaining until it doesn't. Off by default: each pass is
  // real GPU time, and the user should choose to spend it.
  const [autoRepeat, setAutoRepeat] = useState(false)
  // The two polish passes that run before the reduction phases. Both are
  // non-worsening, but height fine-tuning is the slower of the two, so only
  // color-seed search defaults on.
  const [polish, setPolish] = useState({
    seedSearch: true,
    seedSearchCount: 200,
    fineTuneHeight: false,
    fineTuneSteps: 50,
  })

  // Fresh suggestions each time the dialog opens (the result may have changed).
  useEffect(() => {
    if (!open) return
    setLimits(suggestPruningLimits(current))
    setError(null)
    if (!running) setStartedHere(false)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  const showProgress = running || (startedHere && !!pruningJob)

  const handleSubmit = async () => {
    setPruningSettings({
      pruning_max_colors: limits.colors,
      pruning_max_swaps: limits.swaps,
      pruning_max_layer: limits.layers,
      auto_repeat: autoRepeat,
      max_passes: MAX_AUTO_PASSES,
      seed_search: polish.seedSearch,
      seed_search_count: polish.seedSearchCount,
      fine_tune_height: polish.fineTuneHeight,
      fine_tune_steps: polish.fineTuneSteps,
    })
    useAppStore.setState({ pruningBaseline: current })
    setError(null)
    try {
      await startPruning()
      setStartedHere(true)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to start pruning')
    }
  }

  const safely = (fn: () => Promise<void>) => async () => {
    try {
      await fn()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const lossReadout = useMemo(() => {
    const value = pruningJob?.loss
    if (typeof value !== 'number' || !Number.isFinite(value)) return null
    const from = typeof pruningJob?.pruning_start_loss === 'number' ? pruningJob.pruning_start_loss : null
    return {
      value,
      from,
      improved: from !== null && value < from,
      // A tiny wobble is re-scoring noise, not a regression worth flagging.
      worse: from !== null && value > from + 0.01,
    }
  }, [pruningJob?.loss, pruningJob?.pruning_start_loss])

  // "pass 3 of 25" while auto-repeat is working through them.
  const passLabel =
    pruningJob?.pruning_pass && pruningJob.pruning_pass > 0
      ? `pass ${pruningJob.pruning_pass}${pruningJob.pruning_max_passes ? ` of ${pruningJob.pruning_max_passes}` : ''}`
      : null

  const statusText = !pruningJob
    ? ''
    : pruningJob.status === 'completed'
      ? 'Pruning finished — the color layers were updated.'
      : pruningJob.status === 'cancelled'
        ? 'Pruning cancelled. The result up to that point was kept.'
        : pruningJob.status === 'failed'
          ? `Pruning failed: ${pruningJob.error ?? 'unknown error'}`
          : pruningJob.status === 'paused'
            ? 'Paused'
            : pruningJob.phase ?? 'Starting…'

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="max-w-md max-h-[90vh] flex flex-col p-0 gap-0 bg-gray-900 border-gray-700" data-testid="pruning-modal">
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
          <DialogTitle className="flex items-center gap-2 text-sm font-semibold text-gray-100">
            <Scissors className="w-4 h-4" /> Pruning
          </DialogTitle>
          <button onClick={() => setOpen(false)} className="p-1 rounded text-gray-400 hover:text-gray-100 hover:bg-gray-800" aria-label="Close pruning" data-testid="pruning-close-x">
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-4">
          <DialogDescription className="text-xs text-gray-300">
            Makes the result easier to print — fewer filaments, swaps and layers — while keeping the picture as close as possible.
          </DialogDescription>

          <div>
            <div className="text-[11px] uppercase tracking-wide text-gray-500 mb-1">
              {running ? 'Result so far' : 'Current result'}
            </div>
            <div className="flex gap-2" data-testid="pruning-current-counts" data-live={running || undefined}>
              <Current label="colors incl. base" value={current.colors} from={running ? pruningBaseline?.colors : undefined} testId="pruning-current-colors" />
              <Current label="swaps" value={current.swaps} from={running ? pruningBaseline?.swaps : undefined} testId="pruning-current-swaps" />
              <Current label="layers" value={current.layers} from={running ? pruningBaseline?.layers : undefined} testId="pruning-current-layers" />
            </div>
            {/* The searches that run before any reduction don't change the
                counts at all — the loss is the only place their improvement
                shows, so it needs to be on screen while they work. */}
            {lossReadout && (
              <div className="flex items-baseline gap-2 mt-2 px-3 py-1.5 rounded bg-gray-800 text-xs" data-testid="pruning-loss">
                <span className="text-gray-400">Difference from your image</span>
                {lossReadout.from !== null && lossReadout.from !== lossReadout.value && (
                  <span className="text-gray-500 line-through tabular-nums" data-testid="pruning-loss-before">
                    {lossReadout.from.toFixed(2)}
                  </span>
                )}
                {/* Amber when it went up: pruning is meant to be
                    non-worsening, so a regression is something to see rather
                    than something to blend in. */}
                <span
                  className={`font-semibold tabular-nums ${
                    lossReadout.improved ? 'text-emerald-500' : lossReadout.worse ? 'text-amber-500' : 'text-gray-100'
                  }`}
                  data-testid="pruning-loss-value"
                  data-trend={lossReadout.improved ? 'better' : lossReadout.worse ? 'worse' : 'same'}
                >
                  {lossReadout.value.toFixed(2)}
                </span>
                <span className="ml-auto text-[11px] text-gray-500">lower is closer to the original</span>
              </div>
            )}
          </div>

          {showProgress && pruningJob ? (
            <div className="space-y-2" data-testid="pruning-progress-view">
              <div className="flex items-center gap-2 text-xs text-gray-200">
                {running && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
                <span data-testid="pruning-modal-status">{statusText}</span>
                {passLabel && (
                  <span className="ml-auto text-[11px] text-gray-400 tabular-nums" data-testid="pruning-pass-label">
                    {passLabel}
                  </span>
                )}
              </div>
              <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                <div className="h-full bg-purple-500 transition-all" style={{ width: `${pruningJob.status === 'completed' ? 100 : pruningJob.progress}%` }} />
              </div>
              {pruningJob.status === 'completed' && pruningBaseline && (
                <>
                  <p className="text-xs text-gray-200" data-testid="pruning-summary">
                    {describePruningChange(pruningBaseline, current)}
                  </p>
                  <p className="text-[11px] text-gray-400" data-testid="pruning-again-hint">
                    Running it again on the same limits often improves the result further — each pass starts from where
                    the last one stopped. "Prune further" reopens the limits, or tick "keep pruning until it stops
                    improving" to have it repeat on its own.
                  </p>
                </>
              )}
              {running && (
                <div className="flex gap-2">
                  {pruningJob.status === 'paused' ? (
                    <button onClick={safely(() => resumePruning(pruningJob.job_id))} className="flex items-center gap-1 px-2.5 py-1 text-xs rounded bg-gray-700 hover:bg-gray-600 text-gray-100" data-testid="pruning-modal-resume">
                      <Play className="w-3.5 h-3.5" /> Resume
                    </button>
                  ) : (
                    <button onClick={safely(() => pausePruning(pruningJob.job_id))} className="flex items-center gap-1 px-2.5 py-1 text-xs rounded bg-gray-700 hover:bg-gray-600 text-gray-100" data-testid="pruning-modal-pause">
                      <Pause className="w-3.5 h-3.5" /> Pause
                    </button>
                  )}
                  <button onClick={safely(() => cancelPruning(pruningJob.job_id))} className="flex items-center gap-1 px-2.5 py-1 text-xs rounded bg-red-600 hover:bg-red-500 text-white" data-testid="pruning-modal-cancel">
                    <Square className="w-3.5 h-3.5" /> Cancel
                  </button>
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-3">
              <p className="flex items-start gap-2 p-2.5 rounded bg-blue-500/10 border border-blue-500/30 text-[11px] text-gray-200" data-testid="pruning-repeat-hint">
                <Info className="w-3.5 h-3.5 flex-shrink-0 mt-px text-blue-400" />
                <span>
                  <strong className="font-medium">Run pruning several times.</strong> Each pass is a greedy search that
                  starts from the result it is given, so a second and third run on the same limits usually keep
                  improving the picture — and lowering the limits a little at a time gives a better print than one big
                  jump. The values below start at what the result has now; lower one to push further, or tick the box
                  at the bottom to let it repeat by itself.
                </span>
              </p>
              {LIMIT_FIELDS.map(({ key, label, help, min, testId }) => (
                <div key={key}>
                  <label className="flex items-center justify-between text-xs text-gray-300 mb-1" htmlFor={`pruning-${key}`}>
                    <span>{label}</span>
                    <span className="text-gray-500">now {current[key]}</span>
                  </label>
                  <NumberInput
                    id={`pruning-${key}`}
                    value={limits[key]}
                    onValueChange={(v) => setLimits((prev) => ({ ...prev, [key]: v }))}
                    integer
                    min={min}
                    max={key === 'layers' ? 200 : undefined}
                    className="w-full text-sm bg-gray-800 border border-gray-600 rounded px-2 py-1.5 text-gray-100 focus:outline-none focus:border-blue-500"
                    data-testid={testId}
                  />
                  <p className="text-[11px] text-gray-400 mt-0.5">{help}</p>
                </div>
              ))}

              {/* Before the reduction phases: two searches that only ever
                  improve the result, so the greedy phases after them start
                  from a better solution. */}
              <div className="space-y-2 pt-1 border-t border-gray-800" data-testid="pruning-polish">
                <div className="text-[11px] uppercase tracking-wide text-gray-500">Before reducing anything</div>
                {POLISH_FIELDS.map((field) => (
                  <div key={field.key} className="p-2.5 rounded border border-gray-700 bg-gray-800/60">
                    <label className="flex items-start gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={polish[field.key] as boolean}
                        onChange={(e) => setPolish((p) => ({ ...p, [field.key]: e.target.checked }))}
                        className="mt-0.5"
                        data-testid={field.testId}
                      />
                      <span className="text-xs text-gray-200">
                        {field.label}
                        <span className="block text-[11px] text-gray-400 mt-0.5">{field.help}</span>
                      </span>
                    </label>
                    {(polish[field.key] as boolean) && (
                      <label className="flex items-center gap-2 mt-2 ml-6 text-[11px] text-gray-400">
                        {field.limitLabel}
                        <NumberInput
                          value={polish[field.limitKey] as number}
                          onValueChange={(v) => setPolish((p) => ({ ...p, [field.limitKey]: v }))}
                          integer
                          min={field.min}
                          max={field.max}
                          className="w-20 text-xs bg-gray-900 border border-gray-600 rounded px-1.5 py-0.5 text-center text-gray-100"
                          data-testid={field.limitTestId}
                        />
                        <span className="text-gray-500">{field.limitHint}</span>
                      </label>
                    )}
                  </div>
                ))}
              </div>

              <label className="flex items-start gap-2 p-2.5 rounded border border-gray-700 bg-gray-800/60 cursor-pointer" data-testid="pruning-auto-repeat-label">
                <input
                  type="checkbox"
                  checked={autoRepeat}
                  onChange={(e) => setAutoRepeat(e.target.checked)}
                  className="mt-0.5"
                  data-testid="pruning-auto-repeat"
                />
                <span className="text-xs text-gray-200">
                  Keep pruning until it stops improving
                  <span className="block text-[11px] text-gray-400 mt-0.5">
                    Runs pass after pass with these same limits and stops on its own once a pass no longer improves the
                    picture (at most {MAX_AUTO_PASSES}). You can stop it at any point — the best result so far is kept.
                  </span>
                </span>
              </label>
            </div>
          )}

          {error && (
            <div className="p-2.5 rounded bg-red-600/15 border border-red-600/40 text-xs text-red-500" role="alert">
              {error}
            </div>
          )}
        </div>

        <div className="flex-shrink-0 flex justify-end gap-2 px-4 py-3 border-t border-gray-700">
          <button onClick={() => setOpen(false)} className="px-3 py-1.5 text-xs text-gray-200 bg-gray-700 hover:bg-gray-600 rounded" data-testid="pruning-close-btn">
            {running ? 'Close (keeps running)' : 'Close'}
          </button>
          {!showProgress && (
            <button onClick={handleSubmit} className="flex items-center gap-1 px-3 py-1.5 text-xs text-white bg-purple-600 hover:bg-purple-500 rounded" data-testid="pruning-start-btn">
              <Play className="w-3 h-3" /> Start pruning
            </button>
          )}
          {showProgress && !running && (
            <button onClick={() => { setStartedHere(false); setLimits(suggestPruningLimits(current)) }} className="px-3 py-1.5 text-xs text-white bg-purple-600 hover:bg-purple-500 rounded" data-testid="pruning-again-btn">
              Prune further
            </button>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
