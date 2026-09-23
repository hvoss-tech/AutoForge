export interface ProgressState {
  progress: number    // 0.0 – 1.0
  elapsed: number     // ms
  eta: number         // ms (estimated remaining)
  rate: number        // progress/second (smoothed)
  stalled: boolean    // true if progress hasn't changed > 30s
}

const PROGRESS_THRESHOLD = 0.0015  // 0.15%
const TIME_THRESHOLD = 120         // ms
const STALL_THRESHOLD = 30000      // 30s
// A time constant, not a per-call decay: the caller here (TopBar) samples on
// a steady 1s interval, but nothing guarantees that — a caller driven
// straight off websocket messages can call update() every 100ms during a
// burst and every 2s during a lull. A fixed per-call decay (the old
// `smoothedRate * 0.7 + sampleRate * 0.3`) implicitly assumes evenly-spaced
// calls: a burst of frequent samples barely moves the average (fine), but a
// single call after a long gap is *also* only weighted 0.3 even though far
// more real time — and therefore real information — has passed since the
// last one, which is backwards. Converting to `alpha = 1 - exp(-dt / TAU)`
// below makes the blend weight follow elapsed wall-clock time instead of
// call count, so the same underlying rate signal comes out equally smooth
// whether update() is polled once a second or fed every websocket message.
const RATE_TIME_CONSTANT_MS = 4000

export interface ProgressTracker {
  update(progress: number, now: number): ProgressState
  reset(): void
}

export function createProgressTracker(timeConstantMs: number = RATE_TIME_CONSTANT_MS): ProgressTracker {
  let lastProgress = 0
  let lastTime = 0
  let smoothedRate = 0
  let lastChangeTime = 0
  let startTime = 0

  return {
    update(progress: number, now: number): ProgressState {
      if (startTime === 0) startTime = now
      if (lastTime === 0) {
        lastProgress = progress
        lastTime = now
        lastChangeTime = now
        return { progress, elapsed: 0, eta: 0, rate: 0, stalled: false }
      }

      const elapsed = now - startTime
      const dt = now - lastTime
      const dp = progress - lastProgress

      // Only update rate when progress actually changes enough
      if (Math.abs(dp) > PROGRESS_THRESHOLD || dt > TIME_THRESHOLD) {
        if (dp > 0) lastChangeTime = now
        const sampleRate = dt > 0 ? dp / (dt / 1000) : 0
        const alpha = dt > 0 ? 1 - Math.exp(-dt / timeConstantMs) : 1
        smoothedRate = smoothedRate === 0
          ? sampleRate
          : smoothedRate * (1 - alpha) + sampleRate * alpha

        lastProgress = progress
        lastTime = now
      }

      const stalled = (now - lastChangeTime) > STALL_THRESHOLD
      const remaining = smoothedRate > 0
        ? ((1 - progress) / smoothedRate) * 1000
        : 0

      return {
        progress,
        elapsed,
        eta: remaining,
        rate: smoothedRate,
        stalled,
      }
    },

    reset() {
      lastProgress = 0
      lastTime = 0
      smoothedRate = 0
      lastChangeTime = 0
      startTime = 0
    },
  }
}

/**
 * Format ms to "Xm Ys" or "Xs" for display.
 */
export function formatDuration(ms: number): string {
  if (ms !== 0 && !ms) return '...'
  if (!Number.isFinite(ms) || ms < 0) return '...'
  const totalSeconds = Math.round(ms / 1000)
  if (totalSeconds < 0) return '...'
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  if (minutes > 0) return `${minutes}m ${seconds}s`
  return `${seconds}s`
}
