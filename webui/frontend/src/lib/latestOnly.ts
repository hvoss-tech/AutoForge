/** Runs async jobs one at a time, newest first: a job asked for while one
 * is running waits, and replaces any other job waiting (which resolves to
 * null without running). For renders that take longer than a slider moves:
 * every slider position used to be rendered in turn, long after the slider
 * had stopped. */
export class LatestOnly<T> {
  private running = false
  private waiting: { job: () => Promise<T>; resolve: (value: T | null) => void; reject: (err: unknown) => void } | null = null

  run(job: () => Promise<T>): Promise<T | null> {
    if (this.running) {
      this.waiting?.resolve(null)
      return new Promise((resolve, reject) => {
        this.waiting = { job, resolve, reject }
      })
    }
    this.running = true
    return job().finally(() => {
      this.running = false
      const next = this.waiting
      this.waiting = null
      if (next) this.run(next.job).then(next.resolve, next.reject)
    })
  }
}
