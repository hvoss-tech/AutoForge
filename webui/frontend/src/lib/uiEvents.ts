/** One-off UI requests between panels that don't share state — e.g. clicking
 * a workflow step asks the library to take focus. */
export type UiCommand = 'focus-image' | 'focus-library' | 'focus-run' | 'focus-layers' | 'open-file-menu' | 'reveal-band'

const bus = new EventTarget()

export function sendUiCommand(command: UiCommand, detail?: unknown): void {
  bus.dispatchEvent(new CustomEvent(command, { detail }))
}

export function onUiCommand(command: UiCommand, handler: (detail: unknown) => void): () => void {
  const listener = (e: Event) => handler((e as CustomEvent).detail)
  bus.addEventListener(command, listener)
  return () => bus.removeEventListener(command, listener)
}
