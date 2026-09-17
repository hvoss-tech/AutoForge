import { parseColoredMesh } from './plyParser'

/** Worker half of parseColoredMeshAsync (see plyWorkerClient.ts). */
self.onmessage = (event: MessageEvent) => {
  const { id, buffer } = event.data as { id: number; buffer: ArrayBuffer }
  try {
    const { positions, colors, indices, positionsHash } = parseColoredMesh(buffer)
    // Transfer the arrays rather than structured-cloning them: a preview
    // mesh is several MB, and copying it back would undo much of the point.
    const transfer: ArrayBuffer[] = [positions.buffer, indices.buffer]
    if (colors) transfer.push(colors.buffer)
    ;(self as unknown as Worker).postMessage({ id, positions, colors, indices, positionsHash }, transfer)
  } catch (err) {
    ;(self as unknown as Worker).postMessage({ id, error: err instanceof Error ? err.message : String(err) })
  }
}
