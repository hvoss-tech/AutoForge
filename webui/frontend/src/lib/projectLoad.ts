import type { Filament } from '../types'

export interface MissingFilamentResult {
  /** Whether anything was added to the library. */
  libraryChanged: boolean
  /** Filaments the server refused, with its reason. */
  failed: { filament: Filament; detail: string }[]
}

/** Add the project's filaments the library doesn't have yet, one request
 * each, and report which the server refused. The responses used to go
 * unchecked: a refused filament (e.g. a non-hex colour) was skipped
 * silently, and the active-list sync that followed then failed for every
 * filament with no hint which one was bad. */
export async function addMissingFilaments(
  filaments: Filament[],
  libraryUuids: Set<string>,
  post: (filament: Filament) => Promise<{ ok: boolean; status: number; detail?: string }>,
): Promise<MissingFilamentResult> {
  const result: MissingFilamentResult = { libraryChanged: false, failed: [] }
  for (const f of filaments) {
    if (libraryUuids.has(f.uuid)) continue
    try {
      const response = await post(f)
      if (response.ok) result.libraryChanged = true
      else result.failed.push({ filament: f, detail: response.detail || `HTTP ${response.status}` })
    } catch (e) {
      result.failed.push({ filament: f, detail: e instanceof Error ? e.message : String(e) })
    }
  }
  return result
}

export function filamentLabel(f: Pick<Filament, 'brand' | 'name'>): string {
  return [f.brand, f.name].filter(Boolean).join(' - ') || 'Unnamed filament'
}
