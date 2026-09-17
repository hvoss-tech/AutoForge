/** Saving/naming projects and telling whether there are unsaved changes. */

export function projectFileName(name: string): string {
  const slug = name
    .trim()
    .replace(/[^\p{L}\p{N}]+/gu, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80)
  return `${slug || 'autoforge-project'}.json`
}

/** Project name from a loaded file: its stored name, else the file name. */
export function projectNameFromFile(storedName: unknown, fileName: string): string {
  if (typeof storedName === 'string' && storedName.trim()) return storedName.trim()
  return fileName.replace(/\.json$/i, '').replace(/^autoforge-project(-\d+)?$/, '')
}

interface ProjectContent {
  name: string
  inputImage: string | null
  settings: unknown
  colorSliders: unknown
  activeFilamentUuids: string[]
}

/** FNV-1a over the saved content — a cheap fingerprint to compare against the
 * one taken at save time. Undoing back to the saved state counts as saved. */
export function projectFingerprint(content: ProjectContent): string {
  const text = JSON.stringify([content.name, content.inputImage, content.settings, content.colorSliders, [...content.activeFilamentUuids].sort()])
  let hash = 0x811c9dc5
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193)
  }
  return (hash >>> 0).toString(16)
}

export function hasUnsavedChanges(savedFingerprint: string | null, current: string, hasContent: boolean): boolean {
  return savedFingerprint === null ? hasContent : savedFingerprint !== current
}
