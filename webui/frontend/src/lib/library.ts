/** Pick the tab to show for a set of library types: keep the current one if
 * it still exists, otherwise the first available. The store starts on "PLA",
 * so a library without PLA filaments used to open on an empty tab. */
export function chooseActiveTab(types: string[], current: string): string {
  if (types.length === 0 || types.includes(current)) return current
  return types[0]
}
