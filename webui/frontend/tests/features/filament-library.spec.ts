import fs from 'node:fs'
import { test, expect } from '@playwright/test'
import { activeLibraryItem, byTestId, createFilament, openApp, resetBackend, setActive, showTab } from '../helpers'

test.beforeEach(async ({ baseURL }) => {
  await resetBackend(baseURL!)
})

const libraryItem = (page: import('@playwright/test').Page, uuid: string) => page.locator(`[data-testid="filament-${uuid}"]`)

test.describe('Browsing', () => {
  test('tabs list the library types and each tab shows only that type', async ({ page, request }) => {
    const petg = await createFilament(request, { filament_type: 'PETG', name: 'E2E Petg Only' })
    const pla = await createFilament(request, { filament_type: 'PLA', name: 'E2E Pla Only' })
    await openApp(page)

    const types: string[] = await (await request.get('/api/filaments/types')).json()
    for (const t of types) await expect(byTestId(page, `tab-${t}`)).toBeVisible()

    await showTab(page, 'PLA')
    await expect(libraryItem(page, pla.uuid)).toBeVisible()
    await expect(libraryItem(page, petg.uuid)).toHaveCount(0)

    await byTestId(page, 'tab-PETG').click()
    await expect(libraryItem(page, petg.uuid)).toBeVisible()
    await expect(libraryItem(page, pla.uuid)).toHaveCount(0)
  })

  test('brand folders collapse and expand', async ({ page, request }) => {
    const f = await createFilament(request, { brand: 'E2E Folder' })
    await openApp(page)
    await expect(libraryItem(page, f.uuid)).toBeVisible()
    await byTestId(page, 'brand-E2E Folder').click()
    await expect(libraryItem(page, f.uuid)).toHaveCount(0)
    await byTestId(page, 'brand-E2E Folder').click()
    await expect(libraryItem(page, f.uuid)).toBeVisible()
  })

  test('typing a search quickly ends on the results for the final query', async ({ page, request }) => {
    const target = await createFilament(request, { name: 'E2E Zebrawood' })
    const other = await createFilament(request, { name: 'E2E Zinc' })
    await openApp(page)

    // Slow down the *earlier* queries so their responses arrive last — the
    // list used to show whichever response landed last, not the newest query.
    await page.route('**/api/filaments?*', async (route) => {
      const query = new URL(route.request().url()).searchParams.get('query') ?? ''
      if (query.length > 0 && query.length < 'zebra'.length) await new Promise((r) => setTimeout(r, 700))
      await route.continue()
    })
    await byTestId(page, 'filter-input').pressSequentially('zebra', { delay: 40 })
    await page.waitForTimeout(1500)

    await expect(libraryItem(page, target.uuid)).toBeVisible()
    await expect(libraryItem(page, other.uuid)).toHaveCount(0)

    await page.unroute('**/api/filaments?*')
    await byTestId(page, 'filter-input').fill('')
    await expect(libraryItem(page, other.uuid)).toBeVisible()
  })

  test('a library without PLA opens on a tab that has filaments', async ({ page, request }) => {
    const petg = await createFilament(request, { filament_type: 'PETG' })
    await page.route('**/api/filaments/types', (route) => route.fulfill({ json: ['PETG'] }))
    await openApp(page)
    await expect(byTestId(page, 'tab-label')).toContainText('PETG')
    await expect(libraryItem(page, petg.uuid)).toBeVisible()
  })
})

test.describe('Active filaments', () => {
  test('the +/- toggle adds and removes, persisted on the server and across reloads', async ({ page, request }) => {
    const f = await createFilament(request)
    await openApp(page)

    await byTestId(page, `toggle-filament-${f.uuid}`).click()
    await expect(activeLibraryItem(page, f.uuid)).toBeVisible()
    await expect.poll(async () => (await (await request.get('/api/filaments/active')).json()).map((x: any) => x.uuid)).toContain(f.uuid)

    await page.reload()
    await expect(activeLibraryItem(page, f.uuid)).toBeVisible()
    await expect(byTestId(page, 'active-filaments-summary')).toHaveAttribute('data-count', '1')

    await byTestId(page, `toggle-filament-${f.uuid}`).click()
    await expect(activeLibraryItem(page, f.uuid)).toHaveCount(0)
    await expect.poll(async () => (await (await request.get('/api/filaments/active')).json()).length).toBe(0)
  })

  test('shows the empty-state hint with no active filaments', async ({ page }) => {
    await openApp(page)
    await expect(page.getByText('No active filaments')).toBeVisible()
  })
})

test.describe('Creating filaments', () => {
  test('a new filament of another type switches to that tab so it is visible', async ({ page }) => {
    await openApp(page)
    await showTab(page, 'PLA')

    await byTestId(page, 'new-filament-btn').click()
    await byTestId(page, 'new-filament-brand').fill('E2E Maker')
    await byTestId(page, 'new-filament-name').fill('E2E Silk Teal')
    await byTestId(page, 'new-filament-type').selectOption('PETG')
    await byTestId(page, 'create-filament-submit').click()

    await expect(byTestId(page, 'new-filament-modal')).toHaveCount(0)
    await expect(byTestId(page, 'tab-label')).toContainText('PETG')
    await expect(page.locator('[data-testid="filament-list"]').getByText('E2E Silk Teal')).toBeVisible()
    // And the PLA tab doesn't suddenly contain it.
    await showTab(page, 'PLA')
    await expect(page.locator('[data-testid="filament-list"]').getByText('E2E Silk Teal')).toHaveCount(0)
  })

  test('a translucent filament with TD above 10 can be saved', async ({ page, request }) => {
    await openApp(page)
    await byTestId(page, 'new-filament-btn').click()
    await byTestId(page, 'new-filament-brand').fill('E2E Clear')
    await byTestId(page, 'new-filament-name').fill('E2E Natural')
    await byTestId(page, 'new-filament-td').fill('25')
    await byTestId(page, 'create-filament-submit').click()

    await expect(byTestId(page, 'new-filament-modal')).toHaveCount(0)
    const all = await (await request.get('/api/filaments')).json()
    expect(all.find((f: any) => f.name === 'E2E Natural')?.td).toBe(25)
  })

  test('brand and name are required', async ({ page, request }) => {
    await openApp(page)
    await byTestId(page, 'new-filament-btn').click()
    await byTestId(page, 'new-filament-name').fill('E2E No Brand')
    await byTestId(page, 'create-filament-submit').click()
    await expect(byTestId(page, 'new-filament-modal')).toBeVisible()
    const all = await (await request.get('/api/filaments')).json()
    expect(all.some((f: any) => f.name === 'E2E No Brand')).toBe(false)
  })

  test('a brand-new category becomes a tab', async ({ page }) => {
    await openApp(page)
    await byTestId(page, 'new-filament-btn').click()
    await byTestId(page, 'new-filament-brand').fill('E2E Maker')
    await byTestId(page, 'new-filament-name').fill('E2E Wood')
    await byTestId(page, 'new-filament-type').selectOption('__new_type__')
    await byTestId(page, 'new-filament-custom-type').fill('E2EWOOD')
    await byTestId(page, 'create-filament-submit').click()
    await expect(byTestId(page, 'tab-E2EWOOD')).toBeVisible()
    await expect(byTestId(page, 'tab-label')).toContainText('E2EWOOD')
  })
})

test.describe('Editing filaments', () => {
  test('saving keeps the current tab and updates the active list', async ({ page, request }) => {
    const pla = await createFilament(request, { name: 'E2E Before' })
    const petg = await createFilament(request, { filament_type: 'PETG', name: 'E2E Other Type' })
    await setActive(request, [pla])
    await openApp(page)
    await showTab(page, 'PLA')

    await libraryItem(page, pla.uuid).dblclick()
    await byTestId(page, 'edit-filament-name').fill('E2E After')
    await byTestId(page, 'save-filament-btn').click()

    await expect(byTestId(page, 'edit-filament-modal')).toHaveCount(0)
    await expect(libraryItem(page, pla.uuid)).toContainText('E2E After')
    await byTestId(page, 'active-only-toggle').click()
    await expect(activeLibraryItem(page, pla.uuid)).toContainText('E2E After')
    await byTestId(page, 'active-only-toggle').click()
    // Editing used to reload every type into the current tab.
    await expect(libraryItem(page, petg.uuid)).toHaveCount(0)
    const active = await (await request.get('/api/filaments/active')).json()
    expect(active[0].name).toBe('E2E After')
  })

  test('changing TD updates the sliders that use the filament', async ({ page, request }) => {
    const f = await createFilament(request, { td: 2 })
    await setActive(request, [f])
    const state = await (await request.get('/api/project/state')).json()
    state.color_sliders[0] = { ...state.color_sliders[0], filament_uuid: f.uuid, td: 2, enabled: true }
    await request.post('/api/project/state', { data: state })
    await openApp(page)
    await expect(byTestId(page, 'td-input-0')).toHaveValue('2')

    await activeLibraryItem(page, f.uuid).dblclick()
    await byTestId(page, 'edit-filament-td').fill('4.5')
    await byTestId(page, 'save-filament-btn').click()

    await expect(byTestId(page, 'td-input-0')).toHaveValue('4.5')
  })

  test('delete needs a confirmation click and also deactivates', async ({ page, request }) => {
    const f = await createFilament(request)
    await setActive(request, [f])
    await openApp(page)

    await libraryItem(page, f.uuid).dblclick()
    await byTestId(page, 'delete-filament-btn').click()
    await expect(byTestId(page, 'delete-filament-btn')).toHaveText(/Confirm delete/)
    await expect(libraryItem(page, f.uuid)).toBeVisible()
    await byTestId(page, 'delete-filament-btn').click()

    await expect(libraryItem(page, f.uuid)).toHaveCount(0)
    await expect(byTestId(page, 'active-filaments-summary')).toHaveAttribute('data-count', '0')
    const all = await (await request.get('/api/filaments')).json()
    expect(all.some((x: any) => x.uuid === f.uuid)).toBe(false)
  })
})

test.describe('Import and export', () => {
  const csv = (rows: string[]) => ['Brand,Type,Color,Name,TD,Owned,UUID', ...rows].join('\n')

  test('merging a CSV adds its filaments and shows new types as tabs right away', async ({ page, request }, testInfo) => {
    const file = testInfo.outputPath('import.csv')
    fs.writeFileSync(file, csv(['E2E Import,E2EIMPORTTYPE,#123456,E2E Imported One,3.2,False,']))
    await openApp(page)

    await byTestId(page, 'import-btn').click()
    await byTestId(page, 'file-input').setInputFiles(file)
    await byTestId(page, 'import-mode-merge').click()
    await expect(byTestId(page, 'import-modal')).toContainText('Imported 1 filaments')

    // Previously only visible after a page reload.
    await expect(byTestId(page, 'tab-E2EIMPORTTYPE')).toBeVisible()
    const all = await (await request.get('/api/filaments')).json()
    expect(all.find((f: any) => f.name === 'E2E Imported One')).toMatchObject({ td: 3.2, filament_type: 'E2EIMPORTTYPE' })
  })

  test('the replace warning states the size of the whole library, not the current tab', async ({ page, request }, testInfo) => {
    await createFilament(request, { filament_type: 'PETG' })
    const total = (await (await request.get('/api/filaments')).json()).length
    const plaCount = (await (await request.get('/api/filaments?filament_type=PLA')).json()).length
    expect(total).not.toBe(plaCount)

    const file = testInfo.outputPath('replace.csv')
    fs.writeFileSync(file, csv(['E2E Import,PLA,#654321,E2E Replacement,1,False,']))
    await openApp(page)
    await showTab(page, 'PLA')
    await byTestId(page, 'import-btn').click()
    await byTestId(page, 'file-input').setInputFiles(file)

    await expect(byTestId(page, 'import-mode-replace')).toContainText(`Removes all ${total} current filaments`)
    await byTestId(page, 'import-mode-replace').click()
    await expect(byTestId(page, 'import-replace-confirm')).toContainText(`deletes all ${total} filaments`)
    // Back out — nothing replaced.
    await byTestId(page, 'import-replace-confirm').getByRole('button', { name: 'Cancel' }).click()
    expect((await (await request.get('/api/filaments')).json()).length).toBe(total)
  })

  test('replacing the library keeps only the file contents', async ({ page, request }, testInfo) => {
    const before = await (await request.get('/api/filaments')).json()
    const file = testInfo.outputPath('library.json')
    fs.writeFileSync(file, JSON.stringify([{ brand: 'E2E Json', name: 'E2E Only One', color: '#00ff00', td: 2, filament_type: 'PLA' }]))
    try {
      await openApp(page)
      await byTestId(page, 'import-btn').click()
      await byTestId(page, 'file-input').setInputFiles(file)
      await byTestId(page, 'import-mode-replace').click()
      await byTestId(page, 'import-replace-confirm-btn').click()
      await expect(byTestId(page, 'import-modal')).toContainText('library replaced')
      const after = await (await request.get('/api/filaments')).json()
      expect(after.map((f: any) => f.name)).toEqual(['E2E Only One'])
    } finally {
      // Restore the seeded library for the rest of the suite.
      await request.post('/api/filaments/import-json?mode=replace', { data: before })
    }
  })

  test('unsupported files are rejected with a message', async ({ page }, testInfo) => {
    const file = testInfo.outputPath('notes.txt')
    fs.writeFileSync(file, 'hello')
    await openApp(page)
    await byTestId(page, 'import-btn').click()
    await byTestId(page, 'file-input').setInputFiles(file)
    await expect(byTestId(page, 'import-modal')).toContainText('Only JSON and CSV files are supported')
    await expect(byTestId(page, 'import-mode-choice')).toHaveCount(0)
  })

  test('Save Library downloads every filament as JSON', async ({ page, request }) => {
    await createFilament(request, { filament_type: 'PETG' })
    const total = (await (await request.get('/api/filaments')).json()).length
    await openApp(page)
    const [download] = await Promise.all([page.waitForEvent('download'), byTestId(page, 'save-library-btn').click()])
    expect(download.suggestedFilename()).toBe('filament_library.json')
    const saved = JSON.parse(fs.readFileSync(await download.path(), 'utf8'))
    expect(saved).toHaveLength(total)
  })
})
