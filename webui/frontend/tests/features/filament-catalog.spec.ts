import { test, expect, type APIRequestContext, type Page } from '@playwright/test'
import { activeLibraryItem, byTestId, createFilament, openApp, resetBackend } from '../helpers'

// The filamentcolors.xyz catalog dialog. The test server never contacts the
// site (AUTOFORGE_WEBUI_FILAMENTCOLORS_AUTO_UPDATE=false in test-server.mjs),
// so everything here runs against the bundled snapshot.

interface Entry {
  id: number
  brand: string
  name: string
  color_name: string
  filament_type: string
  color: string
  td: number
  color_family: string
}

test.beforeEach(async ({ baseURL }) => {
  await resetBackend(baseURL!)
})

test.afterEach(async ({ baseURL }, testInfo) => {
  for (const a of testInfo.annotations) {
    if (a.type === 'cleanup' && a.description) await fetch(`${baseURL}/api/filaments/${a.description}`, { method: 'DELETE' }).catch(() => {})
  }
  // resetBackend's cleanup also removes filamentcolors-* filaments; run it
  // afterwards too so none linger for the next spec file.
  await resetBackend(baseURL!)
})

async function catalogEntries(request: APIRequestContext): Promise<Entry[]> {
  const res = await request.get('/api/filaments/catalog')
  expect(res.ok()).toBeTruthy()
  return (await res.json()).filaments
}

/** An entry whose brand + color name identify it uniquely in the catalog. */
async function uniqueEntry(request: APIRequestContext, predicate: (e: Entry) => boolean = () => true): Promise<Entry> {
  const entries = await catalogEntries(request)
  const key = (e: Entry) => `${e.brand} ${e.color_name}`.toLowerCase()
  const counts = new Map<string, number>()
  for (const e of entries) counts.set(key(e), (counts.get(key(e)) ?? 0) + 1)
  const found = entries.find((e) => counts.get(key(e)) === 1 && predicate(e) && !/[()]/.test(e.color_name))
  expect(found).toBeTruthy()
  return found!
}

async function openCatalog(page: Page) {
  await openApp(page)
  await byTestId(page, 'catalog-btn').click()
  await expect(byTestId(page, 'catalog-modal')).toBeVisible()
  await expect(page.locator('[data-testid^="catalog-card-"]').first()).toBeVisible()
}

const resultCount = async (page: Page) => Number(await byTestId(page, 'catalog-result-count').getAttribute('data-count'))

test.describe('Filament catalog', () => {
  test('opens from the library with the whole catalog and the search focused', async ({ page, request }) => {
    const total = (await catalogEntries(request)).length
    await openCatalog(page)
    await expect(byTestId(page, 'catalog-summary')).toContainText(`${total} filaments with a measured TD`)
    await expect(byTestId(page, 'catalog-summary')).toContainText('filamentcolors.xyz')
    await expect(byTestId(page, 'catalog-search')).toBeFocused()
    expect(await resultCount(page)).toBe(total)
    // Rendered in batches, not all at once.
    await expect(page.locator('[data-testid^="catalog-card-"]')).toHaveCount(60)

    await page.keyboard.press('Escape')
    await expect(byTestId(page, 'catalog-modal')).toHaveCount(0)
  })

  test('search narrows the results by brand and color name, in any word order', async ({ page, request }) => {
    const target = await uniqueEntry(request)
    await openCatalog(page)
    await byTestId(page, 'catalog-search').fill(`${target.color_name} ${target.brand}`)
    await expect(byTestId(page, `catalog-card-${target.id}`)).toBeVisible()
    const n = await resultCount(page)
    expect(n).toBeGreaterThanOrEqual(1)
    expect(n).toBeLessThan(20)

    await byTestId(page, 'catalog-search').fill(target.color.replace('#', ''))
    await expect(byTestId(page, `catalog-card-${target.id}`)).toBeVisible()

    await byTestId(page, 'catalog-search-clear').click()
    expect(await resultCount(page)).toBe((await catalogEntries(request)).length)
  })

  test('type, brand and color-family filters combine, and reset clears them', async ({ page, request }) => {
    const entries = await catalogEntries(request)
    const petgBlue = entries.filter((e) => e.filament_type === 'PETG' && e.color_family === 'Blue')
    expect(petgBlue.length).toBeGreaterThan(0)
    await openCatalog(page)

    await byTestId(page, 'catalog-type-filter').selectOption('PETG')
    await byTestId(page, 'catalog-family-Blue').click()
    await expect(byTestId(page, 'catalog-family-Blue')).toHaveAttribute('aria-pressed', 'true')
    await expect.poll(() => resultCount(page)).toBe(petgBlue.length)

    const brand = petgBlue[0].brand
    await byTestId(page, 'catalog-brand-filter').selectOption(brand)
    await expect.poll(() => resultCount(page)).toBe(petgBlue.filter((e) => e.brand === brand).length)
    for (const card of await page.locator('[data-testid^="catalog-card-"]').all()) {
      await expect(card).toContainText(brand)
      await expect(card).toContainText('PETG')
    }

    await byTestId(page, 'catalog-reset-filters').click()
    await expect.poll(() => resultCount(page)).toBe(entries.length)
    await expect(byTestId(page, 'catalog-family-all')).toHaveAttribute('aria-pressed', 'true')
  })

  test('adding a filament puts it in the library, marked owned, and the card says so', async ({ page, request }) => {
    const target = await uniqueEntry(request)
    const uuid = `filamentcolors-${target.id}`
    await openCatalog(page)
    await byTestId(page, 'catalog-search').fill(`${target.brand} ${target.color_name}`)
    await byTestId(page, `catalog-add-${target.id}`).click()

    await expect(byTestId(page, `catalog-in-library-${target.id}`)).toBeVisible()
    await expect(byTestId(page, `catalog-card-${target.id}`)).toHaveAttribute('data-in-library', 'true')
    await expect(byTestId(page, 'toast-info')).toContainText(`Added ${target.brand} ${target.name}`)

    const saved = (await (await request.get('/api/filaments')).json()).find((f: any) => f.uuid === uuid)
    expect(saved).toMatchObject({ brand: target.brand, name: target.name, color: target.color, td: target.td, owned: true, filament_type: target.filament_type, source: 'filamentcolors' })
    // Not activated unless asked.
    expect((await (await request.get('/api/filaments/active')).json()).map((f: any) => f.uuid)).not.toContain(uuid)

    await page.keyboard.press('Escape')
    await expect(byTestId(page, `filament-${uuid}`)).toBeVisible()
    await expect(byTestId(page, `owned-badge-${uuid}`)).toBeVisible()

    // Survives a reload, and the catalog still knows it's in the library.
    await page.reload()
    await expect(byTestId(page, `filament-${uuid}`)).toBeVisible({ timeout: 15000 })
    await byTestId(page, 'catalog-btn').click()
    await byTestId(page, 'catalog-search').fill(`${target.brand} ${target.color_name}`)
    await expect(byTestId(page, `catalog-in-library-${target.id}`)).toBeVisible()
    const inLibrary = Object.keys((await (await request.get('/api/filaments/catalog')).json()).library_uuids).length
    await expect(byTestId(page, 'catalog-summary')).toContainText(`${inLibrary} already in your library`)
  })

  test('"also make active" activates the added filament; "owned" can be switched off', async ({ page, request }) => {
    const target = await uniqueEntry(request)
    const uuid = `filamentcolors-${target.id}`
    await openCatalog(page)
    await byTestId(page, 'catalog-owned-toggle').uncheck()
    await byTestId(page, 'catalog-activate-toggle').check()
    await byTestId(page, 'catalog-search').fill(`${target.brand} ${target.color_name}`)
    await byTestId(page, `catalog-add-${target.id}`).click()
    await expect(byTestId(page, `catalog-in-library-${target.id}`)).toBeVisible()

    await expect.poll(async () => (await (await request.get('/api/filaments/active')).json()).map((f: any) => f.uuid)).toContain(uuid)
    const saved = (await (await request.get('/api/filaments')).json()).find((f: any) => f.uuid === uuid)
    expect(saved.owned).toBe(false)
    await page.keyboard.press('Escape')
    await expect(activeLibraryItem(page, uuid)).toBeVisible()

    // The choices are remembered for next time.
    await page.reload()
    await byTestId(page, 'catalog-btn').click()
    await expect(byTestId(page, 'catalog-owned-toggle')).not.toBeChecked()
    await expect(byTestId(page, 'catalog-activate-toggle')).toBeChecked()
  })

  test('"hide ones in my library" hides filaments already there, including your own', async ({ page, request }) => {
    const target = await uniqueEntry(request)
    // One of the user's own that is the same filament as a catalog entry.
    const own = await uniqueEntry(request, (e) => e.id !== target.id)
    // A real brand name, which the shared cleanup doesn't recognise — removed below.
    const mine = await createFilament(request, { brand: own.brand, name: own.name, filament_type: own.filament_type, color: '#123456', td: 1 })
    test.info().annotations.push({ type: 'cleanup', description: mine.uuid })
    await request.post('/api/filaments/catalog/add', { data: { ids: [target.id] } })
    const catalog = await (await request.get('/api/filaments/catalog')).json()
    const total = catalog.filaments.length
    const inLibrary = Object.keys(catalog.library_uuids).length
    expect(catalog.library_uuids[String(own.id)]).toBe(mine.uuid)

    await openCatalog(page)
    await byTestId(page, 'catalog-search').fill(`${own.brand} ${own.color_name}`)
    await expect(byTestId(page, `catalog-in-library-${own.id}`)).toBeVisible()
    await byTestId(page, 'catalog-search-clear').click()

    await byTestId(page, 'catalog-hide-in-library').check()
    await expect.poll(() => resultCount(page)).toBe(total - inLibrary)
    await byTestId(page, 'catalog-search').fill(`${target.brand} ${target.color_name}`)
    await expect(byTestId(page, `catalog-card-${target.id}`)).toHaveCount(0)
    await byTestId(page, 'catalog-hide-in-library').uncheck()
    await expect(byTestId(page, `catalog-card-${target.id}`)).toBeVisible()
  })

  test('matching a color ranks the closest filaments first', async ({ page, request }) => {
    const target = await uniqueEntry(request)
    await openCatalog(page)
    // The native color input is visually hidden behind its label.
    await byTestId(page, 'catalog-match-color').fill(target.color, { force: true })
    await expect(byTestId(page, 'catalog-sort')).toHaveValue('closest')
    const first = page.locator('[data-testid^="catalog-card-"]').first()
    await expect(first).toHaveAttribute('data-testid', /catalog-card-\d+/)
    // The exact color is in the catalog, so the top card is a perfect match.
    await expect(first.locator('[data-testid^="catalog-distance-"]')).toContainText('ΔE 0.0')

    await byTestId(page, 'catalog-match-clear').click()
    await expect(byTestId(page, 'catalog-sort')).toHaveValue('brand')
    await expect(page.locator('[data-testid^="catalog-distance-"]')).toHaveCount(0)
  })

  test('sorting by TD puts the most opaque filaments first', async ({ page, request }) => {
    const minTd = Math.min(...(await catalogEntries(request)).map((e) => e.td))
    await openCatalog(page)
    await byTestId(page, 'catalog-sort').selectOption('td')
    await expect(page.locator('[data-testid^="catalog-td-"]').first()).toHaveText(`TD ${minTd}`)
  })

  test('more results load on demand', async ({ page }) => {
    await openCatalog(page)
    const cards = page.locator('[data-testid^="catalog-card-"]')
    await expect(cards).toHaveCount(60)
    await byTestId(page, 'catalog-show-more').click()
    await expect.poll(() => cards.count()).toBeGreaterThanOrEqual(120)
  })

  test('no results offers to clear filters or create the filament yourself', async ({ page }) => {
    await openCatalog(page)
    await byTestId(page, 'catalog-search').fill('zzqqxx no such filament')
    await expect(byTestId(page, 'catalog-empty')).toBeVisible()
    await byTestId(page, 'catalog-clear-filters').click()
    await expect(byTestId(page, 'catalog-search')).toHaveValue('')
    await expect(page.locator('[data-testid^="catalog-card-"]').first()).toBeVisible()

    await byTestId(page, 'catalog-search').fill('zzqqxx no such filament')
    await byTestId(page, 'catalog-create-own').click()
    await expect(byTestId(page, 'catalog-modal')).toHaveCount(0)
    await expect(byTestId(page, 'new-filament-modal')).toBeVisible()
  })

  test('a failed load shows an error with a working retry', async ({ page }) => {
    let fail = true
    await page.route('**/api/filaments/catalog', (route) => (fail ? route.fulfill({ status: 500, json: { detail: 'boom' } }) : route.continue()))
    await openApp(page)
    await byTestId(page, 'catalog-btn').click()
    await expect(byTestId(page, 'catalog-error')).toBeVisible()
    fail = false
    await byTestId(page, 'catalog-error').getByRole('button', { name: 'Try again' }).click()
    await expect(page.locator('[data-testid^="catalog-card-"]').first()).toBeVisible()
  })

  test('a failed add is reported and the card stays addable', async ({ page, request }) => {
    const target = await uniqueEntry(request)
    await page.route('**/api/filaments/catalog/add', (route) => route.fulfill({ status: 500, json: { detail: 'disk full' } }))
    await openCatalog(page)
    await byTestId(page, 'catalog-search').fill(`${target.brand} ${target.color_name}`)
    await byTestId(page, `catalog-add-${target.id}`).click()
    await expect(byTestId(page, 'toast-error')).toContainText('disk full')
    await expect(byTestId(page, `catalog-add-${target.id}`)).toBeEnabled()
  })

  test('the empty library links to the catalog', async ({ page }) => {
    await openApp(page)
    await byTestId(page, 'filter-input').fill('zzqqxx nothing here')
    await byTestId(page, 'empty-catalog-link').click()
    await expect(byTestId(page, 'catalog-modal')).toBeVisible()
  })
})
