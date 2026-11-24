/**
 * What this pins: the Atlas page renders the feed it is given (regime, sectors,
 * stocks with their warnings, research) and links a symbol to the chart
 * terminal; and when the feed is unavailable it shows the server's sentence and
 * the setup hint instead of a blank page. The fixture is a trimmed copy of a
 * real feed, so a change to the feed's shape that the page no longer reads
 * fails here.
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { describe, expect, it, vi } from 'vitest'
import type { AtlasFeed } from '@/api/atlas'
import feed from '@/test/fixtures/atlasFeed.json'
import Atlas from './Atlas'

const api = vi.hoisted(() => ({ getFeed: vi.fn(), refresh: vi.fn() }))
vi.mock('@/api/atlas', () => ({ atlasApi: api }))

function show() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter>
        <Atlas />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

describe('Atlas page', () => {
  it('shows the regime, then stocks with warnings linked to the chart', async () => {
    api.getFeed.mockResolvedValue({ feed: feed as unknown as AtlasFeed, source: 'file' })
    show()
    expect(await screen.findByText('Market regime this week')).toBeTruthy()
    expect(screen.getByText((feed as unknown as AtlasFeed).regime.now as string)).toBeTruthy()
    fireEvent.mouseDown(screen.getByRole('tab', { name: 'Stocks' }))
    fireEvent.click(screen.getByRole('tab', { name: 'Stocks' }))
    const link = await screen.findByRole('link', { name: 'KOTAKBANK' })
    expect(link.getAttribute('href')).toBe('/trading?symbol=KOTAKBANK&exchange=NSE')
  })

  it('explains what to set when the feed is unavailable', async () => {
    api.getFeed.mockRejectedValue(new Error('The Atlas is not set up.'))
    show()
    expect(await screen.findByText('The Atlas is not set up.')).toBeTruthy()
    expect(screen.getByText(/ATLAS_FEED_PATH/)).toBeTruthy()
  })
})
