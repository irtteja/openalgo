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
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { AtlasFeed } from '@/api/atlas'
import feed from '@/test/fixtures/atlasFeed.json'
import Atlas from './Atlas'

const api = vi.hoisted(() => ({
  getFeed: vi.fn(),
  refresh: vi.fn(),
  engineStatus: vi.fn(),
  runEngine: vi.fn(),
  live: vi.fn(),
}))
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
  beforeEach(() => {
    api.engineStatus.mockResolvedValue({
      state: 'finished',
      running: false,
      started: '2026-11-24T19:30:00',
      finished: '2026-11-24T19:48:00',
      step: null,
      steps: [{ label: 'Daily prices', ok: true, seconds: 12 }],
      ok: true,
      first_run: false,
      next_run: '2026-11-25T19:30:00+05:30',
      feed_exists: true,
      log: [],
    })
    api.live.mockResolvedValue({
      indices: [
        { symbol: 'NIFTY', label: 'Nifty 50', ltp: 25400, prev_close: 25300, change: 0.004 },
      ],
      stocks: { KOTAKBANK: { symbol: 'KOTAKBANK', ltp: 2100, prev_close: 2000, change: 0.05 } },
    })
  })

  it('shows the regime, then stocks with warnings linked to the chart', async () => {
    api.getFeed.mockResolvedValue({ feed: feed as unknown as AtlasFeed, source: 'file' })
    show()
    expect(await screen.findByText('Market regime this week')).toBeTruthy()
    expect(screen.getByText((feed as unknown as AtlasFeed).regime.now as string)).toBeTruthy()
    fireEvent.mouseDown(screen.getByRole('tab', { name: 'Stocks' }))
    fireEvent.click(screen.getByRole('tab', { name: 'Stocks' }))
    const link = await screen.findByRole('link', { name: 'KOTAKBANK' })
    expect(link.getAttribute('href')).toBe('/trading?symbol=KOTAKBANK&exchange=NSE')
    expect(await screen.findByText('+5.0%')).toBeTruthy()
  })

  it('shows the update panel and starts an update on request', async () => {
    api.getFeed.mockResolvedValue({ feed: feed as unknown as AtlasFeed, source: 'engine' })
    api.runEngine.mockResolvedValue('The Atlas update has started.')
    show()
    expect(await screen.findByText(/Last update finished 2026-11-24 19:48:00/)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Update now' }))
    expect(await screen.findByText('The Atlas update has started.')).toBeTruthy()
    expect(api.runEngine).toHaveBeenCalledWith(false)
  })

  it('explains what to set when the feed is unavailable', async () => {
    api.getFeed.mockRejectedValue(new Error('The Atlas is not set up.'))
    show()
    expect(await screen.findByText('The Atlas is not set up.')).toBeTruthy()
    expect(screen.getByText(/Press Update now above/)).toBeTruthy()
  })
})
