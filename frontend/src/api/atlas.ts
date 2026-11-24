/**
 * The Atlas feed: market analysis written each evening by the NSE Pattern Atlas
 * from NSE's public data, read by blueprints/atlas.py from a local file or a
 * Cloudflare D1 database. Read-only; nothing here touches the broker.
 */
import { webClient } from './client'

export interface AtlasGroup {
  key: string
  name: string
  kind: string
  category: string | null
  phase: string | null
  rs_rating: number | null
  rs_trend: string | null
  vs_n500_1m: number | null
  vs_n500_3m: number | null
  vs_n500_6m: number | null
  vs_n500_1y: number | null
  from_high: number | null
  pct_above_50d: number | null
  breadth_change_1m: number | null
  in_base: boolean | null
  breakout: boolean | null
  p_beat_1m: number | null
  p_beat_3m: number | null
  p_up_3m: number | null
}

export interface AtlasStock {
  symbol: string
  industry: string | null
  price: number | null
  ret_1m: number | null
  ret_3m: number | null
  ret_1y: number | null
  from_high: number | null
  stage: number | null
  stage_weeks: number | null
  grade: string | null
  rs_rating: number | null
  rs_4w_ago: number | null
  vs_sector_3m: number | null
  vs_market_3m: number | null
  sector_phase: string | null
  above_50d: boolean | null
  above_200d: boolean | null
  in_base: boolean | null
  breakout: boolean | null
  delivery: number | null
  pe: number | null
  pledge: boolean | null
  insider_selling: boolean | null
  lists: string[]
  p_beat_1m: number | null
  p_beat_3m: number | null
  p_big_gain_3m: number | null
  p_big_fall_3m: number | null
  breakdown_3m: {
    tot: number | null
    mkt: number | null
    ind: number | null
    own: number | null
  } | null
}

export interface AtlasRegimeState {
  name: string
  desc: string
  share_of_weeks: number | null
  typical_weeks: number | null
  avg: Record<string, number | null>
  nifty500_next_4w: { mean: number | null; pos: number | null; n: number }
  nifty500_next_13w: { mean: number | null; pos: number | null; n: number }
  in_4w: Record<string, number | null>
}

export interface AtlasFeed {
  version: number
  date: string
  market: {
    regime_flags: string[]
    nifty500: number | null
    n500_1m: number | null
    n500_3m: number | null
    n500_1y: number | null
    n500_from_high: number | null
    vix: number | null
    breadth: Record<string, number | null>
    breadth_4w_change: Record<string, number | null>
  }
  regime: {
    now?: string
    weeks_in_state?: number
    probabilities?: Record<string, number | null>
    today?: Record<string, number | null>
    states?: AtlasRegimeState[]
    method?: string
  }
  groups: AtlasGroup[]
  stocks: AtlasStock[]
  screener: {
    key: string
    name: string
    rule: string
    avg_vs_n500_3m: number | null
    beat_n500_3m: number | null
    per_week: number | null
    today: string[]
  }[]
  forces: {
    k: number
    share: number | null
    name: string
    plus_industries: string[]
    minus_industries: string[]
    plus_stocks: string[]
    minus_stocks: string[]
    move_1m: number | null
    move_3m: number | null
  }[]
  hidden_groups: {
    name: string
    n: number
    industries: [string, number][]
    members: string[]
    togetherness: number | null
    vs_n500_1m: number | null
    vs_n500_3m: number | null
    above50: number | null
    state: string
  }[]
  broad_or_narrow: {
    name: string
    n: number
    ret_m1: number | null
    up_m1: number | null
    top3_share_m1: number | null
    top3_weight_m1: number | null
    top3_m1: string[]
    ret_m3: number | null
    up_m3: number | null
    verdict: string
  }[]
  research: {
    id: string
    title: string
    question: string
    method: string
    summary: string
    conclusion?: string
    verdict: string
    data_to: string
  }[]
  notes: string
}

interface Envelope {
  status: string
  source?: string
  data?: AtlasFeed
  message?: string
}

export interface AtlasEngineStatus {
  state: string
  running: boolean
  started: string | null
  finished: string | null
  step: string | null
  steps: { label: string; ok: boolean; seconds: number }[]
  ok: boolean | null
  first_run: boolean
  next_run: string | null
  feed_exists: boolean
  log: string[]
}

export interface AtlasLiveQuote {
  symbol: string
  ltp: number
  prev_close: number | null
  change: number | null
  label?: string
}

export interface AtlasLive {
  indices: AtlasLiveQuote[]
  stocks: Record<string, AtlasLiveQuote>
}

function errorText(e: unknown, fallback: string): string {
  const msg = (e as { response?: { data?: { message?: string } } })?.response?.data?.message
  return msg || (e instanceof Error ? e.message : fallback)
}

export const atlasApi = {
  async getFeed(): Promise<{ feed: AtlasFeed; source: string }> {
    try {
      const res = await webClient.get<Envelope>('/atlas/api/feed')
      if (res.data.status !== 'success' || !res.data.data) {
        throw new Error(res.data.message || 'The Atlas analysis could not be loaded.')
      }
      return { feed: res.data.data, source: res.data.source || '' }
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: Envelope } })?.response?.data?.message
      throw new Error(
        msg || (e instanceof Error ? e.message : 'The Atlas analysis could not be loaded.')
      )
    }
  },
  async refresh(): Promise<void> {
    await webClient.post('/atlas/api/refresh')
  },
  async engineStatus(): Promise<AtlasEngineStatus> {
    const res = await webClient.get<{ status: string; data: AtlasEngineStatus }>(
      '/atlas/api/engine/status'
    )
    return res.data.data
  },
  async runEngine(full = false): Promise<string> {
    try {
      const res = await webClient.post<{ status: string; message: string }>(
        '/atlas/api/engine/run',
        { full }
      )
      return res.data.message
    } catch (e: unknown) {
      throw new Error(errorText(e, 'The Atlas update could not be started.'))
    }
  },
  async live(symbols: string[]): Promise<AtlasLive> {
    try {
      const res = await webClient.post<{ status: string; data: AtlasLive; message?: string }>(
        '/atlas/api/live',
        {
          symbols,
        }
      )
      return res.data.data
    } catch (e: unknown) {
      throw new Error(errorText(e, 'Live prices are not available right now.'))
    }
  },
}
