/**
 * Atlas: market analysis from the NSE Pattern Atlas, inside OpenAlgo.
 *
 * The Atlas studies NSE's public data (the whole market, since 2015) every
 * evening: the market's regime, which sectors and stocks are gaining or losing
 * strength, each stock's stage and buying grade, warnings, historical chances,
 * screener lists, hidden groups and a research log. This page only shows that
 * analysis; it places no orders. A symbol opens on the chart terminal, where
 * any trade is the trader's own decision.
 */
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Play, RefreshCw } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router'
import {
  type AtlasEngineStatus,
  type AtlasFeed,
  type AtlasLive,
  type AtlasStock,
  atlasApi,
} from '@/api/atlas'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { chartHref } from '@/lib/trading/chartLink'

const pct = (x: number | null | undefined, d = 0) => (x == null ? '-' : `${(x * 100).toFixed(d)}%`)
const sgn = (x: number | null | undefined, d = 1) =>
  x == null ? '-' : `${x > 0 ? '+' : ''}${(x * 100).toFixed(d)}%`
/** Group returns arrive already in percent. */
const sgnP = (x: number | null | undefined, d = 1) =>
  x == null ? '-' : `${x > 0 ? '+' : ''}${x.toFixed(d)}%`
const tone = (x: number | null | undefined) =>
  x == null
    ? ''
    : x > 0
      ? 'text-emerald-600 dark:text-emerald-400'
      : x < 0
        ? 'text-red-600 dark:text-red-400'
        : ''

const PHASE: Record<string, string> = {
  Leading: 'bg-emerald-600 text-white',
  Improving: 'bg-sky-600 text-white',
  Weakening: 'bg-amber-500 text-black',
  Lagging: 'bg-red-600 text-white',
}
const STAGE: Record<number, string> = { 1: 'Base', 2: 'Uptrend', 3: 'Top', 4: 'Downtrend' }

function PhaseBadge({ phase }: { phase: string | null }) {
  if (!phase) return <span className="text-muted-foreground">-</span>
  return <Badge className={PHASE[phase] ?? ''}>{phase}</Badge>
}

function Sym({ s }: { s: string }) {
  return (
    <Link to={chartHref(s, 'NSE')} className="font-medium underline-offset-2 hover:underline">
      {s}
    </Link>
  )
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-lg border p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="text-xl font-semibold">{value}</div>
      {sub && <div className="text-xs text-muted-foreground">{sub}</div>}
    </div>
  )
}

function Overview({ f, live }: { f: AtlasFeed; live?: AtlasLive }) {
  const R = f.regime
  const state = R.states?.find((s) => s.name === R.now)
  const ranked = f.groups
    .filter((g) => g.rs_rating != null)
    .sort((a, b) => (b.rs_rating ?? 0) - (a.rs_rating ?? 0))
  const verdicts = f.broad_or_narrow.reduce<Record<string, number>>((acc, r) => {
    acc[r.verdict] = (acc[r.verdict] ?? 0) + 1
    return acc
  }, {})
  return (
    <div className="space-y-4">
      {live && live.indices.length > 0 && <LiveCard live={live} />}
      {R.now && (
        <Card>
          <CardHeader>
            <CardDescription>Market regime this week</CardDescription>
            <CardTitle className="text-2xl">{R.now}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <p>
              {state?.desc} In this state for {R.weeks_in_state} week
              {R.weeks_in_state === 1 ? '' : 's'}.
            </p>
            {state && (
              <p className="text-muted-foreground">
                In the past, Nifty 500 over the next 13 weeks from this state:{' '}
                {sgn(state.nifty500_next_13w.mean)} on average, up{' '}
                {pct(state.nifty500_next_13w.pos)} of the time ({state.nifty500_next_13w.n} weeks).
              </p>
            )}
          </CardContent>
        </Card>
      )}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat
          label="Nifty 500, last month"
          value={sgn(f.market.n500_1m)}
          sub={`3 months ${sgn(f.market.n500_3m)}`}
        />
        <Stat label="India VIX" value={f.market.vix?.toFixed(1) ?? '-'} />
        <Stat
          label="Stocks above 50-day average"
          value={pct(f.market.breadth.above50)}
          sub={`4 weeks ago ${pct((f.market.breadth.above50 ?? 0) - (f.market.breadth_4w_change.above50 ?? 0))}`}
        />
        <Stat label="Stocks above 200-day average" value={pct(f.market.breadth.above200)} />
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        {[
          ['Strongest sectors and themes', ranked.slice(0, 10)],
          ['Weakest sectors and themes', ranked.slice(-10).reverse()],
        ].map(([title, rows]) => (
          <Card key={title as string}>
            <CardHeader>
              <CardTitle className="text-base">{title as string}</CardTitle>
              <CardDescription>Strength rating 1-99 against Nifty 500</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableBody>
                  {(rows as AtlasFeed['groups']).map((g) => (
                    <TableRow key={g.key}>
                      <TableCell>{g.name}</TableCell>
                      <TableCell>
                        <PhaseBadge phase={g.phase} />
                      </TableCell>
                      <TableCell className="text-right">{g.rs_rating}</TableCell>
                      <TableCell className={`text-right ${tone(g.vs_n500_3m)}`}>
                        {sgnP(g.vs_n500_3m)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        ))}
      </div>
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Is the rise broad or narrow? (last month)</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-2">
          {Object.entries(verdicts).map(([k, v]) => (
            <Badge key={k} variant="outline">
              {k}: {v}
            </Badge>
          ))}
        </CardContent>
      </Card>
    </div>
  )
}

function Sectors({ f }: { f: AtlasFeed }) {
  const [kind, setKind] = useState('index')
  const rows = f.groups
    .filter((g) => g.kind === kind)
    .sort((a, b) => (b.rs_rating ?? -1) - (a.rs_rating ?? -1))
  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        {[
          ['index', 'Indices'],
          ['industry', 'Industries'],
        ].map(([k, l]) => (
          <Button
            key={k}
            size="sm"
            variant={kind === k ? 'default' : 'outline'}
            onClick={() => setKind(k)}
          >
            {l}
          </Button>
        ))}
      </div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
            <TableHead>Phase</TableHead>
            <TableHead className="text-right">Strength</TableHead>
            <TableHead className="text-right">vs Nifty 500, 1m</TableHead>
            <TableHead className="text-right">3m</TableHead>
            <TableHead className="text-right">6m</TableHead>
            <TableHead className="text-right">Members above 50-day</TableHead>
            <TableHead className="text-right">Beat market in 3m, past odds</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((g) => (
            <TableRow key={g.key}>
              <TableCell>{g.name}</TableCell>
              <TableCell>
                <PhaseBadge phase={g.phase} />
              </TableCell>
              <TableCell className="text-right">{g.rs_rating ?? '-'}</TableCell>
              <TableCell className={`text-right ${tone(g.vs_n500_1m)}`}>
                {sgnP(g.vs_n500_1m)}
              </TableCell>
              <TableCell className={`text-right ${tone(g.vs_n500_3m)}`}>
                {sgnP(g.vs_n500_3m)}
              </TableCell>
              <TableCell className={`text-right ${tone(g.vs_n500_6m)}`}>
                {sgnP(g.vs_n500_6m)}
              </TableCell>
              <TableCell className="text-right">
                {g.pct_above_50d == null ? '-' : `${g.pct_above_50d.toFixed(0)}%`}
              </TableCell>
              <TableCell className="text-right">{pct(g.p_beat_3m)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

function Screener({ f }: { f: AtlasFeed }) {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {f.screener.map((p) => (
        <Card key={p.key}>
          <CardHeader>
            <CardTitle className="text-base">{p.name}</CardTitle>
            <CardDescription>{p.rule}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <p className="text-muted-foreground">
              Past record: {sgn(p.avg_vs_n500_3m)} vs Nifty 500 over 3 months on average, ahead{' '}
              {pct(p.beat_n500_3m)} of the time.
            </p>
            <div className="flex flex-wrap gap-2">
              {p.today.length ? (
                p.today.slice(0, 40).map((s) => <Sym key={s} s={s} />)
              ) : (
                <span>None today.</span>
              )}
              {p.today.length > 40 && (
                <span className="text-muted-foreground">+{p.today.length - 40} more</span>
              )}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}

function StockRow({ x, today }: { x: AtlasStock; today?: number | null }) {
  return (
    <TableRow>
      <TableCell>
        <Sym s={x.symbol} />
        <div className="text-xs text-muted-foreground">{x.industry}</div>
      </TableCell>
      <TableCell>{x.stage ? `${x.stage} ${STAGE[x.stage] ?? ''}` : '-'}</TableCell>
      <TableCell>{x.grade ?? '-'}</TableCell>
      <TableCell className="text-right">{x.rs_rating ?? '-'}</TableCell>
      <TableCell>
        <PhaseBadge phase={x.sector_phase} />
      </TableCell>
      <TableCell className={`text-right ${tone(today)}`}>{sgn(today)}</TableCell>
      <TableCell className={`text-right ${tone(x.ret_1m)}`}>{sgn(x.ret_1m)}</TableCell>
      <TableCell className={`text-right ${tone(x.ret_3m)}`}>{sgn(x.ret_3m)}</TableCell>
      <TableCell className={`text-right ${tone(x.breakdown_3m?.own)}`}>
        {sgn(x.breakdown_3m?.own)}
      </TableCell>
      <TableCell className="text-right">{pct(x.p_beat_3m)}</TableCell>
      <TableCell className="text-right">{pct(x.p_big_fall_3m)}</TableCell>
      <TableCell>
        {x.pledge && <Badge variant="destructive">Pledge</Badge>}{' '}
        {x.insider_selling && <Badge variant="destructive">Insider selling</Badge>}
      </TableCell>
    </TableRow>
  )
}

function Stocks({ f, live }: { f: AtlasFeed; live?: AtlasLive }) {
  const [q, setQ] = useState('')
  const rows = useMemo(() => {
    const u = q.trim().toUpperCase()
    const base = u
      ? f.stocks.filter((x) => x.symbol.includes(u) || (x.industry ?? '').toUpperCase().includes(u))
      : [...f.stocks].sort((a, b) => (b.rs_rating ?? -1) - (a.rs_rating ?? -1))
    return base.slice(0, 150)
  }, [f.stocks, q])
  return (
    <div className="space-y-3">
      <Input
        placeholder="Find a stock or industry, e.g. KOTAKBANK or Healthcare"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        className="max-w-md"
      />
      <p className="text-xs text-muted-foreground">
        Stage 2 is an uptrend and 4 a downtrend. Grade A-E: buying (A) to selling (E) from volume
        and delivery. Own part: the 3-month move left after the market and the industry. Odds are
        from similar past cases, not advice.
        {!q && ' Showing the 150 strongest.'}
      </p>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Stock</TableHead>
            <TableHead>Stage</TableHead>
            <TableHead>Grade</TableHead>
            <TableHead className="text-right">Strength</TableHead>
            <TableHead>Sector</TableHead>
            <TableHead className="text-right">Today</TableHead>
            <TableHead className="text-right">1m</TableHead>
            <TableHead className="text-right">3m</TableHead>
            <TableHead className="text-right">Own part 3m</TableHead>
            <TableHead className="text-right">Beat market 3m</TableHead>
            <TableHead className="text-right">Big fall 3m</TableHead>
            <TableHead>Warnings</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((x) => (
            <StockRow key={x.symbol} x={x} today={live?.stocks[x.symbol]?.change} />
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

function Hidden({ f }: { f: AtlasFeed }) {
  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-3">
        {f.forces.map((x) => (
          <Card key={x.k}>
            <CardHeader>
              <CardDescription>
                Force {x.k}, explains {pct(x.share, 1)} of daily moves
              </CardDescription>
              <CardTitle className="text-base">{x.name}</CardTitle>
            </CardHeader>
            {x.k > 1 && (
              <CardContent className="text-sm">
                One side: {x.plus_industries.join(', ')}. Other side:{' '}
                {x.minus_industries.join(', ')}.
              </CardContent>
            )}
          </Card>
        ))}
      </div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Group (found from how stocks move)</TableHead>
            <TableHead className="text-right">Stocks</TableHead>
            <TableHead className="text-right">vs Nifty 500, 1m</TableHead>
            <TableHead className="text-right">3m</TableHead>
            <TableHead>State</TableHead>
            <TableHead>Main members</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {f.hidden_groups.map((g) => (
            <TableRow key={g.name + g.members[0]}>
              <TableCell>
                {g.name}
                <div className="text-xs text-muted-foreground">
                  {g.industries.map(([k, v]) => `${k} ${v}`).join(' · ')}
                </div>
              </TableCell>
              <TableCell className="text-right">{g.n}</TableCell>
              <TableCell className={`text-right ${tone(g.vs_n500_1m)}`}>
                {sgn(g.vs_n500_1m)}
              </TableCell>
              <TableCell className={`text-right ${tone(g.vs_n500_3m)}`}>
                {sgn(g.vs_n500_3m)}
              </TableCell>
              <TableCell>{g.state}</TableCell>
              <TableCell className="space-x-2">
                {g.members.slice(0, 6).map((s) => (
                  <Sym key={s} s={s} />
                ))}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

function Research({ f }: { f: AtlasFeed }) {
  if (!f.research.length)
    return <p className="text-sm text-muted-foreground">No research experiments yet.</p>
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {[...f.research].reverse().map((x) => (
        <Card key={x.id}>
          <CardHeader>
            <CardDescription>
              {x.id} · {x.verdict} · data to {x.data_to}
            </CardDescription>
            <CardTitle className="text-base">{x.title}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <p>{x.question}</p>
            <p className="text-muted-foreground">{x.conclusion || x.summary}</p>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}

function LiveCard({ live }: { live: AtlasLive }) {
  const rows = [...live.indices].sort((a, b) => (b.change ?? 0) - (a.change ?? 0))
  return (
    <Card>
      <CardHeader>
        <CardDescription>Today, live from your broker</CardDescription>
        <CardTitle className="text-base">Indices now</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-wrap gap-2">
        {rows.map((q) => (
          <Badge key={q.symbol} variant="outline" className={tone(q.change)}>
            {q.label}: {sgn(q.change, 2)}
          </Badge>
        ))}
      </CardContent>
    </Card>
  )
}

function EngineCard({
  st,
  onRun,
  busy,
}: {
  st?: AtlasEngineStatus
  onRun: (full: boolean) => void
  busy: boolean
}) {
  if (!st) return null
  const running = st.running
  const last = st.finished
    ? `Last update finished ${st.finished.replace('T', ' ')}${st.ok === false ? ', with some steps failed' : ''}.`
    : st.state === 'stopped'
      ? 'The last update stopped before it finished; the next one continues where it left off.'
      : 'No update has run on this computer yet.'
  return (
    <Card>
      <CardHeader className="flex flex-row flex-wrap items-start justify-between gap-2">
        <div>
          <CardDescription>Updates on this computer, from NSE's public data</CardDescription>
          <CardTitle className="text-base">
            {running ? `Updating: ${st.step ?? 'starting'}` : last}
          </CardTitle>
          {!running && st.next_run && (
            <CardDescription>
              Next scheduled update: {st.next_run.replace('T', ' ').slice(0, 16)}
            </CardDescription>
          )}
          {running && st.first_run && (
            <CardDescription>
              First update: downloading NSE history since 2015 takes a few hours. You can keep using
              OpenAlgo meanwhile; if it stops, the next update continues where it left off.
            </CardDescription>
          )}
        </div>
        <div className="flex gap-2">
          <Button size="sm" onClick={() => onRun(false)} disabled={running || busy}>
            <Play className="mr-2 h-4 w-4" />
            Update now
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => onRun(true)}
            disabled={running || busy}
          >
            Update and relearn patterns
          </Button>
        </div>
      </CardHeader>
      {(running || st.steps.length > 0) && (
        <CardContent>
          <details>
            <summary className="cursor-pointer text-sm text-muted-foreground">
              {st.steps.length} steps finished{running ? ' so far' : ''}; show the log
            </summary>
            <pre className="mt-2 max-h-64 overflow-auto rounded bg-muted p-2 text-xs">
              {st.log.join('\n')}
            </pre>
          </details>
        </CardContent>
      )}
    </Card>
  )
}

export default function Atlas() {
  const qc = useQueryClient()
  const { data, error, isLoading, isFetching } = useQuery({
    queryKey: ['atlas-feed'],
    queryFn: atlasApi.getFeed,
    staleTime: 10 * 60 * 1000,
    retry: false,
  })
  const reload = async () => {
    await atlasApi.refresh()
    await qc.invalidateQueries({ queryKey: ['atlas-feed'] })
  }
  const [busy, setBusy] = useState(false)
  const [runMsg, setRunMsg] = useState('')
  const { data: st, refetch: refetchStatus } = useQuery({
    queryKey: ['atlas-engine'],
    queryFn: atlasApi.engineStatus,
    refetchInterval: (q) => (q.state.data?.running ? 5000 : 60000),
  })
  // a finished update means a new feed: read it again
  const lastFinished = useRef<string | null | undefined>(undefined)
  useEffect(() => {
    if (
      lastFinished.current !== undefined &&
      st?.finished &&
      st.finished !== lastFinished.current
    ) {
      qc.invalidateQueries({ queryKey: ['atlas-feed'] })
    }
    lastFinished.current = st?.finished ?? null
  }, [st?.finished, qc])
  const runEngine = async (full: boolean) => {
    setBusy(true)
    try {
      setRunMsg(await atlasApi.runEngine(full))
    } catch (e) {
      setRunMsg((e as Error).message)
    } finally {
      setBusy(false)
      refetchStatus()
    }
  }
  const liveSymbols = useMemo(() => {
    if (!data) return []
    const top = [...data.feed.stocks]
      .sort((a, b) => (b.rs_rating ?? -1) - (a.rs_rating ?? -1))
      .slice(0, 150)
      .map((x) => x.symbol)
    const lists = data.feed.screener.flatMap((p) => p.today)
    return [...new Set([...top, ...lists])].slice(0, 200)
  }, [data])
  const { data: live } = useQuery({
    queryKey: ['atlas-live', liveSymbols.length],
    queryFn: () => atlasApi.live(liveSymbols),
    enabled: liveSymbols.length > 0,
    refetchInterval: 60000,
    retry: false,
  })
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h1 className="text-2xl font-bold">Atlas</h1>
          <p className="text-sm text-muted-foreground">
            Market analysis from NSE's public data, updated each evening; today's prices from your
            broker.
            {data && ` Data to ${data.feed.date}.`} Odds and history, not advice. Nothing here
            places an order.
          </p>
        </div>
        <Button size="sm" variant="outline" onClick={reload} disabled={isFetching}>
          <RefreshCw className="mr-2 h-4 w-4" />
          Reload
        </Button>
      </div>
      <EngineCard st={st} onRun={runEngine} busy={busy} />
      {runMsg && <p className="text-sm text-muted-foreground">{runMsg}</p>}
      {isLoading && <p className="text-sm text-muted-foreground">Loading the analysis...</p>}
      {error && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">The Atlas analysis is not available</CardTitle>
            <CardDescription>{(error as Error).message}</CardDescription>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground">
            Press Update now above to build the analysis on this computer from NSE's public data
            (the first time takes a few hours), or connect the cloud Atlas with ATLAS_FEED_PATH or
            the three ATLAS_D1_ settings in .env.
          </CardContent>
        </Card>
      )}
      {data && (
        <Tabs defaultValue="overview">
          <TabsList className="flex-wrap">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="sectors">Sectors</TabsTrigger>
            <TabsTrigger value="screener">Screener</TabsTrigger>
            <TabsTrigger value="stocks">Stocks</TabsTrigger>
            <TabsTrigger value="hidden">Hidden groups</TabsTrigger>
            <TabsTrigger value="research">Research</TabsTrigger>
          </TabsList>
          <TabsContent value="overview">
            <Overview f={data.feed} live={live} />
          </TabsContent>
          <TabsContent value="sectors">
            <Sectors f={data.feed} />
          </TabsContent>
          <TabsContent value="screener">
            <Screener f={data.feed} />
          </TabsContent>
          <TabsContent value="stocks">
            <Stocks f={data.feed} live={live} />
          </TabsContent>
          <TabsContent value="hidden">
            <Hidden f={data.feed} />
          </TabsContent>
          <TabsContent value="research">
            <Research f={data.feed} />
          </TabsContent>
        </Tabs>
      )}
    </div>
  )
}
