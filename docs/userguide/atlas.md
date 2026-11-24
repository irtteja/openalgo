# Atlas

The **Atlas** page (`/atlas`, also under Tools and the profile menu) shows market analysis made by the NSE Pattern
Atlas, a separate system that studies NSE's public data for the whole market since 2015 and writes its results each
evening. OpenAlgo only reads and shows them. **Nothing on this page places an order**; a symbol opens on the chart
terminal, where any trade is your own decision.

## What it shows

| Tab | What you see |
|---|---|
| Overview | The market's regime this week (strong uptrend, steady, weak, stress) and what followed it in the past; Nifty 500, VIX and breadth; the strongest and weakest sectors; whether sector rises are broad or narrow |
| Sectors | Every NSE index and industry: phase (Leading, Weakening, Lagging, Improving), strength 1-99 against Nifty 500, moves against Nifty 500, members above their 50-day average, past odds of beating the market |
| Screener | The Atlas's screener lists with their past record and today's names |
| Stocks | Each index stock: stage, buying grade A-E, strength, sector phase, 1- and 3-month moves, its own part of the move, past odds of beating the market and of a big fall, and pledge or insider-selling warnings |
| Hidden groups | Groups of stocks that actually move together, found from their daily moves, and the few forces moving the market |
| Research | Questions the Atlas tested on history and what it found |

The odds are how often similar past cases did something; they are not advice.

## Updating on this computer

OpenAlgo includes the Atlas engine (`atlas_engine/`). It downloads NSE's public data from this computer (daily prices
and delivery for every stock, index levels and members, insider trades and pledges, F&O positions, results, VIX) and
rebuilds the analysis every weekday evening at 7:30 pm IST (`ATLAS_ENGINE_TIME`). The Atlas page shows its progress
and has **Update now** and **Update and relearn patterns** buttons.

- **The first update** downloads NSE history since 2015: about 2 GB and a few hours. Start it from the Atlas page.
  OpenAlgo stays usable meanwhile, and an update that stops continues where it left off next time.
- **Every later update** takes about 10 to 20 minutes, and about an hour once a week when the patterns are relearned.
- If the computer was off at 7:30 pm, the update runs a couple of minutes after OpenAlgo next starts.
- It runs as a separate process, so OpenAlgo's trading screens are not slowed by it.

During market hours the page also shows **today's moves** for the main indices and the listed stocks, from your
broker's quotes. That is a quote request only; it never places an order.

## Other sources

Instead of, or as well as, the engine on this computer:

- **A local file:** if the Atlas PC app runs on the same computer, set `ATLAS_FEED_PATH` to its
  `explorer/data/atlas_feed.json`.
- **Cloudflare D1:** if the Atlas uploads to a Cloudflare D1 database, set `ATLAS_D1_ACCOUNT_ID`,
  `ATLAS_D1_DATABASE_ID` and `ATLAS_D1_API_TOKEN`. Create a token with only **Account - D1 - Read**. One page load
  reads about ten small rows, far inside Cloudflare's free daily allowance.

The feed is kept for 15 minutes (`ATLAS_CACHE_SECONDS`). **Reload** on the page fetches the latest at once, for example
after the Atlas's evening update.

## Installing on Windows with one file

`installer/windows/OpenAlgoAtlasSetup.exe` installs OpenAlgo with the Atlas on a Windows PC: download it, double-click
it and follow the black setup window. It needs no administrator rights and no other software; it downloads uv
(which brings Python), the latest code of this branch and the libraries, into `%USERPROFILE%\OpenAlgoAtlas`, then
makes two desktop icons:

- **OpenAlgo Atlas** starts OpenAlgo and opens it in the browser. Keep its window open; closing it stops OpenAlgo.
- **Update OpenAlgo Atlas** downloads the latest version. Close OpenAlgo first.

The setup also offers to start OpenAlgo whenever you sign in to Windows, so the 7:30 pm update runs. Running it again
updates the program files only: `.env`, `db\` (your account and broker keys) and `atlas_engine\data` are never
touched. Windows may warn that the file is from an unknown publisher, because it is not code-signed; choose
**More info** and then **Run anyway**.

To rebuild the file after changing `install.ps1`: `makensis installer/windows/OpenAlgoAtlasSetup.nsi`.
