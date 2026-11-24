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

## Setting it up

Choose one source in `.env`, then restart OpenAlgo:

- **A local file:** if the Atlas PC app runs on the same computer, set `ATLAS_FEED_PATH` to its
  `explorer/data/atlas_feed.json`.
- **Cloudflare D1:** if the Atlas uploads to a Cloudflare D1 database, set `ATLAS_D1_ACCOUNT_ID`,
  `ATLAS_D1_DATABASE_ID` and `ATLAS_D1_API_TOKEN`. Create a token with only **Account - D1 - Read**. One page load
  reads about ten small rows, far inside Cloudflare's free daily allowance.

The feed is kept for 15 minutes (`ATLAS_CACHE_SECONDS`). **Reload** on the page fetches the latest at once, for example
after the Atlas's evening update.
