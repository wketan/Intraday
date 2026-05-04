# 📈 Intraday Options Signal Engine

Real-time NIFTY/BANKNIFTY/FINNIFTY options trading signals with live Angel One integration.

## Architecture

```
┌─────────────────────────┐     ┌──────────────────────────┐
│   GitHub Pages (Free)   │     │   Render.com (Free)      │
│                         │     │                          │
│   index.html            │────▶│   server.py              │
│   (Dashboard UI)        │ API │   (Signal Engine)        │
│                         │     │   - Angel One API        │
│   Works on phone! 📱    │     │   - Instrument Master    │
│                         │     │   - Option Chain          │
└─────────────────────────┘     │   - AI Analysis          │
  yourname.github.io/           │   - Slack Alerts         │
  trading-dashboard             └──────────────────────────┘
                                  your-app.onrender.com
```

## Quick Setup (10 minutes)

### Step 1: Create GitHub Repo

```bash
# On your Mac
cd ~/Desktop
git clone https://github.com/YOUR_USERNAME/trading-dashboard.git
# Copy all files into this folder
cd trading-dashboard
git add .
git commit -m "Initial commit"
git push
```

### Step 2: Enable GitHub Pages

1. Go to your repo → **Settings** → **Pages**
2. Source: **Deploy from a branch**
3. Branch: **main** → **/ (root)**
4. Save → Your dashboard is live at `https://YOUR_USERNAME.github.io/trading-dashboard/`

### Step 3: Deploy Server to Render (Free)

1. Go to [render.com](https://render.com) → Sign up (free)
2. **New** → **Web Service** → Connect your GitHub repo
3. Settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn server:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --threads 4`
4. **Environment Variables** (add these):
   ```
   ANGEL_API_KEY=your_key
   ANGEL_CLIENT_ID=your_id
   ANGEL_PASSWORD=your_password
   ANGEL_TOTP_SECRET=your_totp_secret
   ANTHROPIC_API_KEY=sk-ant-...    (optional, for AI analysis)
   SLACK_WEBHOOK=https://hooks...   (optional, for Slack alerts)
   ```
5. Click **Create Web Service** → Wait 2-3 min for deploy

### Step 4: Connect Dashboard to Server

1. Open your GitHub Pages dashboard on phone
2. Click ⚙️ gear icon in header
3. Enter your Render URL: `https://your-app.onrender.com`
4. Click **Save & Reload**
5. Done! 🎉

## Local Development

For running on your laptop (faster, no cloud needed):

```bash
# Create .env file with your credentials
cp .env.example .env
# Edit .env with your actual credentials
nano .env

# Install dependencies
pip install -r requirements.txt

# Run server
python server.py

# Open dashboard
open http://localhost:5050/dashboard
```

## Features

- **5 Strategies**: ORB, VWAP Bounce, EMA+RSI, SuperTrend, PDH/PDL
- **Live Option Chain**: Real prices via Angel One batch API (1 call for 14 options)
- **AI Analysis**: Claude Sonnet reviews each signal (TAKE/SKIP/WAIT)
- **Browser Notifications**: Audio + visual alerts when signals fire
- **Profit Calculator**: Quick P&L calculation for any trade
- **Slack Alerts**: Real-time signal notifications
- **Replay Mode**: Backtest on historical data

## Phone Usage (Add to Home Screen)

**iPhone**: Open dashboard in Safari → Share → Add to Home Screen
**Android**: Open in Chrome → Menu → Add to Home Screen

This gives you a full-screen app experience with push notifications.

## Important Notes

- **Render free tier** spins down after 15 min of inactivity. First request takes ~30s to wake up — meaning you'll miss the opening-range window if the dyno is asleep at 9:15. **Recommended: switch to Fly.io free tier** (see below) which stays warm 24/7.
- **Market hours only**: Engine scans 9:20 AM until the auto-close cutoff (default 15:15 IST, configurable).
- **Budget**: Configured for ₹20K capital, max 50% per trade.
- **Never commit `.env`** — it contains your credentials.

## Backtesting

Before risking real capital on any strategy tweak, run the backtest:

```bash
python backtest.py --instrument NIFTY --days 30 --csv out.csv
python backtest.py --all --days 60                 # all three indices
```

Reports win-rate, gross/net P&L (after estimated brokerage + slippage), expectancy
per trade, and max drawdown. **Note**: option premiums are estimated (Angel One has no
historical option-chain API), so absolute rupee figures are ±20% — use the win-rate
and expectancy as the primary edge signals, not the rupee total.

## Risk guards

The engine has a daily kill-switch that latches when EITHER limit is hit:

- `DAILY_LOSS_LIMIT` rupees of net P&L (default ₹2000)
- `MAX_TRADES_PER_DAY` total signals (default 8)

On trip, no new alerts fire for the rest of the day. You can also trip/reset
manually via `POST /api/killswitch {"action":"trip"|"reset"}` (auth required).

## Deploy to Fly.io (recommended)

```bash
brew install flyctl                   # or: curl -L https://fly.io/install.sh | sh
fly auth signup
fly launch --no-deploy --copy-config  # uses the included fly.toml + Dockerfile
fly secrets set ANGEL_API_KEY=... ANGEL_CLIENT_ID=... ANGEL_PASSWORD=... \
               ANGEL_TOTP_SECRET=... ANTHROPIC_API_KEY=... \
               SLACK_WEBHOOK=... AUTH_TOKEN=...
fly deploy
```

After deploy, point the dashboard at `https://<app-name>.fly.dev` via the gear icon.
Fly's `bom` (Mumbai) region is closest to NSE for the lowest API latency.
