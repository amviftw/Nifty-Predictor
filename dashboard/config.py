"""
Dashboard-specific configuration constants.
Sectoral index tickers, supply chain tickers, sector-commodity mappings, and layout settings.
"""

from config.nifty50_tickers import SECTOR_INDICES

# Sectoral/thematic index tickers for Yahoo Finance.
# Keep this display-name layer in sync with the canonical symbols in
# config.nifty50_tickers.SECTOR_INDICES.
SECTORAL_INDICES = {
    "Nifty Auto": SECTOR_INDICES["Auto"],
    "Nifty Bank": SECTOR_INDICES["Banking"],
    "Nifty Chemicals": SECTOR_INDICES["Chemicals"],
    "Nifty Consumer Durables": SECTOR_INDICES["Consumer Durables"],
    "Nifty Energy": SECTOR_INDICES["Energy"],
    "Nifty Financial Services": SECTOR_INDICES["Financial Services"],
    "Nifty FMCG": SECTOR_INDICES["FMCG"],
    "Nifty Healthcare": SECTOR_INDICES["Healthcare"],
    "Nifty Infra": SECTOR_INDICES["Infrastructure"],
    "Nifty IT": SECTOR_INDICES["IT"],
    "Nifty Media": SECTOR_INDICES["Media"],
    "Nifty Metal": SECTOR_INDICES["Metals"],
    "Nifty Oil & Gas": SECTOR_INDICES["Oil & Gas"],
    "Nifty Pharma": SECTOR_INDICES["Pharma"],
    "Nifty Private Bank": SECTOR_INDICES["Private Bank"],
    "Nifty PSU Bank": SECTOR_INDICES["PSU Bank"],
    "Nifty Realty": SECTOR_INDICES["Realty"],
    "Nifty India Defence": SECTOR_INDICES["India Defence"],
}

# Maps dashboard sectoral index names back to sector names used in NIFTY50_STOCKS
SECTOR_INDEX_TO_SECTOR = {
    "Nifty Auto": "Auto",
    "Nifty Bank": "Banking",
    "Nifty Chemicals": "Chemicals",
    "Nifty Consumer Durables": "Consumer Goods",
    "Nifty Energy": "Oil & Gas",
    "Nifty Financial Services": "Financial Services",
    "Nifty FMCG": "FMCG",
    "Nifty Healthcare": "Healthcare",
    "Nifty Infra": "Infrastructure",
    "Nifty IT": "IT",
    "Nifty Media": "Media",
    "Nifty Metal": "Metals",
    "Nifty Oil & Gas": "Oil & Gas",
    "Nifty Pharma": "Pharma",
    "Nifty Private Bank": "Banking",
    "Nifty PSU Bank": "Banking",
    "Nifty Realty": "Realty",
    "Nifty India Defence": "Defence",
}

# Supply chain / international factor tickers
SUPPLY_CHAIN_TICKERS = {
    "Crude Oil (WTI)": "CL=F",
    "Brent Crude": "BZ=F",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Copper": "HG=F",
    "Natural Gas": "NG=F",
    "US 10Y Yield": "^TNX",
    "US Dollar Index": "DX-Y.NYB",
}

# Which supply chain factors affect which Indian sectors
SECTOR_SUPPLY_CHAIN = {
    "Oil & Gas": {
        "factors": ["Crude Oil (WTI)", "Brent Crude", "Natural Gas"],
        "note": "Upstream benefits from higher crude; downstream (refining/marketing) hurt by higher input costs",
    },
    "Metals": {
        "factors": ["Copper", "Gold", "Silver"],
        "note": "Metal prices directly impact revenue for steel, aluminium, and mining companies",
    },
    "IT": {
        "factors": ["US Dollar Index"],
        "note": "Weak USD/strong INR hurts IT export revenue; US recession fears reduce tech spending",
    },
    "Auto": {
        "factors": ["Crude Oil (WTI)", "Copper"],
        "note": "Higher crude raises fuel costs dampening demand; copper is key raw material",
    },
    "Pharma": {
        "factors": ["US Dollar Index"],
        "note": "US is largest export market; FDA actions and USD strength affect earnings",
    },
    "Banking": {
        "factors": ["US 10Y Yield"],
        "note": "Rising US yields attract FII outflows from Indian debt/equity; impacts liquidity",
    },
    "Financial Services": {
        "factors": ["US 10Y Yield"],
        "note": "FII flows and global risk appetite driven by US yield movements",
    },
    "FMCG": {
        "factors": ["Crude Oil (WTI)"],
        "note": "Crude impacts packaging and logistics costs; rural demand linked to agri-commodity prices",
    },
    "Infrastructure": {
        "factors": ["Crude Oil (WTI)", "Copper", "US 10Y Yield"],
        "note": "Construction materials (steel, cement) and fuel costs; interest rate sensitivity",
    },
    "Power": {
        "factors": ["Natural Gas", "Crude Oil (WTI)"],
        "note": "Fuel costs for thermal power generation; gas prices affect city gas distribution",
    },
    "Mining": {
        "factors": ["Copper", "Gold"],
        "note": "Global commodity prices directly drive mining revenues",
    },
    "Consumer Goods": {
        "factors": ["Crude Oil (WTI)"],
        "note": "Input cost inflation from crude; impacts margins on paints, chemicals",
    },
    "Cement": {
        "factors": ["Crude Oil (WTI)"],
        "note": "Energy-intensive industry; fuel and logistics costs tied to crude",
    },
    "Chemicals": {
        "factors": ["Crude Oil (WTI)", "Natural Gas", "US Dollar Index"],
        "note": "Feedstock, energy costs, exports, and China+1 supply chain shifts drive margins",
    },
    "Healthcare": {
        "factors": ["US Dollar Index"],
        "note": "Export-heavy earnings are sensitive to USD/INR and global regulatory events",
    },
    "Defence": {
        "factors": ["Copper", "US Dollar Index"],
        "note": "Order inflows, electronics inputs, import substitution, and rupee moves affect defence manufacturers",
    },
}

# Global index display names (mapped from config/nifty50_tickers.py GLOBAL_INDICES keys)
GLOBAL_INDEX_DISPLAY = {
    "SP500": "S&P 500",
    "NASDAQ": "NASDAQ",
    "DOW": "Dow Jones",
    "FTSE": "FTSE 100",
    "NIKKEI": "Nikkei 225",
    "HANGSENG": "Hang Seng",
}

# Cache TTL for Streamlit @st.cache_data.
# This is a *safety upper bound*; actual freshness is driven by the
# market-aware cache key in dashboard.data_loader._market_minute_bucket(),
# which rotates every minute during NSE market hours and at every new
# trading-day boundary after close. Keeping the decorator TTL short means
# stale data cannot linger longer than this even if the bucket key somehow
# stays constant.
CACHE_TTL_SECONDS = 60

# Number of top gainers/losers to show
TOP_MOVERS_COUNT = 10

# Data periods
PERIOD_DAILY = "5d"
PERIOD_WEEKLY = "1mo"

# Target Hunter thresholds
TARGET_HUNTER_MIN_UPSIDE = 0.15          # 15% minimum mean-target upside
TARGET_HUNTER_MIN_ANALYSTS = 5           # minimum analyst coverage
TARGET_HUNTER_TECH_SCORE_THRESHOLD = 40  # 0–100; filter weak-momentum names
ANALYST_CACHE_TTL_SECONDS = 1800         # 30 minutes — analyst data moves slowly
TARGET_REVISION_WINDOW_DAYS = 90         # how far back to show broker revisions
