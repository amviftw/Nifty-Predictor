"""
Dashboard-specific configuration constants.
Sectoral index tickers, supply chain tickers, sector-commodity mappings, and layout settings.
"""

# Sectoral index tickers for Yahoo Finance
# Extends config/nifty50_tickers.py SECTOR_INDICES with additional indices
SECTORAL_INDICES = {
    "Nifty Bank": "^NSEBANK",
    "Nifty IT": "^CNXIT",
    "Nifty Pharma": "^CNXPHARMA",
    "Nifty Auto": "^CNXAUTO",
    "Nifty FMCG": "^CNXFMCG",
    "Nifty Metal": "^CNXMETAL",
    "Nifty Infra": "^CNXINFRA",
    "Nifty Energy": "^CNXENERGY",
    "Nifty Fin Svc": "^CNXFIN",
    "Nifty Realty": "^CNXREALTY",
    "Nifty PSU Bank": "^CNXPSUBANK",
    "Nifty Media": "^CNXMEDIA",
}

# Maps dashboard sectoral index names back to sector names used in NIFTY50_STOCKS
SECTOR_INDEX_TO_SECTOR = {
    "Nifty Bank": "Banking",
    "Nifty IT": "IT",
    "Nifty Pharma": "Pharma",
    "Nifty Auto": "Auto",
    "Nifty FMCG": "FMCG",
    "Nifty Metal": "Metals",
    "Nifty Infra": "Infrastructure",
    "Nifty Energy": "Oil & Gas",
    "Nifty Fin Svc": "Financial Services",
    "Nifty Realty": "Realty",
    "Nifty PSU Bank": "Banking",
    "Nifty Media": "Media",
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

# Cache TTL for Streamlit @st.cache_data
CACHE_TTL_SECONDS = 300  # 5 minutes

# Number of top gainers/losers to show
TOP_MOVERS_COUNT = 10

# Data periods
PERIOD_DAILY = "5d"
PERIOD_WEEKLY = "1mo"
