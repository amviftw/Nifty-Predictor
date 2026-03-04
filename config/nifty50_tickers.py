"""
Nifty 50 stock tickers mapping: NSE Symbol -> Yahoo Finance ticker (.NS suffix).
Also includes sector classification for each stock.

Update this list after each Nifty 50 rebalancing (March and September).
Last updated: March 2026.
"""

NIFTY50_STOCKS = {
    # Symbol: (Yahoo ticker, Company Name, Sector)
    "ADANIENT": ("ADANIENT.NS", "Adani Enterprises", "Conglomerate"),
    "ADANIPORTS": ("ADANIPORTS.NS", "Adani Ports & SEZ", "Infrastructure"),
    "APOLLOHOSP": ("APOLLOHOSP.NS", "Apollo Hospitals", "Healthcare"),
    "ASIANPAINT": ("ASIANPAINT.NS", "Asian Paints", "Consumer Goods"),
    "AXISBANK": ("AXISBANK.NS", "Axis Bank", "Banking"),
    "BAJAJ-AUTO": ("BAJAJ-AUTO.NS", "Bajaj Auto", "Auto"),
    "BAJFINANCE": ("BAJFINANCE.NS", "Bajaj Finance", "Financial Services"),
    "BAJAJFINSV": ("BAJAJFINSV.NS", "Bajaj Finserv", "Financial Services"),
    "BPCL": ("BPCL.NS", "Bharat Petroleum", "Oil & Gas"),
    "BHARTIARTL": ("BHARTIARTL.NS", "Bharti Airtel", "Telecom"),
    "BRITANNIA": ("BRITANNIA.NS", "Britannia Industries", "FMCG"),
    "CIPLA": ("CIPLA.NS", "Cipla", "Pharma"),
    "COALINDIA": ("COALINDIA.NS", "Coal India", "Mining"),
    "DIVISLAB": ("DIVISLAB.NS", "Divi's Laboratories", "Pharma"),
    "DRREDDY": ("DRREDDY.NS", "Dr. Reddy's Labs", "Pharma"),
    "EICHERMOT": ("EICHERMOT.NS", "Eicher Motors", "Auto"),
    "GRASIM": ("GRASIM.NS", "Grasim Industries", "Cement"),
    "HCLTECH": ("HCLTECH.NS", "HCL Technologies", "IT"),
    "HDFCBANK": ("HDFCBANK.NS", "HDFC Bank", "Banking"),
    "HDFCLIFE": ("HDFCLIFE.NS", "HDFC Life Insurance", "Insurance"),
    "HEROMOTOCO": ("HEROMOTOCO.NS", "Hero MotoCorp", "Auto"),
    "HINDALCO": ("HINDALCO.NS", "Hindalco Industries", "Metals"),
    "HINDUNILVR": ("HINDUNILVR.NS", "Hindustan Unilever", "FMCG"),
    "ICICIBANK": ("ICICIBANK.NS", "ICICI Bank", "Banking"),
    "INDUSINDBK": ("INDUSINDBK.NS", "IndusInd Bank", "Banking"),
    "INFY": ("INFY.NS", "Infosys", "IT"),
    "ITC": ("ITC.NS", "ITC", "FMCG"),
    "JSWSTEEL": ("JSWSTEEL.NS", "JSW Steel", "Metals"),
    "KOTAKBANK": ("KOTAKBANK.NS", "Kotak Mahindra Bank", "Banking"),
    "LT": ("LT.NS", "Larsen & Toubro", "Infrastructure"),
    "M&M": ("M&M.NS", "Mahindra & Mahindra", "Auto"),
    "MARUTI": ("MARUTI.NS", "Maruti Suzuki", "Auto"),
    "NESTLEIND": ("NESTLEIND.NS", "Nestle India", "FMCG"),
    "NTPC": ("NTPC.NS", "NTPC", "Power"),
    "ONGC": ("ONGC.NS", "Oil & Natural Gas Corp", "Oil & Gas"),
    "POWERGRID": ("POWERGRID.NS", "Power Grid Corp", "Power"),
    "RELIANCE": ("RELIANCE.NS", "Reliance Industries", "Conglomerate"),
    "SBILIFE": ("SBILIFE.NS", "SBI Life Insurance", "Insurance"),
    "SBIN": ("SBIN.NS", "State Bank of India", "Banking"),
    "SUNPHARMA": ("SUNPHARMA.NS", "Sun Pharmaceutical", "Pharma"),
    "TATACONSUM": ("TATACONSUM.NS", "Tata Consumer Products", "FMCG"),
    "TATAMOTORS": ("TATAMOTORS.NS", "Tata Motors", "Auto"),
    "TATASTEEL": ("TATASTEEL.NS", "Tata Steel", "Metals"),
    "TCS": ("TCS.NS", "Tata Consultancy Services", "IT"),
    "TECHM": ("TECHM.NS", "Tech Mahindra", "IT"),
    "TITAN": ("TITAN.NS", "Titan Company", "Consumer Goods"),
    "ULTRACEMCO": ("ULTRACEMCO.NS", "UltraTech Cement", "Cement"),
    "UPL": ("UPL.NS", "UPL", "Chemicals"),
    "WIPRO": ("WIPRO.NS", "Wipro", "IT"),
    "LTIM": ("LTIM.NS", "LTIMindtree", "IT"),
}


def get_yahoo_tickers() -> list[str]:
    """Return list of Yahoo Finance tickers for all Nifty 50 stocks."""
    return [info[0] for info in NIFTY50_STOCKS.values()]


def get_symbols() -> list[str]:
    """Return list of NSE symbols."""
    return list(NIFTY50_STOCKS.keys())


def get_sector(symbol: str) -> str:
    """Return sector for a given NSE symbol."""
    return NIFTY50_STOCKS.get(symbol, (None, None, "Unknown"))[2]


def get_company_name(symbol: str) -> str:
    """Return company name for a given NSE symbol."""
    return NIFTY50_STOCKS.get(symbol, (None, "Unknown", None))[1]


def symbol_to_yahoo(symbol: str) -> str:
    """Convert NSE symbol to Yahoo Finance ticker."""
    return NIFTY50_STOCKS.get(symbol, (f"{symbol}.NS",))[0]


# Global index tickers for macro features
GLOBAL_INDICES = {
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "INDIAVIX": "^INDIAVIX",
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DOW": "^DJI",
    "FTSE": "^FTSE",
    "NIKKEI": "^N225",
    "HANGSENG": "^HSI",
    "USDINR": "INR=X",
}

# Sector indices for relative strength computation
SECTOR_INDICES = {
    "IT": "^CNXIT",
    "Banking": "^NSEBANK",
    "Pharma": "^CNXPHARMA",
    "Auto": "^CNXAUTO",
    "FMCG": "^CNXFMCG",
    "Metals": "^CNXMETAL",
    "Infrastructure": "^CNXINFRA",
    "Oil & Gas": "^CNXENERGY",
    "Financial Services": "^CNXFIN",
}
