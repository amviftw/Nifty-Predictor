"""
Nifty Next 50 stocks — the largecap names that sit just outside Nifty 50.

Together with NIFTY50_STOCKS this gives us the full Nifty 100 (true largecap)
universe used by the dashboard. Kept separate so the model-training pipeline
can stay anchored on Nifty 50 while the dashboard analytics span largecap.

Update after each Nifty Next 50 rebalancing (March / September).
"""

LARGECAP_NEXT50_STOCKS = {
    # Symbol: (Yahoo ticker, Company Name, Sector)
    "ABB": ("ABB.NS", "ABB India", "Capital Goods"),
    "ADANIGREEN": ("ADANIGREEN.NS", "Adani Green Energy", "Power"),
    "ADANIPOWER": ("ADANIPOWER.NS", "Adani Power", "Power"),
    "AMBUJACEM": ("AMBUJACEM.NS", "Ambuja Cements", "Cement"),
    "ATGL": ("ATGL.NS", "Adani Total Gas", "Oil & Gas"),
    "BAJAJHLDNG": ("BAJAJHLDNG.NS", "Bajaj Holdings & Investment", "Financial Services"),
    "BANKBARODA": ("BANKBARODA.NS", "Bank of Baroda", "Banking"),
    "BEL": ("BEL.NS", "Bharat Electronics", "Defence"),
    "BOSCHLTD": ("BOSCHLTD.NS", "Bosch", "Auto"),
    "CHOLAFIN": ("CHOLAFIN.NS", "Cholamandalam Inv. & Finance", "Financial Services"),
    "CGPOWER": ("CGPOWER.NS", "CG Power & Industrial", "Capital Goods"),
    "DABUR": ("DABUR.NS", "Dabur India", "FMCG"),
    "DLF": ("DLF.NS", "DLF", "Realty"),
    "DMART": ("DMART.NS", "Avenue Supermarts", "Retail"),
    "GAIL": ("GAIL.NS", "GAIL India", "Oil & Gas"),
    "GODREJCP": ("GODREJCP.NS", "Godrej Consumer Products", "FMCG"),
    "HAL": ("HAL.NS", "Hindustan Aeronautics", "Defence"),
    "HAVELLS": ("HAVELLS.NS", "Havells India", "Consumer Durables"),
    "HINDZINC": ("HINDZINC.NS", "Hindustan Zinc", "Metals"),
    "HYUNDAI": ("HYUNDAI.NS", "Hyundai Motor India", "Auto"),
    "ICICIGI": ("ICICIGI.NS", "ICICI Lombard", "Insurance"),
    "ICICIPRULI": ("ICICIPRULI.NS", "ICICI Prudential Life", "Insurance"),
    "INDHOTEL": ("INDHOTEL.NS", "Indian Hotels", "Hospitality"),
    "INDIGO": ("INDIGO.NS", "InterGlobe Aviation", "Aviation"),
    "INDUSTOWER": ("INDUSTOWER.NS", "Indus Towers", "Telecom"),
    "IOC": ("IOC.NS", "Indian Oil Corporation", "Oil & Gas"),
    "IRCTC": ("IRCTC.NS", "Indian Railway Catering", "Consumer Services"),
    "IRFC": ("IRFC.NS", "Indian Railway Finance", "Financial Services"),
    "JINDALSTEL": ("JINDALSTEL.NS", "Jindal Steel & Power", "Metals"),
    "JIOFIN": ("JIOFIN.NS", "Jio Financial Services", "Financial Services"),
    "LICI": ("LICI.NS", "Life Insurance Corporation", "Insurance"),
    "LODHA": ("LODHA.NS", "Macrotech Developers (Lodha)", "Realty"),
    "MARICO": ("MARICO.NS", "Marico", "FMCG"),
    "MOTHERSON": ("MOTHERSON.NS", "Samvardhana Motherson", "Auto"),
    "NAUKRI": ("NAUKRI.NS", "Info Edge (Naukri)", "Consumer Services"),
    "NHPC": ("NHPC.NS", "NHPC", "Power"),
    "NMDC": ("NMDC.NS", "NMDC", "Mining"),
    "PFC": ("PFC.NS", "Power Finance Corporation", "Financial Services"),
    "PIDILITIND": ("PIDILITIND.NS", "Pidilite Industries", "Chemicals"),
    "RECLTD": ("RECLTD.NS", "REC", "Financial Services"),
    "SBICARD": ("SBICARD.NS", "SBI Cards & Payment", "Financial Services"),
    "SHREECEM": ("SHREECEM.NS", "Shree Cement", "Cement"),
    "SHRIRAMFIN": ("SHRIRAMFIN.NS", "Shriram Finance", "Financial Services"),
    "SIEMENS": ("SIEMENS.NS", "Siemens", "Capital Goods"),
    "SWIGGY": ("SWIGGY.NS", "Swiggy", "Consumer Services"),
    "TATAPOWER": ("TATAPOWER.NS", "Tata Power", "Power"),
    "TRENT": ("TRENT.NS", "Trent", "Retail"),
    "TVSMOTOR": ("TVSMOTOR.NS", "TVS Motor Company", "Auto"),
    "UNITDSPR": ("UNITDSPR.NS", "United Spirits", "FMCG"),
    "VBL": ("VBL.NS", "Varun Beverages", "FMCG"),
    "VEDL": ("VEDL.NS", "Vedanta", "Metals"),
    "ZOMATO": ("ZOMATO.NS", "Eternal (Zomato)", "Consumer Services"),
    "ZYDUSLIFE": ("ZYDUSLIFE.NS", "Zydus Lifesciences", "Pharma"),
}


def get_largecap_extra_yahoo_tickers() -> list[str]:
    return [info[0] for info in LARGECAP_NEXT50_STOCKS.values()]


def get_largecap_extra_symbols() -> list[str]:
    return list(LARGECAP_NEXT50_STOCKS.keys())
