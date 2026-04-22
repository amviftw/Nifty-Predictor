"""
Curated Nifty Midcap 100 subset for the analyst targets view.

Kept intentionally smaller (~50 names) than the full index to keep Yahoo
Finance fetch latency manageable for an interactive dashboard. Names are
selected for analyst coverage and liquidity.

Update after index rebalancing (March / September).
Last updated: April 2026.
"""

MIDCAP_STOCKS = {
    # Symbol: (Yahoo ticker, Company Name, Sector)
    "ASHOKLEY": ("ASHOKLEY.NS", "Ashok Leyland", "Auto"),
    "AUROPHARMA": ("AUROPHARMA.NS", "Aurobindo Pharma", "Pharma"),
    "BALKRISIND": ("BALKRISIND.NS", "Balkrishna Industries", "Auto"),
    "BANKBARODA": ("BANKBARODA.NS", "Bank of Baroda", "Banking"),
    "BEL": ("BEL.NS", "Bharat Electronics", "Defence"),
    "BHARATFORG": ("BHARATFORG.NS", "Bharat Forge", "Auto"),
    "BHEL": ("BHEL.NS", "Bharat Heavy Electricals", "Capital Goods"),
    "BIOCON": ("BIOCON.NS", "Biocon", "Pharma"),
    "BOSCHLTD": ("BOSCHLTD.NS", "Bosch", "Auto"),
    "CANBK": ("CANBK.NS", "Canara Bank", "Banking"),
    "CHOLAFIN": ("CHOLAFIN.NS", "Cholamandalam Inv. & Finance", "Financial Services"),
    "COLPAL": ("COLPAL.NS", "Colgate-Palmolive India", "FMCG"),
    "CONCOR": ("CONCOR.NS", "Container Corporation", "Logistics"),
    "CUMMINSIND": ("CUMMINSIND.NS", "Cummins India", "Capital Goods"),
    "DABUR": ("DABUR.NS", "Dabur India", "FMCG"),
    "DLF": ("DLF.NS", "DLF", "Realty"),
    "GAIL": ("GAIL.NS", "GAIL India", "Oil & Gas"),
    "GODREJCP": ("GODREJCP.NS", "Godrej Consumer Products", "FMCG"),
    "GODREJPROP": ("GODREJPROP.NS", "Godrej Properties", "Realty"),
    "HAVELLS": ("HAVELLS.NS", "Havells India", "Consumer Durables"),
    "HINDPETRO": ("HINDPETRO.NS", "Hindustan Petroleum", "Oil & Gas"),
    "ICICIGI": ("ICICIGI.NS", "ICICI Lombard", "Insurance"),
    "ICICIPRULI": ("ICICIPRULI.NS", "ICICI Prudential Life", "Insurance"),
    "IDFCFIRSTB": ("IDFCFIRSTB.NS", "IDFC First Bank", "Banking"),
    "INDHOTEL": ("INDHOTEL.NS", "Indian Hotels", "Hospitality"),
    "INDIGO": ("INDIGO.NS", "InterGlobe Aviation", "Aviation"),
    "IOC": ("IOC.NS", "Indian Oil Corporation", "Oil & Gas"),
    "JINDALSTEL": ("JINDALSTEL.NS", "Jindal Steel & Power", "Metals"),
    "JUBLFOOD": ("JUBLFOOD.NS", "Jubilant FoodWorks", "Consumer Services"),
    "LICHSGFIN": ("LICHSGFIN.NS", "LIC Housing Finance", "Financial Services"),
    "LUPIN": ("LUPIN.NS", "Lupin", "Pharma"),
    "MARICO": ("MARICO.NS", "Marico", "FMCG"),
    "MFSL": ("MFSL.NS", "Max Financial Services", "Insurance"),
    "MOTHERSON": ("MOTHERSON.NS", "Samvardhana Motherson", "Auto"),
    "MRF": ("MRF.NS", "MRF", "Auto"),
    "MUTHOOTFIN": ("MUTHOOTFIN.NS", "Muthoot Finance", "Financial Services"),
    "OBEROIRLTY": ("OBEROIRLTY.NS", "Oberoi Realty", "Realty"),
    "PAGEIND": ("PAGEIND.NS", "Page Industries", "Textiles"),
    "PERSISTENT": ("PERSISTENT.NS", "Persistent Systems", "IT"),
    "PETRONET": ("PETRONET.NS", "Petronet LNG", "Oil & Gas"),
    "PFC": ("PFC.NS", "Power Finance Corporation", "Financial Services"),
    "PIIND": ("PIIND.NS", "PI Industries", "Chemicals"),
    "PNB": ("PNB.NS", "Punjab National Bank", "Banking"),
    "POLYCAB": ("POLYCAB.NS", "Polycab India", "Consumer Durables"),
    "RECLTD": ("RECLTD.NS", "REC", "Financial Services"),
    "SIEMENS": ("SIEMENS.NS", "Siemens", "Capital Goods"),
    "SRF": ("SRF.NS", "SRF", "Chemicals"),
    "TATAPOWER": ("TATAPOWER.NS", "Tata Power", "Power"),
    "TORNTPHARM": ("TORNTPHARM.NS", "Torrent Pharmaceuticals", "Pharma"),
    "TRENT": ("TRENT.NS", "Trent", "Retail"),
    "TVSMOTOR": ("TVSMOTOR.NS", "TVS Motor Company", "Auto"),
    "UNIONBANK": ("UNIONBANK.NS", "Union Bank of India", "Banking"),
    "ZYDUSLIFE": ("ZYDUSLIFE.NS", "Zydus Lifesciences", "Pharma"),
}


def get_midcap_yahoo_tickers() -> list[str]:
    return [info[0] for info in MIDCAP_STOCKS.values()]


def get_midcap_symbols() -> list[str]:
    return list(MIDCAP_STOCKS.keys())
