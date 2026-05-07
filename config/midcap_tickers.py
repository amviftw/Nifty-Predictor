"""
Nifty Midcap 150 representative subset for the dashboard analytics universe.

Names are selected from the Nifty Midcap 150 to give broad sector coverage
with high analyst coverage and liquidity. Some Nifty Next 50 names that
historically lived here (BANKBARODA, BEL, DLF, etc.) have moved to
`largecap_extra_tickers.py` to keep the universes clean — once a name is
clearly largecap, it should be tracked there.

Update after index rebalancing (March / September).
Last updated: May 2026.
"""

MIDCAP_STOCKS = {
    # Symbol: (Yahoo ticker, Company Name, Sector)
    "ABCAPITAL": ("ABCAPITAL.NS", "Aditya Birla Capital", "Financial Services"),
    "ABFRL": ("ABFRL.NS", "Aditya Birla Fashion & Retail", "Retail"),
    "ACC": ("ACC.NS", "ACC", "Cement"),
    "ALKEM": ("ALKEM.NS", "Alkem Laboratories", "Pharma"),
    "APLAPOLLO": ("APLAPOLLO.NS", "APL Apollo Tubes", "Metals"),
    "ASHOKLEY": ("ASHOKLEY.NS", "Ashok Leyland", "Auto"),
    "ASTRAL": ("ASTRAL.NS", "Astral", "Consumer Durables"),
    "AUBANK": ("AUBANK.NS", "AU Small Finance Bank", "Banking"),
    "AUROPHARMA": ("AUROPHARMA.NS", "Aurobindo Pharma", "Pharma"),
    "BALKRISIND": ("BALKRISIND.NS", "Balkrishna Industries", "Auto"),
    "BERGEPAINT": ("BERGEPAINT.NS", "Berger Paints", "Consumer Goods"),
    "BHARATFORG": ("BHARATFORG.NS", "Bharat Forge", "Auto"),
    "BHEL": ("BHEL.NS", "Bharat Heavy Electricals", "Capital Goods"),
    "BIOCON": ("BIOCON.NS", "Biocon", "Pharma"),
    "BSE": ("BSE.NS", "BSE", "Financial Services"),
    "CAMS": ("CAMS.NS", "Computer Age Management Services", "Financial Services"),
    "CANBK": ("CANBK.NS", "Canara Bank", "Banking"),
    "CGCL": ("CGCL.NS", "Capri Global Capital", "Financial Services"),
    "COFORGE": ("COFORGE.NS", "Coforge", "IT"),
    "COLPAL": ("COLPAL.NS", "Colgate-Palmolive India", "FMCG"),
    "CONCOR": ("CONCOR.NS", "Container Corporation", "Logistics"),
    "COROMANDEL": ("COROMANDEL.NS", "Coromandel International", "Chemicals"),
    "CUMMINSIND": ("CUMMINSIND.NS", "Cummins India", "Capital Goods"),
    "DEEPAKNTR": ("DEEPAKNTR.NS", "Deepak Nitrite", "Chemicals"),
    "DIXON": ("DIXON.NS", "Dixon Technologies", "Consumer Durables"),
    "ESCORTS": ("ESCORTS.NS", "Escorts Kubota", "Capital Goods"),
    "EXIDEIND": ("EXIDEIND.NS", "Exide Industries", "Auto"),
    "FEDERALBNK": ("FEDERALBNK.NS", "Federal Bank", "Banking"),
    "GLAND": ("GLAND.NS", "Gland Pharma", "Pharma"),
    "GODREJPROP": ("GODREJPROP.NS", "Godrej Properties", "Realty"),
    "HDFCAMC": ("HDFCAMC.NS", "HDFC Asset Management", "Financial Services"),
    "HINDPETRO": ("HINDPETRO.NS", "Hindustan Petroleum", "Oil & Gas"),
    "HONAUT": ("HONAUT.NS", "Honeywell Automation India", "Capital Goods"),
    "IDEA": ("IDEA.NS", "Vodafone Idea", "Telecom"),
    "IDFCFIRSTB": ("IDFCFIRSTB.NS", "IDFC First Bank", "Banking"),
    "IGL": ("IGL.NS", "Indraprastha Gas", "Oil & Gas"),
    "INDIANB": ("INDIANB.NS", "Indian Bank", "Banking"),
    "IOB": ("IOB.NS", "Indian Overseas Bank", "Banking"),
    "IPCALAB": ("IPCALAB.NS", "IPCA Laboratories", "Pharma"),
    "JSL": ("JSL.NS", "Jindal Stainless", "Metals"),
    "JUBLFOOD": ("JUBLFOOD.NS", "Jubilant FoodWorks", "Consumer Services"),
    "KPITTECH": ("KPITTECH.NS", "KPIT Technologies", "IT"),
    "LICHSGFIN": ("LICHSGFIN.NS", "LIC Housing Finance", "Financial Services"),
    "LUPIN": ("LUPIN.NS", "Lupin", "Pharma"),
    "LTF": ("LTF.NS", "L&T Finance", "Financial Services"),
    "LTTS": ("LTTS.NS", "L&T Technology Services", "IT"),
    "M&MFIN": ("M&MFIN.NS", "Mahindra & Mahindra Financial", "Financial Services"),
    "MFSL": ("MFSL.NS", "Max Financial Services", "Insurance"),
    "MGL": ("MGL.NS", "Mahanagar Gas", "Oil & Gas"),
    "MOTILALOFS": ("MOTILALOFS.NS", "Motilal Oswal Financial Services", "Financial Services"),
    "MPHASIS": ("MPHASIS.NS", "Mphasis", "IT"),
    "MRF": ("MRF.NS", "MRF", "Auto"),
    "MUTHOOTFIN": ("MUTHOOTFIN.NS", "Muthoot Finance", "Financial Services"),
    "NAM-INDIA": ("NAM-INDIA.NS", "Nippon Life India AMC", "Financial Services"),
    "NYKAA": ("NYKAA.NS", "FSN E-Commerce (Nykaa)", "Consumer Services"),
    "OBEROIRLTY": ("OBEROIRLTY.NS", "Oberoi Realty", "Realty"),
    "OFSS": ("OFSS.NS", "Oracle Financial Services", "IT"),
    "PAGEIND": ("PAGEIND.NS", "Page Industries", "Textiles"),
    "PATANJALI": ("PATANJALI.NS", "Patanjali Foods", "FMCG"),
    "PAYTM": ("PAYTM.NS", "One 97 Communications (Paytm)", "Financial Services"),
    "PEL": ("PEL.NS", "Piramal Enterprises", "Financial Services"),
    "PERSISTENT": ("PERSISTENT.NS", "Persistent Systems", "IT"),
    "PETRONET": ("PETRONET.NS", "Petronet LNG", "Oil & Gas"),
    "PHOENIXLTD": ("PHOENIXLTD.NS", "Phoenix Mills", "Realty"),
    "PIIND": ("PIIND.NS", "PI Industries", "Chemicals"),
    "PNB": ("PNB.NS", "Punjab National Bank", "Banking"),
    "POLICYBZR": ("POLICYBZR.NS", "PB Fintech (PolicyBazaar)", "Financial Services"),
    "POLYCAB": ("POLYCAB.NS", "Polycab India", "Consumer Durables"),
    "PRESTIGE": ("PRESTIGE.NS", "Prestige Estates", "Realty"),
    "RVNL": ("RVNL.NS", "Rail Vikas Nigam", "Infrastructure"),
    "SAIL": ("SAIL.NS", "Steel Authority of India", "Metals"),
    "SOLARINDS": ("SOLARINDS.NS", "Solar Industries India", "Chemicals"),
    "SONACOMS": ("SONACOMS.NS", "Sona BLW Precision Forgings", "Auto"),
    "SRF": ("SRF.NS", "SRF", "Chemicals"),
    "SUNDARMFIN": ("SUNDARMFIN.NS", "Sundaram Finance", "Financial Services"),
    "SUPREMEIND": ("SUPREMEIND.NS", "Supreme Industries", "Consumer Durables"),
    "SUZLON": ("SUZLON.NS", "Suzlon Energy", "Power"),
    "SYNGENE": ("SYNGENE.NS", "Syngene International", "Pharma"),
    "TATACHEM": ("TATACHEM.NS", "Tata Chemicals", "Chemicals"),
    "TATACOMM": ("TATACOMM.NS", "Tata Communications", "Telecom"),
    "TATAELXSI": ("TATAELXSI.NS", "Tata Elxsi", "IT"),
    "TIINDIA": ("TIINDIA.NS", "Tube Investments of India", "Auto"),
    "TORNTPHARM": ("TORNTPHARM.NS", "Torrent Pharmaceuticals", "Pharma"),
    "TORNTPOWER": ("TORNTPOWER.NS", "Torrent Power", "Power"),
    "UNIONBANK": ("UNIONBANK.NS", "Union Bank of India", "Banking"),
    "VOLTAS": ("VOLTAS.NS", "Voltas", "Consumer Durables"),
    "YESBANK": ("YESBANK.NS", "Yes Bank", "Banking"),
    "ZEEL": ("ZEEL.NS", "Zee Entertainment", "Media"),
}


def get_midcap_yahoo_tickers() -> list[str]:
    return [info[0] for info in MIDCAP_STOCKS.values()]


def get_midcap_symbols() -> list[str]:
    return list(MIDCAP_STOCKS.keys())
