"""
Driver Registry — the vocabulary of the market.

A "driver" is a macro/cross-asset variable that moves Indian equities.
Each driver carries Varsity-style pedagogy (what / why / who / example)
so the dashboard can teach as much as it predicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Driver:
    """A single macro/cross-asset factor."""
    id: str                  # Stable identifier, used in code.
    name: str                # Display name.
    ticker: str              # Yahoo Finance ticker.
    category: str            # Commodity | Currency | Yield | Volatility | Equity Overnight | Flow
    unit: str                # Unit of the shock input (e.g. "%" for returns, "bps" for yields).
    default_shock: float     # Default slider value, in the shown unit.
    shock_range: tuple[float, float]  # (min, max) for the slider.
    # Varsity-style pedagogy
    what: str                # What this driver is, in one sentence.
    why: str                 # Why it moves Indian equities.
    who_positive: list[str] = field(default_factory=list)   # Sectors helped when driver rises
    who_negative: list[str] = field(default_factory=list)   # Sectors hurt when driver rises
    example: str = ""        # Concrete Varsity-style example.


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

DRIVERS: dict[str, Driver] = {
    "crude": Driver(
        id="crude",
        name="Crude Oil (Brent)",
        ticker="BZ=F",
        category="Commodity",
        unit="%",
        default_shock=2.0,
        shock_range=(-10.0, 10.0),
        what="The price of Brent crude oil, the global benchmark for oil imports.",
        why=(
            "India imports ~85% of its crude. A rise in crude widens the import bill, "
            "pressures INR, lifts inflation, and squeezes margins in oil-dependent sectors. "
            "Upstream producers (ONGC) benefit; everyone else pays more."
        ),
        who_positive=["Oil & Gas"],
        who_negative=["Auto", "FMCG", "Infrastructure", "Cement", "Power", "Consumer Goods"],
        example=(
            "When Brent jumped from $75 to $90 over Aug–Sep 2023, ONGC gained ~18% "
            "while Asian Paints (a crude-derivative user) fell ~7% in the same window."
        ),
    ),
    "usdinr": Driver(
        id="usdinr",
        name="USD / INR",
        ticker="INR=X",
        category="Currency",
        unit="%",
        default_shock=0.5,
        shock_range=(-3.0, 3.0),
        what="The rupee's price against the US dollar. Higher = weaker rupee.",
        why=(
            "A weaker rupee boosts INR revenue for exporters (IT, Pharma) who bill "
            "in dollars, but raises import costs for everyone else. It often signals "
            "FII outflow pressure, which weighs on broad markets."
        ),
        who_positive=["IT", "Pharma"],
        who_negative=["Auto", "Oil & Gas", "Consumer Goods"],
        example=(
            "USD/INR rose from 82 to 84 in Oct 2023; Nifty IT outperformed Nifty 50 "
            "by ~4% over the same month as the street priced in stronger export earnings."
        ),
    ),
    "us10y": Driver(
        id="us10y",
        name="US 10-Year Yield",
        ticker="^TNX",
        category="Yield",
        unit="%",
        default_shock=0.1,
        shock_range=(-0.5, 0.5),
        what="Yield on the 10-year US Treasury. The world's risk-free benchmark.",
        why=(
            "Rising US yields make US bonds more attractive than emerging-market equities. "
            "FIIs pull money out of India, hurting rate-sensitive sectors (banks, financials, "
            "real estate) and high-P/E tech names."
        ),
        who_positive=[],
        who_negative=["Banking", "Financial Services", "Realty", "IT"],
        example=(
            "When US 10Y climbed from 3.8% to 5.0% in Aug–Oct 2023, FIIs sold ~₹40,000 cr "
            "of Indian equities and Bank Nifty lost ~7% in eight weeks."
        ),
    ),
    "vix": Driver(
        id="vix",
        name="India VIX",
        ticker="^INDIAVIX",
        category="Volatility",
        unit="%",
        default_shock=10.0,
        shock_range=(-30.0, 60.0),
        what="The expected 30-day volatility of Nifty, priced from option premia.",
        why=(
            "VIX rises when traders buy protection — i.e., when fear is building. "
            "A spike in VIX usually coincides with Nifty falling, and high-beta names "
            "(midcaps, PSU banks, metals) fall more than the index."
        ),
        who_positive=[],
        who_negative=["Banking", "Metals", "Auto", "Realty"],
        example=(
            "On 4 Jun 2024 (election result day), India VIX spiked 40% and Nifty fell 5.9%; "
            "PSU banks lost ~15% intraday before recovering."
        ),
    ),
    "sp500": Driver(
        id="sp500",
        name="S&P 500 (overnight)",
        ticker="^GSPC",
        category="Equity Overnight",
        unit="%",
        default_shock=1.0,
        shock_range=(-5.0, 5.0),
        what="The prior-session close-to-close move of the US S&P 500 index.",
        why=(
            "US equities set the overnight risk tone. A strong US close typically lifts "
            "SGX Nifty and Asian markets. The effect is largest on sectors with US-linked "
            "earnings (IT, Pharma) and on broad index names."
        ),
        who_positive=["IT", "Pharma", "Banking", "Financial Services"],
        who_negative=[],
        example=(
            "S&P rose 2.1% on 10 Nov 2022 (CPI print); Nifty gapped up 1.3% the next morning "
            "with IT leading (+2.8%) on renewed global risk appetite."
        ),
    ),
    "dxy": Driver(
        id="dxy",
        name="US Dollar Index (DXY)",
        ticker="DX-Y.NYB",
        category="Currency",
        unit="%",
        default_shock=0.5,
        shock_range=(-3.0, 3.0),
        what="The dollar's value against a basket of developed-market currencies.",
        why=(
            "A strong DXY generally means risk-off for emerging markets: global liquidity "
            "tightens, commodity prices fall, and FIIs reduce EM exposure. Indian IT and "
            "Pharma are a partial hedge — they earn in dollars."
        ),
        who_positive=["IT", "Pharma"],
        who_negative=["Metals", "Banking", "Financial Services"],
        example=(
            "DXY rose from 100 to 107 in Sep–Oct 2023; Nifty Metal fell ~9% while "
            "Nifty IT gained ~3% over the same period."
        ),
    ),
    "copper": Driver(
        id="copper",
        name="Copper",
        ticker="HG=F",
        category="Commodity",
        unit="%",
        default_shock=2.0,
        shock_range=(-10.0, 10.0),
        what="The price of copper futures — a proxy for global industrial demand.",
        why=(
            "Copper is called 'Dr. Copper' because it correlates with global growth. "
            "Rising copper lifts metals producers directly and signals improving "
            "demand for autos, infrastructure, and capex-linked names."
        ),
        who_positive=["Metals", "Mining", "Infrastructure", "Auto"],
        who_negative=[],
        example=(
            "Copper rallied 12% in Mar–Apr 2024 on China stimulus hopes; "
            "Hindalco gained 22% and JSW Steel gained 18% in the same window."
        ),
    ),
    "fii_flow": Driver(
        id="fii_flow",
        name="FII Net Flow (₹ cr)",
        ticker="",  # Not from yfinance; user-simulated only in MVP.
        category="Flow",
        unit="₹ cr",
        default_shock=-2000.0,
        shock_range=(-10000.0, 10000.0),
        what="Net daily purchases (positive) or sales (negative) by Foreign Institutional Investors.",
        why=(
            "FIIs are the marginal buyer in Indian equities — roughly 20% of free-float "
            "ownership. Heavy FII selling drains large-cap liquidity, disproportionately "
            "hurting index-heavy banks and financials. DII buying can partially offset."
        ),
        who_positive=["Banking", "Financial Services", "IT"],
        who_negative=[],
        example=(
            "FIIs sold ₹24,000 cr over 15 Oct – 5 Nov 2024; HDFC Bank and ICICI Bank, "
            "together ~20% of FII India exposure, lost 6% and 4% respectively."
        ),
    ),
}


def get_driver(driver_id: str) -> Driver:
    """Lookup a driver by id, raises KeyError if unknown."""
    return DRIVERS[driver_id]


def driver_ids() -> list[str]:
    """Stable order of driver ids for UI enumeration."""
    return list(DRIVERS.keys())


def price_based_drivers() -> list[str]:
    """Subset of drivers whose historical moves can be read from yfinance."""
    return [d.id for d in DRIVERS.values() if d.ticker]
