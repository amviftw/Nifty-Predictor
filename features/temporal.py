"""
Calendar and temporal feature computation.
Day-of-week, month, F&O expiry proximity, earnings season, etc.
"""

from datetime import date

from config.holidays import days_to_expiry, next_fno_expiry


def compute_temporal_features(target_date: date) -> dict:
    """
    Compute temporal/calendar features for a given date.

    Input: date object
    Output: dict with 10 temporal feature values
    """
    dte = days_to_expiry(target_date)
    fno_expiry = next_fno_expiry(target_date)

    return {
        # Day of week (Monday=0, Friday=4)
        "day_of_week": target_date.weekday(),
        # Month (1-12)
        "month_of_year": target_date.month,
        # Month start/end flags
        "is_month_start": int(target_date.day <= 3),
        "is_month_end": int(target_date.day >= 27),
        # Quarter end
        "is_quarter_end": int(
            target_date.month in (3, 6, 9, 12) and target_date.day >= 25
        ),
        # F&O expiry proximity
        "days_to_fno_expiry": dte,
        "is_expiry_week": int(dte <= 5),
        "is_expiry_day": int(target_date == fno_expiry),
        # Earnings season (approximate windows)
        "is_earnings_season": int(_is_earnings_season(target_date)),
        # Days since start of month (proxy for settlement patterns)
        "day_of_month": target_date.day,
    }


def _is_earnings_season(d: date) -> bool:
    """
    Check if date falls within typical Indian earnings announcement windows.
    Q3: Jan 15 - Feb 15
    Q4: Apr 15 - May 15
    Q1: Jul 15 - Aug 15
    Q2: Oct 15 - Nov 15
    """
    month, day = d.month, d.day

    if month == 1 and day >= 15:
        return True
    if month == 2 and day <= 15:
        return True
    if month == 4 and day >= 15:
        return True
    if month == 5 and day <= 15:
        return True
    if month == 7 and day >= 15:
        return True
    if month == 8 and day <= 15:
        return True
    if month == 10 and day >= 15:
        return True
    if month == 11 and day <= 15:
        return True

    return False


def get_temporal_feature_names() -> list[str]:
    """Return list of temporal feature column names."""
    return [
        "day_of_week", "month_of_year", "is_month_start", "is_month_end",
        "is_quarter_end", "days_to_fno_expiry", "is_expiry_week",
        "is_expiry_day", "is_earnings_season", "day_of_month",
    ]
