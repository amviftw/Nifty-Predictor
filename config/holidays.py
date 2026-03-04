"""
NSE holiday calendar and F&O expiry date computation.
Update the holiday list annually from NSE circulars.
"""

from datetime import date, timedelta
import calendar


# NSE trading holidays for 2026 (update annually from NSE circular)
NSE_HOLIDAYS_2026 = {
    date(2026, 1, 26),   # Republic Day
    date(2026, 2, 17),   # Maha Shivaratri
    date(2026, 3, 10),   # Holi
    date(2026, 3, 30),   # Id-Ul-Fitr (Eid)
    date(2026, 4, 2),    # Ram Navami
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 5, 25),   # Buddha Purnima
    date(2026, 6, 5),    # Bakrid (Eid-ul-Adha)
    date(2026, 7, 6),    # Muharram
    date(2026, 8, 15),   # Independence Day
    date(2026, 8, 25),   # Janmashtami
    date(2026, 9, 4),    # Milad-Un-Nabi
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra
    date(2026, 11, 9),   # Diwali (Laxmi Pujan)
    date(2026, 11, 10),  # Diwali (Balipratipada)
    date(2026, 11, 24),  # Guru Nanak Jayanti
    date(2026, 12, 25),  # Christmas
}


def is_nse_holiday(d: date) -> bool:
    """Check if a given date is an NSE holiday or weekend."""
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return True
    return d in NSE_HOLIDAYS_2026


def is_trading_day(d: date) -> bool:
    """Check if a given date is a trading day."""
    return not is_nse_holiday(d)


def next_trading_day(d: date) -> date:
    """Return the next trading day after the given date."""
    d = d + timedelta(days=1)
    while is_nse_holiday(d):
        d += timedelta(days=1)
    return d


def prev_trading_day(d: date) -> date:
    """Return the previous trading day before the given date."""
    d = d - timedelta(days=1)
    while is_nse_holiday(d):
        d -= timedelta(days=1)
    return d


def last_tuesday_of_month(year: int, month: int) -> date:
    """Return the last Tuesday of the given month."""
    # Find the last day of the month
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)

    # Walk backward to find Tuesday (weekday=1)
    while d.weekday() != 1:  # 1 = Tuesday
        d -= timedelta(days=1)
    return d


def next_fno_expiry(current_date: date) -> date:
    """
    Return the next monthly F&O expiry date.
    NSE monthly expiry is the last Tuesday of the month.
    If that Tuesday is a holiday, expiry shifts to the previous trading day.
    """
    year, month = current_date.year, current_date.month

    expiry = last_tuesday_of_month(year, month)

    # If expiry is a holiday, shift to previous trading day
    while is_nse_holiday(expiry):
        expiry -= timedelta(days=1)

    # If current date is past this month's expiry, get next month's
    if current_date > expiry:
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
        expiry = last_tuesday_of_month(year, month)
        while is_nse_holiday(expiry):
            expiry -= timedelta(days=1)

    return expiry


def days_to_expiry(current_date: date) -> int:
    """Return trading days until next F&O monthly expiry."""
    expiry = next_fno_expiry(current_date)
    count = 0
    d = current_date
    while d < expiry:
        d += timedelta(days=1)
        if is_trading_day(d):
            count += 1
    return count
