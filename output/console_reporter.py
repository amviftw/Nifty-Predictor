"""
Rich-formatted console output for daily predictions.
"""

from datetime import date, datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box


console = Console()


def print_daily_report(
    signals: list[dict],
    macro_context: dict = None,
    yesterday_perf: dict = None,
    model_info: dict = None,
):
    """Print the full daily prediction report to console."""

    today = date.today()
    day_name = today.strftime("%A")
    now = datetime.now().strftime("%H:%M:%S IST")

    # Header
    console.print()
    console.rule(f"[bold cyan]NIFTY 50 DIRECTIONAL PREDICTION - {today} ({day_name})")
    console.print(
        f"  Generated at: {now}",
        style="dim",
    )
    if model_info:
        console.print(
            f"  Model version: {model_info.get('version', 'v1.0')} | "
            f"Retrained: {model_info.get('retrained', 'N/A')}",
            style="dim",
        )
    console.print()

    # Market context
    if macro_context:
        _print_market_context(macro_context)

    # Signal table
    _print_signal_table(signals)

    # Portfolio summary
    _print_portfolio_summary(signals)

    # Yesterday's performance
    if yesterday_perf:
        _print_yesterday_performance(yesterday_perf)

    console.rule(style="cyan")
    console.print()


def _print_market_context(ctx: dict):
    """Print market context panel."""
    lines = []

    sp500 = ctx.get("sp500_overnight_ret") or 0
    nasdaq = ctx.get("nasdaq_overnight_ret") or 0
    vix = ctx.get("india_vix") or 0
    vix_chg = ctx.get("india_vix_change") or 0
    fii = ctx.get("fii_net_buy") or 0
    dii = ctx.get("dii_net_buy") or 0
    usdinr = ctx.get("usdinr_level") or 0

    lines.append(
        f"S&P 500: {_fmt_pct(sp500)} | NASDAQ: {_fmt_pct(nasdaq)} | "
        f"India VIX: {vix:.1f} ({_fmt_pct(vix_chg)})"
    )
    lines.append(
        f"FII Net: {_fmt_crores(fii)} | DII Net: {_fmt_crores(dii)} | "
        f"USD/INR: {usdinr:.2f}"
    )

    panel = Panel(
        "\n".join(lines),
        title="Market Context",
        border_style="blue",
        padding=(0, 2),
    )
    console.print(panel)
    console.print()


def _print_signal_table(signals: list[dict]):
    """Print the main signal table."""
    table = Table(
        title="Trading Signals",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold white on dark_blue",
    )

    table.add_column("Stock", style="bold", width=14)
    table.add_column("Signal", justify="center", width=8)
    table.add_column("Conf%", justify="right", width=8)
    table.add_column("P(Up)", justify="right", width=8)
    table.add_column("P(Down)", justify="right", width=8)
    table.add_column("Strength", justify="center", width=10)
    table.add_column("Size%", justify="right", width=8)
    table.add_column("Sector", width=16)

    for sig in signals:
        signal = sig["signal"]
        conf = sig.get("confidence", 0)
        size = sig.get("position_size_pct", 0)

        # Color coding
        if signal == "BUY":
            signal_style = "bold green"
        elif signal == "SELL":
            signal_style = "bold red"
        else:
            signal_style = "dim"

        # Only show detailed info for actionable signals
        if signal == "HOLD" and size == 0:
            row_style = "dim"
        else:
            row_style = ""

        table.add_row(
            sig.get("symbol", ""),
            Text(signal, style=signal_style),
            f"{conf * 100:.1f}%",
            f"{sig.get('prob_up', 0):.3f}",
            f"{sig.get('prob_down', 0):.3f}",
            sig.get("strength", ""),
            f"{size * 100:.1f}%" if size > 0 else "-",
            sig.get("sector", ""),
            style=row_style,
        )

    console.print(table)
    console.print()

    # Print investment rationale for actionable signals
    actionable_with_reasons = [s for s in signals if s["signal"] != "HOLD" and s.get("reasons")]
    if actionable_with_reasons:
        console.rule("[bold dim]Investment Rationale[/bold dim]", style="dim")
        for sig in actionable_with_reasons:
            signal = sig["signal"]
            color = "green" if signal == "BUY" else "red"
            header = (
                f"[bold {color}]{sig['symbol']}[/bold {color}]"
                f"  [{color}]{signal}[/{color}]"
                f"  [dim]{sig.get('sector', '')}  |  {sig.get('confidence', 0) * 100:.0f}% confidence[/dim]"
            )
            console.print(header)
            layer_icons = ["[dim cyan]◆[/dim cyan]", "[dim yellow]◆[/dim yellow]",
                           "[dim blue]◆[/dim blue]", "[dim magenta]◆[/dim magenta]", "[dim white]◆[/dim white]"]
            for i, reason in enumerate(sig["reasons"]):
                icon = layer_icons[min(i, len(layer_icons) - 1)]
                console.print(f"  {icon} [dim]{reason}[/dim]")
            console.print()
        console.print()


def _print_portfolio_summary(signals: list[dict]):
    """Print portfolio allocation summary."""
    actionable = [s for s in signals if s["signal"] != "HOLD" and s.get("position_size_pct", 0) > 0]
    buy_count = sum(1 for s in actionable if s["signal"] == "BUY")
    sell_count = sum(1 for s in actionable if s["signal"] == "SELL")

    total_long = sum(s.get("position_size_pct", 0) for s in actionable if s["signal"] == "BUY")
    total_short = sum(s.get("position_size_pct", 0) for s in actionable if s["signal"] == "SELL")
    total_exposure = total_long + total_short

    # Sector breakdown
    sectors = {}
    for s in actionable:
        sec = s.get("sector", "Unknown")
        sectors[sec] = sectors.get(sec, 0) + s.get("position_size_pct", 0)

    sector_str = ", ".join(f"{k} {v * 100:.1f}%" for k, v in sorted(sectors.items(), key=lambda x: -x[1]))

    summary = (
        f"Active Signals: {len(actionable)} ({buy_count} BUY, {sell_count} SELL) | "
        f"Total Exposure: {total_exposure * 100:.1f}%\n"
        f"Sector: {sector_str if sector_str else 'None'}"
    )

    panel = Panel(summary, title="Portfolio Summary", border_style="green", padding=(0, 2))
    console.print(panel)
    console.print()


def _print_yesterday_performance(perf: dict):
    """Print yesterday's signal performance."""
    if not perf:
        return

    lines = []
    if "overall_signal_accuracy" in perf:
        acc = perf["overall_signal_accuracy"]
        total = perf.get("actionable_signals", 0)
        lines.append(f"Signal Accuracy: {acc * 100:.1f}% ({total} signals)")

    if "buy_avg_return" in perf:
        lines.append(
            f"BUY signals: avg return {perf['buy_avg_return'] * 100:.2f}%, "
            f"win rate {perf.get('buy_win_rate', 0) * 100:.0f}%"
        )

    if "sell_avg_return" in perf:
        lines.append(
            f"SELL signals: avg return {perf['sell_avg_return'] * 100:.2f}%, "
            f"win rate {perf.get('sell_win_rate', 0) * 100:.0f}%"
        )

    if lines:
        panel = Panel(
            "\n".join(lines),
            title="Recent Performance",
            border_style="yellow",
            padding=(0, 2),
        )
        console.print(panel)


def _fmt_pct(val: float) -> str:
    """Format a decimal as percentage with color hint."""
    if val is None or val == 0:
        return "0.00%"
    return f"{val * 100:+.2f}%"


def _fmt_crores(val: float) -> str:
    """Format FII/DII value in crores."""
    if val is None or val == 0:
        return "0 Cr"
    if abs(val) >= 1e7:
        return f"{val / 1e7:+,.0f} Cr"
    elif abs(val) >= 1e5:
        return f"{val / 1e5:+,.0f} L"
    return f"{val:+,.0f}"
