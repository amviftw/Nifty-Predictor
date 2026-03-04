"""
Email notification module.
Sends daily signals, training reports, and evaluation reports via Gmail SMTP.
Uses only Python built-in libraries (smtplib, email).
"""

import smtplib
from datetime import date, datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from loguru import logger

from config.settings import SETTINGS


def _send_email(subject: str, html_body: str) -> bool:
    """
    Send an HTML email via Gmail SMTP.
    Returns True on success, False on failure.
    """
    if not SETTINGS.EMAIL_SENDER or not SETTINGS.EMAIL_PASSWORD or not SETTINGS.EMAIL_RECIPIENT:
        logger.error(
            "Email not configured. Fill in EMAIL_SENDER, EMAIL_PASSWORD, "
            "EMAIL_RECIPIENT in .env file."
        )
        return False

    recipients = [r.strip() for r in SETTINGS.EMAIL_RECIPIENT.split(",") if r.strip()]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"Nifty Predictor <{SETTINGS.EMAIL_SENDER}>"
    msg["To"] = "undisclosed-recipients:;"

    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(SETTINGS.SMTP_SERVER, SETTINGS.SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SETTINGS.EMAIL_SENDER, SETTINGS.EMAIL_PASSWORD)
            server.sendmail(
                SETTINGS.EMAIL_SENDER,
                recipients,
                msg.as_string(),
            )
        logger.info(f"Email sent: {subject}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False


# ─────────────────────── HTML STYLES ───────────────────────

_STYLES = """
<style>
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6f9; margin: 0; padding: 20px; }
    .container { max-width: 700px; margin: 0 auto; background: #fff; border-radius: 8px;
                 box-shadow: 0 2px 8px rgba(0,0,0,0.08); overflow: hidden; }
    .header { background: linear-gradient(135deg, #1a237e, #283593); color: #fff;
              padding: 24px 28px; }
    .header h1 { margin: 0; font-size: 22px; font-weight: 600; }
    .header p { margin: 6px 0 0; font-size: 13px; opacity: 0.85; }
    .section { padding: 20px 28px; }
    .section h2 { font-size: 16px; color: #333; margin: 0 0 12px; border-bottom: 2px solid #e0e0e0;
                  padding-bottom: 6px; }
    .context-box { background: #f8f9fa; border-left: 4px solid #1a237e; padding: 12px 16px;
                   margin-bottom: 16px; font-size: 13px; color: #555; border-radius: 0 4px 4px 0; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th { background: #1a237e; color: #fff; padding: 10px 12px; text-align: left; font-weight: 600; }
    td { padding: 9px 12px; border-bottom: 1px solid #eee; }
    tr:nth-child(even) { background: #fafafa; }
    .buy { color: #2e7d32; font-weight: 700; }
    .sell { color: #c62828; font-weight: 700; }
    .hold { color: #9e9e9e; }
    .strong { background: #e8f5e9; border-radius: 4px; padding: 2px 8px; font-size: 11px; }
    .moderate { background: #fff3e0; border-radius: 4px; padding: 2px 8px; font-size: 11px; }
    .summary-box { background: #e3f2fd; border-radius: 6px; padding: 14px 18px; margin-top: 12px;
                   font-size: 13px; }
    .footer { text-align: center; padding: 16px; font-size: 11px; color: #999;
              border-top: 1px solid #eee; }
    .metric-grid { display: flex; flex-wrap: wrap; gap: 12px; }
    .metric-card { background: #f8f9fa; border-radius: 6px; padding: 12px 16px; flex: 1;
                   min-width: 140px; text-align: center; }
    .metric-card .value { font-size: 24px; font-weight: 700; color: #1a237e; }
    .metric-card .label { font-size: 11px; color: #777; margin-top: 4px; }
    .pos { color: #2e7d32; }
    .neg { color: #c62828; }
</style>
"""


# ─────────────────── DAILY SIGNALS EMAIL ───────────────────

def send_daily_signals_email(
    signals: list[dict],
    macro_context: dict = None,
    yesterday_perf: dict = None,
) -> bool:
    """Send the daily prediction signals email."""
    today = date.today()
    day_name = today.strftime("%A")

    actionable = [s for s in signals if s["signal"] != "HOLD" and s.get("position_size_pct", 0) > 0]
    buy_count = sum(1 for s in actionable if s["signal"] == "BUY")
    sell_count = sum(1 for s in actionable if s["signal"] == "SELL")

    subject = (
        f"Nifty Signals {today} | "
        f"{buy_count} BUY, {sell_count} SELL"
    )

    # Build HTML
    html = f"""<!DOCTYPE html><html><head>{_STYLES}</head><body>
    <div class="container">
        <div class="header">
            <h1>Nifty 50 Trading Signals</h1>
            <p>{today} ({day_name}) &bull; Generated at {datetime.now().strftime('%H:%M IST')}</p>
        </div>
    """

    # Market context
    if macro_context:
        sp500 = _fmt_pct(macro_context.get("sp500_ret"))
        nasdaq = _fmt_pct(macro_context.get("nasdaq_ret"))
        vix = macro_context.get("india_vix", 0) or 0
        fii = macro_context.get("fii_net_buy", 0) or 0
        dii = macro_context.get("dii_net_buy", 0) or 0
        usdinr = macro_context.get("usdinr", 0) or 0

        html += f"""
        <div class="section">
            <div class="context-box">
                S&amp;P 500: <b>{sp500}</b> &nbsp;|&nbsp; NASDAQ: <b>{nasdaq}</b> &nbsp;|&nbsp;
                India VIX: <b>{vix:.1f}</b><br>
                FII Net: <b>{_fmt_crores(fii)}</b> &nbsp;|&nbsp;
                DII Net: <b>{_fmt_crores(dii)}</b> &nbsp;|&nbsp;
                USD/INR: <b>{usdinr:.2f}</b>
            </div>
        </div>
        """

    # Actionable signals table
    if actionable:
        html += """
        <div class="section">
            <h2>Actionable Signals</h2>
            <table>
                <tr>
                    <th>Stock</th><th>Signal</th><th>Confidence</th>
                    <th>P(Up)</th><th>P(Down)</th><th>Strength</th>
                    <th>Size</th><th>Sector</th>
                </tr>
        """
        for sig in actionable:
            signal = sig["signal"]
            css_class = "buy" if signal == "BUY" else "sell"
            strength = sig.get("strength", "")
            str_class = "strong" if strength == "STRONG" else "moderate"
            size = sig.get("position_size_pct", 0)

            html += f"""
                <tr>
                    <td><b>{sig.get('symbol', '')}</b></td>
                    <td class="{css_class}">{signal}</td>
                    <td>{sig.get('confidence', 0) * 100:.1f}%</td>
                    <td>{sig.get('prob_up', 0):.3f}</td>
                    <td>{sig.get('prob_down', 0):.3f}</td>
                    <td><span class="{str_class}">{strength}</span></td>
                    <td>{size * 100:.1f}%</td>
                    <td>{sig.get('sector', '')}</td>
                </tr>
            """
        html += "</table>"

        # Portfolio summary
        total_long = sum(s.get("position_size_pct", 0) for s in actionable if s["signal"] == "BUY")
        total_short = sum(s.get("position_size_pct", 0) for s in actionable if s["signal"] == "SELL")
        html += f"""
            <div class="summary-box">
                <b>Portfolio:</b> {len(actionable)} positions
                ({buy_count} BUY, {sell_count} SELL) &nbsp;|&nbsp;
                Long: {total_long * 100:.1f}% &nbsp;|&nbsp;
                Short: {total_short * 100:.1f}% &nbsp;|&nbsp;
                Total Exposure: {(total_long + total_short) * 100:.1f}%
            </div>
        </div>
        """

        # Key drivers section — card layout (one card per stock)
        signals_with_reasons = [s for s in actionable if s.get("reasons")]
        if signals_with_reasons:
            html += """
        <div class="section">
            <h2>Investment Rationale</h2>
            """
            for sig in signals_with_reasons:
                signal = sig["signal"]
                border_color = "#2e7d32" if signal == "BUY" else "#c62828"
                label_bg = "#e8f5e9" if signal == "BUY" else "#ffebee"
                label_color = "#2e7d32" if signal == "BUY" else "#c62828"
                symbol = sig.get("symbol", "")
                sector = sig.get("sector", "")
                conf = sig.get("confidence", 0) * 100

                html += f"""
                <div style="border:1px solid #e0e0e0; border-left:4px solid {border_color};
                            border-radius:6px; padding:14px 18px; margin-bottom:12px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                        <div>
                            <b style="font-size:15px;">{symbol}</b>
                            <span style="font-size:12px; color:#777; margin-left:8px;">{sector}</span>
                        </div>
                        <span style="background:{label_bg}; color:{label_color}; font-weight:700;
                                     font-size:12px; padding:3px 10px; border-radius:4px;">{signal} &nbsp;{conf:.0f}%</span>
                    </div>
                """
                for i, reason in enumerate(sig["reasons"]):
                    icon = ["🏭", "📊", "📈", "🌐", "📰"][min(i, 4)]
                    html += f"""
                    <p style="margin:6px 0; font-size:13px; color:#444; line-height:1.5;">
                        {icon}&nbsp; {reason}
                    </p>
                    """
                html += "</div>"
            html += "</div>"
    else:
        html += """
        <div class="section">
            <div class="summary-box">No actionable signals today. All stocks rated HOLD.</div>
        </div>
        """

    # Yesterday's performance
    if yesterday_perf and "overall_signal_accuracy" in yesterday_perf:
        acc = yesterday_perf["overall_signal_accuracy"] * 100
        html += f"""
        <div class="section">
            <h2>Recent Performance</h2>
            <div class="context-box">
                Signal Accuracy: <b>{acc:.0f}%</b>
        """
        if "buy_avg_return" in yesterday_perf:
            ret = yesterday_perf["buy_avg_return"] * 100
            css = "pos" if ret > 0 else "neg"
            html += f' &nbsp;|&nbsp; BUY avg return: <b class="{css}">{ret:+.2f}%</b>'
        if "sell_avg_return" in yesterday_perf:
            ret = yesterday_perf["sell_avg_return"] * 100
            css = "pos" if ret < 0 else "neg"
            html += f' &nbsp;|&nbsp; SELL avg return: <b class="{css}">{ret:+.2f}%</b>'
        html += """
            </div>
        </div>
        """

    html += """
        <div class="footer">
            Nifty 50 Directional Predictor &bull; Automated Signal System
        </div>
    </div></body></html>
    """

    return _send_email(subject, html)


# ─────────────────── TRAINING REPORT EMAIL ───────────────────

def send_training_report_email(
    lgb_score: float,
    xgb_score: float,
    rf_score: float,
    weights: tuple,
    n_features: int,
    n_samples: int,
) -> bool:
    """Send model training completion report email."""
    today = date.today()

    subject = f"Nifty Model Retrained {today} | F1: {max(lgb_score, xgb_score, rf_score):.3f}"

    html = f"""<!DOCTYPE html><html><head>{_STYLES}</head><body>
    <div class="container">
        <div class="header">
            <h1>Model Training Report</h1>
            <p>{today} &bull; Weekly Retrain Complete</p>
        </div>
        <div class="section">
            <h2>Walk-Forward Validation Scores (F1-Macro)</h2>
            <table>
                <tr><th>Model</th><th>F1-Macro</th><th>Ensemble Weight</th></tr>
                <tr><td>LightGBM</td><td><b>{lgb_score:.4f}</b></td><td>{weights[0]:.1%}</td></tr>
                <tr><td>XGBoost</td><td><b>{xgb_score:.4f}</b></td><td>{weights[1]:.1%}</td></tr>
                <tr><td>Random Forest</td><td><b>{rf_score:.4f}</b></td><td>{weights[2]:.1%}</td></tr>
            </table>
            <div class="summary-box">
                Features: <b>{n_features}</b> &nbsp;|&nbsp;
                Training samples: <b>{n_samples:,}</b> &nbsp;|&nbsp;
                Random baseline: <b>0.333</b>
            </div>
        </div>
        <div class="footer">
            Nifty 50 Directional Predictor &bull; Automated Training Pipeline
        </div>
    </div></body></html>
    """

    return _send_email(subject, html)


# ─────────────────── EVALUATION REPORT EMAIL ───────────────────

def send_evaluation_report_email(perf: dict, days: int = 7) -> bool:
    """Send weekly signal evaluation report email."""
    today = date.today()
    acc = perf.get("overall_signal_accuracy", 0) * 100

    subject = f"Nifty Weekly Report {today} | Accuracy: {acc:.0f}%"

    html = f"""<!DOCTYPE html><html><head>{_STYLES}</head><body>
    <div class="container">
        <div class="header">
            <h1>Weekly Performance Report</h1>
            <p>{today} &bull; Last {days} trading days</p>
        </div>
        <div class="section">
            <h2>Signal Performance</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Predictions</td><td><b>{perf.get('total_signals', 0)}</b></td></tr>
                <tr><td>Actionable Signals</td><td><b>{perf.get('actionable_signals', 0)}</b></td></tr>
                <tr><td>Signal Accuracy</td><td><b>{acc:.1f}%</b></td></tr>
    """

    if "buy_count" in perf:
        html += f"""
                <tr><td>BUY Signals</td><td>{perf['buy_count']}</td></tr>
                <tr><td>BUY Avg Return</td>
                    <td class="{'pos' if perf['buy_avg_return'] > 0 else 'neg'}">
                    {perf['buy_avg_return'] * 100:+.2f}%</td></tr>
                <tr><td>BUY Win Rate</td><td>{perf['buy_win_rate'] * 100:.0f}%</td></tr>
        """

    if "sell_count" in perf:
        html += f"""
                <tr><td>SELL Signals</td><td>{perf['sell_count']}</td></tr>
                <tr><td>SELL Avg Return</td>
                    <td class="{'pos' if perf['sell_avg_return'] < 0 else 'neg'}">
                    {perf['sell_avg_return'] * 100:+.2f}%</td></tr>
                <tr><td>SELL Win Rate</td><td>{perf['sell_win_rate'] * 100:.0f}%</td></tr>
        """

    html += """
            </table>
        </div>
        <div class="footer">
            Nifty 50 Directional Predictor &bull; Weekly Evaluation
        </div>
    </div></body></html>
    """

    return _send_email(subject, html)


# ─────────────────── ERROR NOTIFICATION ───────────────────

def send_error_email(script_name: str, error_message: str) -> bool:
    """Send an error notification email when a script fails."""
    today = date.today()
    subject = f"NIFTY PREDICTOR ERROR - {script_name} failed on {today}"

    html = f"""<!DOCTYPE html><html><head>{_STYLES}</head><body>
    <div class="container">
        <div class="header" style="background: linear-gradient(135deg, #b71c1c, #c62828);">
            <h1>Script Failure Alert</h1>
            <p>{today} &bull; {script_name}</p>
        </div>
        <div class="section">
            <div class="context-box" style="border-left-color: #c62828;">
                <b>Error:</b><br>
                <pre style="white-space: pre-wrap; font-size: 12px; margin-top: 8px;">{error_message}</pre>
            </div>
            <p style="font-size: 13px; color: #555;">
                Check the log files in <code>storage/logs/</code> for full details.
            </p>
        </div>
        <div class="footer">
            Nifty 50 Directional Predictor &bull; Error Alert
        </div>
    </div></body></html>
    """

    return _send_email(subject, html)


# ─────────────────── HELPERS ───────────────────

def _fmt_pct(val) -> str:
    if val is None:
        return "N/A"
    return f"{float(val) * 100:+.2f}%"


def _fmt_crores(val) -> str:
    if val is None or val == 0:
        return "0 Cr"
    v = float(val)
    if abs(v) >= 1e7:
        return f"{v / 1e7:+,.0f} Cr"
    elif abs(v) >= 1e5:
        return f"{v / 1e5:+,.0f} L"
    return f"{v:+,.0f}"
