"""
backtest_to_tsv.py
------------------
Converts backtest monthly P&L output into a TSV table ready to
copy-paste into Google Sheets.

Optionally fetches S&P 500, Dow, and Nasdaq monthly returns via yfinance.
Install with:  pip install yfinance

Usage:
    python backtest_to_tsv.py                  # benchmarks left blank
    python backtest_to_tsv.py --benchmarks     # fetches benchmark data (requires yfinance)
    python backtest_to_tsv.py --out results.tsv
"""

import re
import sys
import argparse
from datetime import datetime

# ── 1. Paste your backtest monthly P&L block here ────────────────────────────
RAW_OUTPUT = """
    2016-06: +$101.59  (+) ##
    2016-07: +$13.57  (+)
    2016-08: $-266.17  (-) #####
    2016-09: $-1,001.65  (-) ####################
    2016-10: +$7.24  (+)
    2016-11: +$2,924.42  (+) ####################
    2016-12: +$689.08  (+) #############
    2017-01: $-1,089.09  (-) ####################
    2017-02: +$366.30  (+) #######
    2017-03: $-374.03  (-) #######
    2017-04: $-446.02  (-) ########
    2017-05: +$2,438.25  (+) ####################
    2017-06: $-139.09  (-) ##
    2017-07: +$137.04  (+) ##
    2017-08: +$2,257.02  (+) ####################
    2017-09: +$1,268.41  (+) ####################
    2017-10: +$315.79  (+) ######
    2017-11: +$1,705.84  (+) ####################
    2017-12: +$758.92  (+) ###############
    2018-01: $-1,501.92  (-) ####################
    2018-02: $-1,830.45  (-) ####################
    2018-03: +$520.49  (+) ##########
    2018-04: $-249.83  (-) ####
    2018-05: +$2,265.53  (+) ####################
    2018-06: $-169.72  (-) ###
    2018-07: +$813.19  (+) ################
    2018-08: +$7,485.78  (+) ####################
    2018-09: +$3,373.91  (+) ####################
    2018-10: $-36.99  (-) 
    2018-11: $-487.93  (-) #########
    2018-12: +$3.49  (+)
    2019-02: +$1,223.26  (+) ####################
    2019-03: +$1,173.86  (+) ####################
    2019-04: +$3,807.61  (+) ####################
    2019-05: $-1,036.94  (-) ####################
    2019-06: +$2,122.66  (+) ####################
    2019-07: +$3,986.23  (+) ####################
    2019-08: $-157.62  (-) ###
    2019-09: +$438.44  (+) ########
    2019-10: +$546.37  (+) ##########
    2019-11: +$778.02  (+) ###############
    2019-12: +$6,727.01  (+) ####################
    2020-01: +$1,754.40  (+) ####################
    2020-02: $-924.38  (-) ##################
    2020-06: $-3,704.63  (-) ####################
    2020-07: +$5,343.08  (+) ####################
    2020-08: +$3,832.52  (+) ####################
    2020-09: +$2,786.36  (+) ####################
    2020-10: $-1,053.37  (-) ####################
    2020-11: +$2,497.90  (+) ####################
    2020-12: +$1,903.51  (+) ####################
    2021-01: $-1,184.07  (-) ####################
    2021-02: +$200.79  (+) ####
    2021-03: +$4,462.65  (+) ####################
    2021-04: +$12,062.25  (+) ####################
    2021-05: +$1,502.53  (+) ####################
    2021-06: $-724.54  (-) ##############
    2021-07: +$740.75  (+) ##############
    2021-08: +$11,316.94  (+) ####################
    2021-09: +$9,190.85  (+) ####################
    2021-10: +$4,946.94  (+) ####################
    2021-11: +$10,254.81  (+) ####################
    2021-12: $-3,859.14  (-) ####################
    2022-01: $-2,012.40  (-) ####################
    2022-02: $-336.88  (-) ######
    2022-03: $-2,596.59  (-) ####################
    2022-04: $-1,335.58  (-) ####################
    2022-11: +$1,333.22  (+) ####################
    2022-12: +$451.96  (+) #########
    2023-01: +$987.81  (+) ###################
    2023-02: $-627.03  (-) ############
    2023-03: +$2,172.95  (+) ####################
    2023-04: $-7,248.52  (-) ####################
    2023-05: +$9,027.98  (+) ####################
    2023-06: +$9,758.41  (+) ####################
    2023-07: +$15,377.52  (+) ####################
    2023-08: $-19,149.47  (-) ####################
    2023-09: $-4,176.84  (-) ####################
    2023-10: $-71.15  (-) #
    2023-11: +$534.04  (+) ##########
    2023-12: +$6,001.06  (+) ####################
    2024-01: +$16,485.40  (+) ####################
    2024-02: +$24,921.57  (+) ####################
    2024-03: +$2,758.67  (+) ####################
    2024-04: +$8,167.80  (+) ####################
    2024-05: +$3,615.52  (+) ####################
    2024-06: +$59,312.64  (+) ####################
    2024-07: +$30,830.85  (+) ####################
    2024-08: +$1,207.44  (+) ####################
    2024-09: +$28,335.22  (+) ####################
    2024-10: +$22,851.51  (+) ####################
    2024-11: +$41,023.01  (+) ####################
    2024-12: $-12,918.92  (-) ####################
    2025-01: $-27,077.89  (-) ####################
    2025-02: $-2,274.14  (-) ####################
    2025-03: $-6,605.03  (-) ####################
    2025-05: +$14,939.13  (+) ####################
    2025-06: +$13,252.52  (+) ####################
    2025-07: +$26,883.32  (+) ####################
    2025-08: +$12,992.85  (+) ####################
    2025-09: +$76,202.63  (+) ####################
    2025-10: +$36,519.67  (+) ####################
    2025-11: +$669.38  (+) #############
    2025-12: +$46,597.46  (+) ####################
    2026-01: $-74,783.01  (-) ####################
"""

# ── 2. Set your starting capital ─────────────────────────────────────────────
STARTING_CAPITAL = 25_000.00   # adjust to match your actual initial capital


# ── 3. Parse ─────────────────────────────────────────────────────────────────
def parse_monthly_pnl(raw: str) -> list[tuple[str, float]]:
    """Returns list of (YYYY-MM, pnl_float) sorted chronologically."""
    pattern = re.compile(
        r"(\d{4}-\d{2})\s*:\s*[+]?\$?([-]?[\d,]+\.?\d*)"
    )
    results = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = pattern.search(line)
        if m:
            month = m.group(1)
            # normalize: remove commas, handle "$-" prefix
            raw_val = m.group(2).replace(",", "")
            pnl = float(raw_val)
            # if original line has "$-" at start (negative) but regex captured positive
            if "$-" in line and pnl > 0:
                pnl = -pnl
            results.append((month, pnl))
    results.sort(key=lambda x: x[0])
    return results


# ── 4. Optionally fetch benchmarks via yfinance ───────────────────────────────
def fetch_benchmark_returns(months: list[str]) -> dict[str, dict[str, float]]:
    """
    Returns {YYYY-MM: {"SPY": pct, "DIA": pct, "QQQ": pct}}
    Requires:  pip install yfinance
    """
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed. Run: pip install yfinance", file=sys.stderr)
        return {}

    tickers = {"SPY": "S&P 500", "DIA": "Dow", "QQQ": "Nasdaq"}
    benchmark_data: dict[str, dict[str, float]] = {}

    for ticker in tickers:
        first = months[0] + "-01"
        # end = one month past last entry
        last_dt = datetime.strptime(months[-1], "%Y-%m")
        if last_dt.month == 12:
            end_dt = last_dt.replace(year=last_dt.year + 1, month=1)
        else:
            end_dt = last_dt.replace(month=last_dt.month + 1)
        end = end_dt.strftime("%Y-%m-%d")

        df = yf.download(ticker, start=first, end=end, interval="1mo",
                         auto_adjust=True, progress=False)
        if df.empty:
            continue
        df["pct"] = df["Close"].pct_change() * 100
        for idx, row in df.iterrows():
            key = idx.strftime("%Y-%m")
            if key not in benchmark_data:
                benchmark_data[key] = {}
            if not (row["pct"] != row["pct"]):  # skip NaN
                benchmark_data[key][ticker] = round(float(row["pct"]), 2)

    return benchmark_data


# ── 5. Format helpers ─────────────────────────────────────────────────────────
def fmt_dollar(val: float) -> str:
    if val >= 0:
        return f"${val:,.2f}"
    return f"-${abs(val):,.2f}"

def fmt_pct(val: float | None) -> str:
    if val is None:
        return ""
    return f"{val:.2f}%"


# ── 6. Build TSV ──────────────────────────────────────────────────────────────
def build_tsv(monthly: list[tuple[str, float]],
              benchmarks: dict[str, dict[str, float]]) -> str:
    lines = []

    # Header row — monthly table + cumulative table side by side
    headers = [
        "Month", "BOT P&L", "BOT %",
        "S&P 500 %", "Dow %", "Nasdaq %",
        "",   # spacer column
        "Month", "Your Cumulative %",
        "S&P 500 Cumulative %", "Dow Cumulative %", "Nasdaq Cumulative %",
    ]
    lines.append("\t".join(headers))

    portfolio = STARTING_CAPITAL
    cum_spy = cum_dia = cum_qqq = 100.0   # index base = 100

    for month, pnl in monthly:
        # Monthly BOT %
        bot_pct = (pnl / portfolio) * 100
        portfolio += pnl
        cum_bot_pct = ((portfolio - STARTING_CAPITAL) / STARTING_CAPITAL) * 100

        # Benchmarks
        bm = benchmarks.get(month, {})
        spy_m  = bm.get("SPY")
        dia_m  = bm.get("DIA")
        qqq_m  = bm.get("QQQ")

        # Cumulative benchmark index
        if spy_m is not None:
            cum_spy *= (1 + spy_m / 100)
        if dia_m is not None:
            cum_dia *= (1 + dia_m / 100)
        if qqq_m is not None:
            cum_qqq *= (1 + qqq_m / 100)

        cum_spy_pct = cum_spy - 100 if spy_m is not None else None
        cum_dia_pct = cum_dia - 100 if dia_m is not None else None
        cum_qqq_pct = cum_qqq - 100 if qqq_m is not None else None

        row = [
            month,
            fmt_dollar(pnl),
            fmt_pct(bot_pct),
            fmt_pct(spy_m),
            fmt_pct(dia_m),
            fmt_pct(qqq_m),
            "",   # spacer
            month,
            fmt_pct(cum_bot_pct),
            fmt_pct(cum_spy_pct),
            fmt_pct(cum_dia_pct),
            fmt_pct(cum_qqq_pct),
        ]
        lines.append("\t".join(row))

    return "\n".join(lines)


# ── 7. Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", action="store_true",
                        help="Fetch S&P 500 / Dow / Nasdaq returns via yfinance")
    parser.add_argument("--out", default=None,
                        help="Output file path (default: print to stdout)")
    args = parser.parse_args()

    monthly = parse_monthly_pnl(RAW_OUTPUT)
    months  = [m for m, _ in monthly]

    benchmarks = fetch_benchmark_returns(months) if args.benchmarks else {}

    tsv = build_tsv(monthly, benchmarks)

    if args.out:
        with open(args.out, "w") as f:
            f.write(tsv)
        print(f"Saved to {args.out}")
    else:
        print(tsv)


if __name__ == "__main__":
    main()