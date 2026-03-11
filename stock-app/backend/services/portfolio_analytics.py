import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from .data_fetcher import DataFetcher

class PortfolioAnalytics:
    def __init__(self, positions):
        """
        positions: List of dicts with {ticker, shares, avg_cost}
        """
        self.positions = positions
        self.fetcher = DataFetcher()
        self._value_history_debug = None

    def get_all_metrics(self, timeframe='1y', interval='1d', total_cash=0.0, start_date=None, snapshots=None, daily_balance_history=None, daily_snapshot_history=None, activity_events=None, anchor_start_date=None, anchor_start_value=None, value_floor=None):
        self._value_history_debug = None
        if not self.positions:
            # No current positions (100% cash) — still build value history from
            # balance history / snapshots so the chart shows historical data correctly.
            value_history = self.get_value_history(
                holdings=[],
                timeframe=timeframe,
                interval=interval,
                total_cash=total_cash,
                pre_buy_cash=total_cash,
                start_date=start_date,
                snapshots=snapshots or {},
                daily_balance_history=daily_balance_history,
                daily_snapshot_history=daily_snapshot_history,
                activity_events=activity_events,
                anchor_start_date=anchor_start_date,
                anchor_start_value=anchor_start_value,
                value_floor=value_floor,
            )
            payload = {
                'summary': {
                    'total_value': total_cash,
                    'stocks_value': 0,
                    'total_gain_loss': 0,
                    'total_gain_loss_pct': 0,
                },
                'holdings': [],
                'value_history': value_history,
                'risk': {},
                'correlation': {},
                'has_data_gaps': any(p.get('value') is None for p in value_history),
            }
            if self._value_history_debug is not None:
                payload['value_history_debug'] = self._value_history_debug
            return payload

        tickers = [p['ticker'] for p in self.positions]

        # Risk/correlation always use daily data; use at least 1y for meaningful metrics.
        risk_period = timeframe if interval == '1d' else '1y'
        returns_map_daily = self._build_returns_map(tickers + ['SPY'], period=risk_period, interval='1d')

        basic = self.get_basic_metrics(returns_map=returns_map_daily)
        risk = self.get_risk_metrics(basic['holdings'], returns_map=returns_map_daily)
        correlation = self.get_correlation_matrix(returns_map=returns_map_daily)

        # Pre-buy flat line = cash held before any positions were opened.
        # That equals today's cash + the cost basis of all current positions
        # (i.e. the full amount the user had before deploying into stocks).
        total_cost = sum(h.get('avg_cost', 0) * h.get('shares', 0) for h in basic['holdings'])
        pre_buy_cash = total_cash + total_cost
        stocks_value = basic['total_value']

        value_history = self.get_value_history(
            basic['holdings'],
            timeframe=timeframe,
            interval=interval,
            total_cash=total_cash,
            pre_buy_cash=pre_buy_cash,
            start_date=start_date,
            snapshots=snapshots,
            daily_balance_history=daily_balance_history,
            daily_snapshot_history=daily_snapshot_history,
            activity_events=activity_events,
            anchor_start_date=anchor_start_date,
            anchor_start_value=anchor_start_value,
            value_floor=value_floor,
        )

        payload = {
            'summary': {
                'total_value': stocks_value + total_cash,
                'stocks_value': stocks_value,
                'total_gain_loss': basic['total_gain_loss'],
                'total_gain_loss_pct': basic['total_gain_loss_pct'],
            },
            'holdings': basic['holdings'],
            'value_history': value_history,
            'risk': risk,
            'correlation': correlation,
            'has_data_gaps': any(p.get('value') is None for p in value_history),
        }
        if self._value_history_debug is not None:
            payload['value_history_debug'] = self._value_history_debug
        return payload

    def _build_returns_map(self, tickers, period='1y', interval='1d'):
        returns_map = {}
        unique_tickers = list(dict.fromkeys([t for t in tickers if t]))

        for t in unique_tickers:
            hist = self.fetcher.get_history(t, period=period, interval=interval)
            if not hist:
                continue

            df = pd.DataFrame(hist)
            if df.empty or 'Close' not in df.columns:
                continue

            closes = pd.to_numeric(df['Close'], errors='coerce')
            rets = closes.pct_change()
            if 'Date' in df.columns:
                rets.index = df['Date']
            returns_map[t] = rets

        return returns_map

    def _parse_trade_dt(self, value):
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return None
            if candidate.endswith('Z'):
                candidate = f"{candidate[:-1]}+00:00"
            try:
                return datetime.fromisoformat(candidate)
            except ValueError:
                for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d'):
                    try:
                        return datetime.strptime(candidate, fmt)
                    except ValueError:
                        continue
        return None

    def _period_start(self, timeframe):
        days_by_period = {
            '1d': 1,
            '5d': 5,
            '7d': 7,
            '1mo': 31,
            '3mo': 92,
            '6mo': 183,
            '1y': 366,
            '2y': 731,
            '5y': 1827,
            '10y': 3653,
            'max': None,
        }
        key = str(timeframe or '').lower()
        days = days_by_period.get(key, 366)
        if days is None:
            return None
        return (pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=days - 1)).date()

    def _slice_series_by_timeframe(self, series, timeframe='1y'):
        if not series:
            return []
        start_filter = self._period_start(timeframe)
        if start_filter is None:
            return list(series)

        sliced = []
        for point in series:
            date_str = str(point.get('date', ''))[:10]
            try:
                dt = datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                continue
            if dt >= start_filter:
                sliced.append(point)
        return sliced

    def _daily_activity_cash_deltas(self, activity_events):
        """Build net cash delta per day from normalized activity events."""
        deltas = defaultdict(float)
        if not activity_events:
            return deltas

        for item in activity_events:
            activity_type = str(item.get('activity_type') or item.get('type') or '').upper()
            dt = self._parse_trade_dt(item.get('occurred_at') or item.get('trade_date'))
            if dt is None:
                continue
            day = dt.date().strftime('%Y-%m-%d')

            fee = float(item.get('fee') or 0.0)
            amount = item.get('amount')
            units = float(item.get('units') or 0.0)
            price = float(item.get('price') or 0.0)

            if activity_type == 'BUY':
                deltas[day] -= (units * price + fee)
            elif activity_type == 'SELL':
                deltas[day] += (units * price - fee)
            elif activity_type in {'DEPOSIT', 'WITHDRAWAL', 'DIVIDEND'}:
                if amount is not None:
                    deltas[day] += float(amount) - fee

        return deltas

    def _fetch_ticker_close_map(self, ticker):
        """Return {YYYY-MM-DD: close} for a ticker using daily history."""
        close_map = {}
        if not ticker:
            return close_map
        try:
            history = self.fetcher.get_history(ticker, period='max', interval='1d')
        except Exception:
            history = []
        if not history:
            return close_map

        for row in history:
            raw_date = row.get('Date')
            raw_close = row.get('Close')
            if raw_date is None:
                continue
            try:
                day = str(raw_date)[:10]
                dt = datetime.strptime(day, '%Y-%m-%d').date()
            except Exception:
                continue
            try:
                close = float(raw_close)
            except Exception:
                continue
            close_map[dt.strftime('%Y-%m-%d')] = close
        return close_map

    def _build_close_on_or_before_lookup(self, close_map):
        """Return a function day(date_obj)->latest close on or before that day."""
        if not close_map:
            return lambda _day: None

        ordered = []
        for day_str, close in close_map.items():
            try:
                day = datetime.strptime(day_str, '%Y-%m-%d').date()
            except Exception:
                continue
            ordered.append((day, float(close)))
        ordered.sort(key=lambda x: x[0])

        if not ordered:
            return lambda _day: None

        days = [d for d, _ in ordered]
        vals = [v for _, v in ordered]

        def lookup(target_day):
            lo = 0
            hi = len(days) - 1
            best_idx = -1
            while lo <= hi:
                mid = (lo + hi) // 2
                if days[mid] <= target_day:
                    best_idx = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best_idx < 0:
                return None
            return vals[best_idx]

        return lookup

    def _daily_total_value_deltas_from_activities(self, activity_events, anchor_start_date, end_date, include_debug=False):
        """
        Compute daily total-value deltas from activity logs using:
          day_delta = net_cashflow(day) + (end_positions_value - start_positions_value)

        This captures overnight holds correctly:
        - If position carries overnight, next day P&L uses close_today - close_yesterday.
        - If carried shares are sold next day, that day effect is sell_price - close_yesterday.

        Assumption: anchor value reflects end-of-day state at anchor date.
        Therefore positions at anchor baseline are treated as zero for forward inference.
        """
        deltas = defaultdict(float)
        debug_rows = []
        if anchor_start_date is None or end_date is None or end_date <= anchor_start_date:
            return (deltas, debug_rows) if include_debug else deltas

        # Group relevant events by calendar day (strictly after anchor day).
        events_by_day = defaultdict(list)
        trade_tickers = set()
        for item in (activity_events or []):
            activity_type = str(item.get('activity_type') or item.get('type') or '').upper()
            dt = self._parse_trade_dt(item.get('occurred_at') or item.get('trade_date'))
            if dt is None:
                continue
            day = dt.date()
            if day <= anchor_start_date or day > end_date:
                continue

            ticker = str(item.get('ticker') or '').upper().strip() or None
            if activity_type in {'BUY', 'SELL'} and ticker:
                trade_tickers.add(ticker)

            events_by_day[day].append((dt, item))

        # Prepare EOD close lookup per ticker.
        close_lookup_by_ticker = {}
        for ticker in trade_tickers:
            close_map = self._fetch_ticker_close_map(ticker)
            close_lookup_by_ticker[ticker] = self._build_close_on_or_before_lookup(close_map)

        # Simulate forward from anchor+1 with baseline zero positions at anchor EOD.
        positions = defaultdict(float)
        day = anchor_start_date + timedelta(days=1)
        while day <= end_date:
            day_cashflow = 0.0

            # Start-of-day positions valued at previous day's close.
            prev_day = day - timedelta(days=1)
            start_positions_value = 0.0
            for ticker, qty in positions.items():
                if not qty:
                    continue
                lookup = close_lookup_by_ticker.get(ticker)
                if not lookup:
                    continue
                prev_close = lookup(prev_day)
                if prev_close is None:
                    continue
                start_positions_value += float(qty) * float(prev_close)

            # Apply all events of the day in chronological order.
            day_events = sorted(events_by_day.get(day, []), key=lambda pair: pair[0])
            for _, item in day_events:
                activity_type = str(item.get('activity_type') or item.get('type') or '').upper()
                fee = float(item.get('fee') or 0.0)
                amount = item.get('amount')
                units = float(item.get('units') or 0.0)
                price = float(item.get('price') or 0.0)
                ticker = str(item.get('ticker') or '').upper().strip() or None

                if activity_type == 'BUY' and ticker and units > 0 and price >= 0:
                    day_cashflow -= (units * price + fee)
                    positions[ticker] += units
                elif activity_type == 'SELL' and ticker and units > 0 and price >= 0:
                    day_cashflow += (units * price - fee)
                    positions[ticker] -= units
                elif activity_type in {'DEPOSIT', 'WITHDRAWAL', 'DIVIDEND'}:
                    if amount is not None:
                        day_cashflow += float(amount) - fee

            # End-of-day positions valued at current day's close.
            end_positions_value = 0.0
            for ticker, qty in positions.items():
                if not qty:
                    continue
                lookup = close_lookup_by_ticker.get(ticker)
                if not lookup:
                    continue
                close_today = lookup(day)
                if close_today is None:
                    continue
                end_positions_value += float(qty) * float(close_today)

            market_value_change = float(end_positions_value - start_positions_value)
            net_delta = float(day_cashflow + market_value_change)
            day_str = day.strftime('%Y-%m-%d')
            deltas[day_str] = net_delta
            if include_debug:
                debug_rows.append({
                    'date': day_str,
                    'cashflow_delta': float(day_cashflow),
                    'start_positions_value': float(start_positions_value),
                    'end_positions_value': float(end_positions_value),
                    'market_value_change': market_value_change,
                    'net_delta': net_delta,
                })
            day += timedelta(days=1)

        return (deltas, debug_rows) if include_debug else deltas

    def _infer_daily_series_from_anchors(self, series_days, known_values, daily_deltas):
        """
        Infer daily values from sparse anchors and per-day cash deltas.
        EOD(day) = EOD(previous_day) + delta(day)
        """
        values = dict(known_values)
        day_strs = [d.strftime('%Y-%m-%d') for d in series_days]

        # Forward pass from earliest known anchors
        for i in range(1, len(day_strs)):
            prev_day = day_strs[i - 1]
            day = day_strs[i]
            if day in values:
                continue
            if prev_day in values:
                values[day] = float(values[prev_day]) + float(daily_deltas.get(day, 0.0))

        # Backward pass from latest known anchors
        for i in range(len(day_strs) - 2, -1, -1):
            day = day_strs[i]
            next_day = day_strs[i + 1]
            if day in values:
                continue
            if next_day in values:
                values[day] = float(values[next_day]) - float(daily_deltas.get(next_day, 0.0))

        return values

    def get_value_history(self, holdings, returns_map=None, timeframe='1y', interval='1d',
                          total_cash=0.0, pre_buy_cash=None, start_date=None, snapshots=None,
                          daily_balance_history=None, daily_snapshot_history=None,
                          activity_events=None, anchor_start_date=None, anchor_start_value=None,
                          value_floor=None):
        """
        Build portfolio value history series.

        For intraday intervals (1m): Use stored snapshots if available. Fallback to calculated values.
        For daily+ intervals: Use calculated values, with snapshots as fallback for historical accuracy.

        - Dates before start_date: flat line at pre_buy_cash (cash held before any position
          was opened = remaining_cash + cost_basis_of_all_positions).
        - Dates on/after start_date: computed from weighted returns, anchored to current value.
        - Where a stored snapshot exists, it may override the computed value for
          historical accuracy (survives portfolio composition changes, e.g. selling/rebuying).

        snapshots: dict of {timestamp_str: total_value} from PortfolioSnapshot DB rows.
        pre_buy_cash: value to show as flat line before start_date. Defaults to total_cash.
        """
        snapshots = snapshots or {}
        start_str = start_date.strftime('%Y-%m-%d') if start_date else None
        flat_line_value = pre_buy_cash if pre_buy_cash is not None else total_cash
        intraday = interval not in ('1d', '1wk', '1mo')

        # For intraday intervals, prioritize stored minute snapshots as-is.
        if intraday and snapshots:
            series = []
            for timestamp in sorted(snapshots.keys()):
                series.append({'date': timestamp, 'value': float(snapshots[timestamp])})
            return series

        # Daily history (no trade replay; no returns-model fallback):
        # 1) SnapTrade daily balance history (primary)
        # 2) Daily EOD snapshot history (fallback)
        balance_map = {
            str(p.get('date', ''))[:10]: float(p.get('value'))
            for p in (daily_balance_history or [])
            if str(p.get('date', ''))[:10]
        }
        snapshot_map = {
            str(p.get('date', ''))[:10]: float(p.get('value'))
            for p in (daily_snapshot_history or [])
            if str(p.get('date', ''))[:10]
        }

        if not balance_map and not snapshot_map:
            # If user provided an anchor, we can still compute from activity cashflows.
            if anchor_start_date is None or anchor_start_value is None:
                return []

        # Optional anchor-driven mode: compute from a user-provided start value.
        if anchor_start_date is not None and anchor_start_value is not None:
            start_filter = self._period_start(timeframe)
            today = pd.Timestamp.utcnow().date()
            start_day = anchor_start_date
            if start_filter and start_day < start_filter:
                start_day = start_filter
            if start_day > today:
                return []

            daily_deltas, debug_rows = self._daily_total_value_deltas_from_activities(
                activity_events=activity_events,
                anchor_start_date=anchor_start_date,
                end_date=today,
                include_debug=True,
            )
            series_days = pd.date_range(start=start_day, end=today, freq='D').date

            self._value_history_debug = {
                'mode': 'anchor_with_overnight_hold_logic',
                'anchor_day_locked': True,
                'anchor_date': anchor_start_date.strftime('%Y-%m-%d'),
                'anchor_value': float(anchor_start_value),
                'timeframe': str(timeframe),
                'daily_delta_breakdown': debug_rows,
            }

            running_value = float(anchor_start_value)
            floor = float(value_floor) if value_floor is not None else None
            series = []
            for day in series_days:
                day_str = day.strftime('%Y-%m-%d')

                # Apply delta for the specific day only when it is strictly after anchor day.
                if day > anchor_start_date:
                    running_value += float(daily_deltas.get(day_str, 0.0))

                # Anchor day is immutable by user intent.
                if day == anchor_start_date:
                    value = float(anchor_start_value)
                    running_value = float(anchor_start_value)
                    is_known = False
                # Known daily account values (real data), if present, take precedence.
                elif day_str in balance_map:
                    value = balance_map[day_str]
                    running_value = float(value)
                    is_known = True
                elif day_str in snapshot_map:
                    value = snapshot_map[day_str]
                    running_value = float(value)
                    is_known = True
                else:
                    value = running_value
                    is_known = False

                # Apply floor to ALL values (real or inferred). Real balance snapshots can
                # be incomplete when an account reports $0 due to a sync gap — the floor
                # prevents those bad reads from dragging the chart below a known minimum.
                if floor is not None:
                    value = max(float(value), floor)
                    running_value = max(running_value, floor)

                series.append({
                    'date': day_str,
                    'value': float(value),
                    'is_inferred': not is_known,
                })
            return series

        known_days = sorted(set(balance_map.keys()) | set(snapshot_map.keys()))
        start_filter = self._period_start(timeframe)

        series = []
        for day_str in known_days:
            try:
                day = datetime.strptime(day_str, '%Y-%m-%d').date()
            except ValueError:
                continue
            if start_filter and day < start_filter:
                continue

            if day_str in balance_map:
                value = balance_map[day_str]
            else:
                value = snapshot_map.get(day_str)
            series.append({'date': day_str, 'value': value})

        return series

    def get_basic_metrics(self, returns_map=None):
        total_value = 0
        total_cost = 0
        holdings = []
        rf = 0.045
        returns_map = returns_map or {}
        
        for pos in self.positions:
            ticker = pos['ticker']
            quote = self.fetcher.get_quote(ticker)
            if not quote:
                continue
            
            curr_price = quote['current_price']
            value = curr_price * pos['shares']
            cost = pos['avg_cost'] * pos['shares']
            gain_loss = value - cost
            gain_loss_pct = (gain_loss / cost * 100) if cost > 0 else 0
            sector = quote.get('sector') if quote else None
            if not sector:
                try:
                    f = self.fetcher.get_fundamentals(ticker)
                    sector = f.get('sector') if f else None
                except Exception:
                    sector = None

            # Holding-level Sharpe (1Y daily), used for allocation bucketing.
            holding_sharpe = None
            try:
                series = returns_map.get(ticker)
                if series is not None:
                    series = pd.to_numeric(series, errors='coerce').dropna()
                    if not series.empty:
                        h_mean = series.mean() * 252
                        h_vol = series.std() * np.sqrt(252)
                        holding_sharpe = (h_mean - rf) / h_vol if h_vol != 0 else None
            except Exception:
                holding_sharpe = None
            
            holdings.append({
                'ticker': ticker,
                'long_name': quote.get('longName') if quote else ticker,
                'shares': pos['shares'],
                'avg_cost': pos['avg_cost'],
                'current_price': curr_price,
                'value': value,
                'gain_loss': gain_loss,
                'gain_loss_pct': gain_loss_pct,
                'sector': sector,
                'holding_sharpe': float(holding_sharpe) if holding_sharpe is not None and pd.notna(holding_sharpe) else None,
            })
            total_value += value
            total_cost += cost
            
        total_gain_loss = total_value - total_cost
        total_gain_loss_pct = (total_gain_loss / total_cost * 100) if total_cost > 0 else 0
        
        # Calculate weights
        for h in holdings:
            h['weight'] = (h['value'] / total_value) if total_value > 0 else 0
            
        return {
            'total_value': total_value,
            'total_gain_loss': total_gain_loss,
            'total_gain_loss_pct': total_gain_loss_pct,
            'holdings': holdings
        }

    def get_risk_metrics(self, holdings, returns_map=None):
        if not holdings:
            return {}
        returns_map = returns_map or self._build_returns_map([h['ticker'] for h in holdings] + ['SPY'])
        returns_df = pd.DataFrame({
            t: pd.to_numeric(s, errors='coerce')
            for t, s in returns_map.items()
            if t in [h['ticker'] for h in holdings] + ['SPY']
        })
                
        returns_df.dropna(inplace=True)
        if returns_df.empty:
            return {}
            
        # Portfolio returns (weighted)
        weights = {h['ticker']: h['weight'] for h in holdings}
        total_value = float(sum(h.get('value', 0) for h in holdings))
        
        port_returns = pd.Series(0, index=returns_df.index)
        for t in weights:
            if t in returns_df:
                port_returns += returns_df[t] * weights[t]
        
        # Beta
        beta = 1.0
        if 'SPY' in returns_df:
            market_returns = returns_df['SPY']
            cov = port_returns.cov(market_returns)
            market_var = market_returns.var()
            beta = cov / market_var if market_var != 0 else 1.0
            
        # Volatility
        vol = port_returns.std() * np.sqrt(252)
        
        # Sharpe Ratio
        rf = 0.045
        mean_return = port_returns.mean() * 252
        sharpe = (mean_return - rf) / vol if vol != 0 else 0
        
        # Max Drawdown
        cum_returns = (1 + port_returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Current drawdown (latest value)
        current_drawdown = drawdown.iloc[-1] if not drawdown.empty else 0

        # Sortino ratio (downside deviation only)
        downside = port_returns[port_returns < 0]
        downside_dev = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
        sortino = (mean_return - rf) / downside_dev if downside_dev != 0 else 0

        # Calmar ratio (annual return per unit of max drawdown)
        calmar = mean_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Daily VaR/CVaR at 95% confidence (loss-focused, in dollars)
        var_95_return = float(port_returns.quantile(0.05))
        cvar_slice = port_returns[port_returns <= var_95_return]
        cvar_95_return = float(cvar_slice.mean()) if not cvar_slice.empty else var_95_return
        var_95_dollar = abs(var_95_return) * total_value
        cvar_95_dollar = abs(cvar_95_return) * total_value

        # Skewness of return distribution
        skewness = port_returns.skew()

        # Win rate from live holdings
        win_count = sum(1 for h in holdings if h.get('gain_loss', 0) > 0)
        win_rate = (win_count / len(holdings)) if holdings else 0

        # Ulcer index: penalizes prolonged drawdowns
        ulcer_index = np.sqrt((drawdown.pow(2)).mean()) if not drawdown.empty else 0

        # Effective number of independent bets: 1 / (w' * Corr * w)
        effective_n = 0
        asset_tickers = [h['ticker'] for h in holdings if h['ticker'] in returns_df.columns]
        if asset_tickers:
            w = np.array([weights[t] for t in asset_tickers], dtype=float)
            corr = returns_df[asset_tickers].corr().fillna(0).to_numpy()
            try:
                denom = float(w.T @ corr @ w)
                if denom > 0:
                    effective_n = 1.0 / denom
            except Exception:
                effective_n = 0

        # Market-relative metrics (requires SPY)
        alpha = 0
        up_capture = 0
        down_capture = 0
        if 'SPY' in returns_df:
            market_returns = returns_df['SPY']
            market_annual_return = market_returns.mean() * 252
            alpha = mean_return - (rf + beta * (market_annual_return - rf))

            up_mask = market_returns > 0
            down_mask = market_returns < 0

            market_up = market_returns[up_mask].mean()
            market_down = market_returns[down_mask].mean()
            port_up = port_returns[up_mask].mean() if up_mask.any() else 0
            port_down = port_returns[down_mask].mean() if down_mask.any() else 0

            valid_up = pd.notna(market_up) and abs(float(market_up)) > 1e-12
            valid_down = pd.notna(market_down) and abs(float(market_down)) > 1e-12
            up_capture = (port_up / market_up) if valid_up else 0
            down_capture = (port_down / market_down) if valid_down else 0
        
        return {
            'beta': float(beta),
            'volatility': float(vol),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'sortino_ratio': float(sortino),
            'calmar_ratio': float(calmar),
            'var_95_dollar': float(var_95_dollar),
            'cvar_95_dollar': float(cvar_95_dollar),
            'current_drawdown': float(current_drawdown),
            'skewness': float(skewness) if pd.notna(skewness) else 0,
            'effective_n': float(effective_n),
            'jensen_alpha': float(alpha),
            'up_capture': float(up_capture),
            'down_capture': float(down_capture),
            'win_rate': float(win_rate),
            'ulcer_index': float(ulcer_index),
        }

    def get_correlation_matrix(self, returns_map=None):
        tickers = [p['ticker'] for p in self.positions]
        if not tickers:
            return {}

        returns_map = returns_map or self._build_returns_map(tickers)
        returns_df = pd.DataFrame({
            t: pd.to_numeric(s, errors='coerce')
            for t, s in returns_map.items()
            if t in tickers
        })
        
        if returns_df.empty:
            return {}
            
        corr = returns_df.corr().fillna(0).to_dict()
        return corr
