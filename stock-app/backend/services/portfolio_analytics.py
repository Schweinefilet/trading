import pandas as pd
import numpy as np
from .data_fetcher import DataFetcher

class PortfolioAnalytics:
    def __init__(self, positions):
        """
        positions: List of dicts with {ticker, shares, avg_cost}
        """
        self.positions = positions
        self.fetcher = DataFetcher()

    def get_all_metrics(self):
        if not self.positions:
            return {
                'summary': {
                    'total_value': 0,
                    'total_gain_loss': 0,
                    'total_gain_loss_pct': 0,
                },
                'holdings': [],
                'risk': {},
                'correlation': {}
            }

        basic = self.get_basic_metrics()
        risk = self.get_risk_metrics(basic['holdings'])
        correlation = self.get_correlation_matrix()
        
        return {
            'summary': {
                'total_value': basic['total_value'],
                'total_gain_loss': basic['total_gain_loss'],
                'total_gain_loss_pct': basic['total_gain_loss_pct'],
            },
            'holdings': basic['holdings'],
            'risk': risk,
            'correlation': correlation
        }

    def get_basic_metrics(self):
        total_value = 0
        total_cost = 0
        holdings = []
        rf = 0.045
        
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
                hist = self.fetcher.get_history(ticker, period='1y', interval='1d')
                if hist:
                    hdf = pd.DataFrame(hist)
                    if not hdf.empty and 'Close' in hdf.columns:
                        series = pd.to_numeric(hdf['Close'], errors='coerce').pct_change().dropna()
                        if not series.empty:
                            h_mean = series.mean() * 252
                            h_vol = series.std() * np.sqrt(252)
                            holding_sharpe = (h_mean - rf) / h_vol if h_vol != 0 else None
            except Exception:
                holding_sharpe = None
            
            holdings.append({
                'ticker': ticker,
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

    def get_risk_metrics(self, holdings):
        if not holdings:
            return {}
            
        returns_df = pd.DataFrame()
        tickers = [h['ticker'] for h in holdings] + ['SPY']
        
        for t in tickers:
            hist = self.fetcher.get_history(t, period='1y', interval='1d')
            if hist:
                df = pd.DataFrame(hist)
                if not df.empty and 'Close' in df.columns:
                    df.set_index('Date', inplace=True)
                    returns_df[t] = df['Close'].pct_change()
                
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

    def get_correlation_matrix(self):
        tickers = [p['ticker'] for p in self.positions]
        if not tickers:
            return {}
        
        returns_df = pd.DataFrame()
        for t in tickers:
            hist = self.fetcher.get_history(t, period='1y', interval='1d')
            if hist:
                df = pd.DataFrame(hist)
                if not df.empty and 'Close' in df.columns:
                    df.set_index('Date', inplace=True)
                    returns_df[t] = df['Close'].pct_change()
        
        if returns_df.empty:
            return {}
            
        corr = returns_df.corr().fillna(0).to_dict()
        return corr
