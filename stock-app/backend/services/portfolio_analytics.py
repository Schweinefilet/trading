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
            
            holdings.append({
                'ticker': ticker,
                'shares': pos['shares'],
                'avg_cost': pos['avg_cost'],
                'current_price': curr_price,
                'value': value,
                'gain_loss': gain_loss,
                'gain_loss_pct': gain_loss_pct
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
        
        return {
            'beta': float(beta),
            'volatility': float(vol),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown)
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
