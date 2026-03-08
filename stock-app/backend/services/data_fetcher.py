import yfinance as yf
import finnhub
import os
import pandas as pd
from .cache import CacheService
from dotenv import load_dotenv

load_dotenv()

ANALYST_TARGETS_VERSION = 2

SYMBOL_OVERRIDES = {
    '^GSPC': {'longName': 'S&P 500 Index', 'sector': 'Index', 'industry': 'Equity Index'},
    '^IXIC': {'longName': 'NASDAQ Composite', 'sector': 'Index', 'industry': 'Equity Index'},
    '^DJI': {'longName': 'Dow Jones Industrial Average', 'sector': 'Index', 'industry': 'Equity Index'},
    '^RUT': {'longName': 'Russell 2000 Index', 'sector': 'Index', 'industry': 'Equity Index'},
    '^VIX': {'longName': 'CBOE Volatility Index', 'sector': 'Index', 'industry': 'Volatility Index'},
    '^TNX': {'longName': 'CBOE Interest Rate 10 Year T Note', 'sector': 'Rates', 'industry': 'Treasury Yield Index'},
    'GC=F': {'longName': 'Gold Futures', 'sector': 'Commodities', 'industry': 'Precious Metals'},
    'SI=F': {'longName': 'Silver Futures', 'sector': 'Commodities', 'industry': 'Precious Metals'},
    'CL=F': {'longName': 'Crude Oil WTI Futures', 'sector': 'Commodities', 'industry': 'Energy'},
    'DX=F': {'longName': 'U.S. Dollar Index Futures', 'sector': 'Currencies', 'industry': 'Dollar Index'},
    'ES=F': {'longName': 'E-mini S&P 500 Futures', 'sector': 'Futures', 'industry': 'Equity Index Futures'},
    'NQ=F': {'longName': 'E-mini Nasdaq-100 Futures', 'sector': 'Futures', 'industry': 'Equity Index Futures'},
    'YM=F': {'longName': 'E-mini Dow Jones Futures', 'sector': 'Futures', 'industry': 'Equity Index Futures'},
}

class DataFetcher:
    def __init__(self):
        api_key = os.getenv("FINNHUB_API_KEY")
        self.finnhub_client = finnhub.Client(api_key=api_key) if api_key else None

    def get_quote(self, ticker):
        data = CacheService.get(ticker, 'quote')
        if data:
            # Backfill metadata for older cached entries that may not include names/sectors.
            needs_meta = not data.get('longName') or not data.get('sector')
            if needs_meta:
                data = self._enrich_quote_metadata(ticker, data)
                CacheService.set(ticker, 'quote', data)
            return data
            
        res = None
        if self.finnhub_client:
            try:
                quote = self.finnhub_client.quote(ticker)
                if quote and quote.get('c'):
                    res = {
                        'ticker': ticker,
                        'current_price': quote['c'],
                        'change': quote['d'],
                        'percent_change': quote['dp'],
                        'high': quote['h'],
                        'low': quote['l'],
                        'open': quote['o'],
                        'prev_close': quote['pc'],
                        'timestamp': quote['t']
                    }
                    res = self._enrich_quote_metadata(ticker, res)
                    CacheService.set(ticker, 'quote', res)
                    return res
            except Exception as e:
                print(f"Finnhub error for {ticker}: {e}")
            
        # Fallback to yfinance
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            res = {
                'ticker': ticker,
                'current_price': info.get('currentPrice', info.get('regularMarketPrice')),
                'change': info.get('regularMarketChange'),
                'percent_change': info.get('regularMarketChangePercent'),
                'high': info.get('dayHigh'),
                'low': info.get('dayLow'),
                'open': info.get('regularMarketOpen'),
                'prev_close': info.get('previousClose'),
                'timestamp': info.get('regularMarketTime'),
                'longName': info.get('longName', ticker),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
            }
            res = self._enrich_quote_metadata(ticker, res)
            CacheService.set(ticker, 'quote', res)
            return res
        except Exception as e:
            print(f"yfinance fallback error for {ticker}: {e}")
            return None

    def _enrich_quote_metadata(self, ticker, quote):
        res = dict(quote or {})

        # Try fundamentals cache/API for company metadata if missing.
        if not res.get('longName') or not res.get('sector') or not res.get('industry'):
            try:
                f = self.get_fundamentals(ticker)
                if f:
                    res['longName'] = res.get('longName') or f.get('longName') or ticker
                    res['sector'] = res.get('sector') or f.get('sector')
                    res['industry'] = res.get('industry') or f.get('industry')
            except Exception:
                pass

        # Apply authoritative symbol-name overrides for indexes/futures.
        override = SYMBOL_OVERRIDES.get(ticker)
        if override:
            res['longName'] = override.get('longName') or res.get('longName') or ticker
            res['sector'] = override.get('sector') or res.get('sector')
            res['industry'] = override.get('industry') or res.get('industry')
        elif not res.get('longName'):
            res['longName'] = ticker

        return res

    def get_history(self, ticker, period='1y', interval='1d'):
        key_params = f"period={period}&interval={interval}"
        data = CacheService.get(ticker, 'history', key_params)
        if data:
            return data
            
        try:
            stock = yf.Ticker(ticker)
            history = stock.history(period=period, interval=interval)
            if history.empty:
                return None
            # Convert DF to list of dicts for JSON storage
            res = history.reset_index()
            
            # yfinance returns 'Date' for daily+, and 'Datetime' for intraday
            if 'Datetime' in res.columns:
                res = res.rename(columns={'Datetime': 'Date'})
            
            # Format time appropriately based on interval
            if interval in ['1d', '1wk', '1mo']:
                res['Date'] = res['Date'].dt.strftime('%Y-%m-%d')
            else:
                res['Date'] = res['Date'].dt.strftime('%m-%d %H:%M')
                
            data_list = res.to_dict(orient='records')
            CacheService.set(ticker, 'history', data_list, key_params)
            return data_list
        except Exception as e:
            print(f"History error for {ticker}: {e}")
            return None

    def get_fundamentals(self, ticker):
        data = CacheService.get(ticker, 'fundamentals')
        if data:
            return data
            
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract common metrics
            metrics = {
                'ticker': ticker,
                'longName': info.get('longName'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'summary': info.get('longBusinessSummary'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'pb_ratio': info.get('priceToBook'),
                'ps_ratio': info.get('priceToSalesTrailing12Months'),
                'ev_ebitda': info.get('enterpriseToEbitda'),
                'peg_ratio': info.get('pegRatio'),
                'revenue_ttm': info.get('totalRevenue'),
                'net_income_ttm': info.get('netIncomeToCommon'),
                'eps_ttm': info.get('trailingEps'),
                'eps_forward': info.get('forwardEps'),
                'profit_margin': info.get('profitMargins'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                'free_cash_flow': info.get('freeCashflow'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'market_cap': info.get('marketCap'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
            }
            CacheService.set(ticker, 'fundamentals', metrics)
            return metrics
        except Exception as e:
            print(f"Fundamentals error for {ticker}: {e}")
            return None

    def get_analyst(self, ticker, force_refresh=False):
        data = None if force_refresh else CacheService.get(ticker, 'analyst')
        if data and data.get('targets_extraction_version') == ANALYST_TARGETS_VERSION:
            return data

        try:
            stock = yf.Ticker(ticker)
        except Exception as e:
            print(f"Analyst yfinance init error for {ticker}: {e}")
            return None

        result = {}

        try:
            targets = stock.analyst_price_targets
            if targets:
                result['price_targets'] = {
                    'low': targets.get('low'),
                    'mean': targets.get('mean'),
                    'high': targets.get('high'),
                    'current': targets.get('current'),
                    'numberOfAnalysts': targets.get('numberOfAnalysts'),
                }
            else:
                result['price_targets'] = {}
        except Exception as e:
            print(f"Analyst price targets error for {ticker}: {e}")
            result['price_targets'] = {}

        try:
            recs = stock.recommendations_summary
            if recs is not None and not recs.empty:
                recs_list = recs.head(4).to_dict(orient='records')
                result['ratings'] = recs_list
            else:
                result['ratings'] = []
        except Exception as e:
            print(f"Analyst recommendations error for {ticker}: {e}")
            result['ratings'] = []

        try:
            upgrades = stock.upgrades_downgrades
            if upgrades is not None and not upgrades.empty:
                upgrades = upgrades.reset_index()
                date_col = upgrades.columns[0]

                # Normalize ordering: newest first so "most recent" logic is deterministic.
                upgrades['_grade_ts'] = pd.to_datetime(upgrades[date_col], errors='coerce')
                upgrades = upgrades.sort_values('_grade_ts', ascending=False, na_position='last').reset_index(drop=True)

                # Keep only the same latest 50 actions we render in the table,
                # then dedupe by firm so latest revision wins.
                recent_50_actions = upgrades.head(50).copy()

                result['individual_targets'] = []
                if 'currentPriceTarget' in recent_50_actions.columns:
                    target_rows = recent_50_actions[
                        recent_50_actions['currentPriceTarget'].notna() &
                        (recent_50_actions['currentPriceTarget'] > 0)
                    ].copy()

                    if 'Firm' in target_rows.columns:
                        # latest revision wins, no duplicate firm targets
                        latest_per_firm = target_rows.drop_duplicates(subset=['Firm'], keep='first')
                    else:
                        latest_per_firm = target_rows

                    result['individual_targets'] = [
                        float(v) for v in latest_per_firm['currentPriceTarget'].tolist()
                        if pd.notna(v) and v > 0
                    ]

                # Keep recent upgrades table payload for UI (last 50 actions overall)
                upgrades = recent_50_actions
                upgrades[date_col] = upgrades[date_col].astype(str)
                # Include price target columns if available
                base_cols = ['GradeDate', 'Firm', 'ToGrade', 'FromGrade', 'Action']
                extra_cols = [c for c in ['priceTargetAction', 'currentPriceTarget', 'priorPriceTarget']
                              if c in upgrades.columns]
                # Replace NaN with None for safe JSON serialization
                subset = upgrades[base_cols + extra_cols].where(
                    upgrades[base_cols + extra_cols].notna(), other=None
                )
                result['upgrades_downgrades'] = subset.to_dict(orient='records')
            else:
                result['upgrades_downgrades'] = []
                result['individual_targets'] = []
        except Exception as e:
            print(f"Analyst upgrades/downgrades error for {ticker}: {e}")
            result['upgrades_downgrades'] = []
            result['individual_targets'] = []

        result['targets_extraction_version'] = ANALYST_TARGETS_VERSION

        if result:
            CacheService.set(ticker, 'analyst', result)
        return result

    def get_financials(self, ticker):
        data = CacheService.get(ticker, 'financials')
        if data:
            return data
            
        try:
            stock = yf.Ticker(ticker)
            
            # Last 4 quarters
            income = stock.quarterly_income_stmt.head(4).to_dict()
            balance = stock.quarterly_balance_sheet.head(4).to_dict()
            cashflow = stock.quarterly_cashflow.head(4).to_dict()
            
            res = {
                'income_statement': income,
                'balance_sheet': balance,
                'cash_flow': cashflow
            }
            CacheService.set(ticker, 'financials', res)
            return res
        except Exception as e:
            print(f"Financials error for {ticker}: {e}")
            return None
