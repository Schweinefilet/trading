import yfinance as yf
import finnhub
import os
import pandas as pd
from .cache import CacheService
from dotenv import load_dotenv

load_dotenv()

class DataFetcher:
    def __init__(self):
        api_key = os.getenv("FINNHUB_API_KEY")
        self.finnhub_client = finnhub.Client(api_key=api_key) if api_key else None

    def get_quote(self, ticker):
        data = CacheService.get(ticker, 'quote')
        if data:
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
            CacheService.set(ticker, 'quote', res)
            return res
        except Exception as e:
            print(f"yfinance fallback error for {ticker}: {e}")
            return None

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

    def get_analyst(self, ticker):
        data = CacheService.get(ticker, 'analyst')
        if data:
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
                # Convert DatetimeIndex column to string
                date_col = upgrades.columns[0]
                upgrades[date_col] = upgrades[date_col].astype(str)
                upgrades = upgrades.head(50)
                result['upgrades_downgrades'] = upgrades[['GradeDate', 'Firm', 'ToGrade', 'FromGrade', 'Action']].to_dict(orient='records')
            else:
                result['upgrades_downgrades'] = []
        except Exception as e:
            print(f"Analyst upgrades/downgrades error for {ticker}: {e}")
            result['upgrades_downgrades'] = []

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
