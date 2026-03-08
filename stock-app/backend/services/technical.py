import pandas as pd
import pandas_ta_classic as ta
import numpy as np

class TechnicalService:
    @staticmethod
    def compute_indicators(df, requested_indicators=None):
        """
        Computes requested technical indicators on a DataFrame.
        """
        if df.empty:
            return df

        if not requested_indicators:
            requested_indicators = ['sma_20', 'sma_50', 'sma_200', 'rsi_14', 'macd']

        # Ensure numeric columns for TA
        mask = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in mask:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # TREND
        if 'sma' in requested_indicators:
            for length in [9, 20, 50, 100, 200]:
                df.ta.sma(length=length, append=True)
        else:
            for ind in requested_indicators:
                if ind.startswith('sma_'):
                    try:
                        length = int(ind[4:])
                        df.ta.sma(length=length, append=True)
                    except ValueError:
                        pass

        if 'ema' in requested_indicators:
            for length in [9, 20, 50, 100, 200]:
                df.ta.ema(length=length, append=True)
        else:
            for ind in requested_indicators:
                if ind.startswith('ema_'):
                    try:
                        length = int(ind[4:])
                        df.ta.ema(length=length, append=True)
                    except ValueError:
                        pass

        if 'bbands' in requested_indicators or 'bb' in requested_indicators: 
            df.ta.bbands(append=True)
        
        if 'ichimoku' in requested_indicators: 
            try:
                ichimoku_df, _ = df.ta.ichimoku()
                df = pd.concat([df, ichimoku_df], axis=1)
            except: pass
            
        if 'psar' in requested_indicators: df.ta.psar(append=True)
        if 'supertrend' in requested_indicators: df.ta.supertrend(append=True)

        # MOMENTUM
        if 'rsi' in requested_indicators: df.ta.rsi(length=14, append=True)
        if 'macd' in requested_indicators: df.ta.macd(append=True)
        if 'stoch' in requested_indicators: df.ta.stoch(append=True)
        if 'cci' in requested_indicators: df.ta.cci(append=True)

        # VOLATILITY
        if 'atr' in requested_indicators: df.ta.atr(append=True)
        if 'stdev' in requested_indicators: df.ta.stdev(append=True)

        # VOLUME
        if 'obv' in requested_indicators: df.ta.obv(append=True)
        if 'vwap' in requested_indicators: df.ta.vwap(append=True)
        if 'cmf' in requested_indicators: df.ta.cmf(append=True)

        # Replace NaN with None for JSON
        df = df.replace(np.nan, None)
        return df

    @staticmethod
    def compute_indicators_from_list(data_list, requested_indicators=None):
        if not data_list:
            return []
        df = pd.DataFrame(data_list)
        
        # Ensure 'Date' exists and is cleaned
        # yfinance columns might be capitalized or not
        if 'Date' not in df.columns and 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'Date'}, inplace=True)
        
        if 'Date' in df.columns:
            # Drop timezone if exists to avoid serialization issues
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        df = TechnicalService.compute_indicators(df, requested_indicators)
        
        # Format Date back to string for JSON serialization
        # Use ISO format for better compatibility
        if 'Date' in df.columns:
            df['Date'] = df['Date'].apply(lambda x: x.isoformat() if hasattr(x, 'isoformat') else str(x))
            
        return df.to_dict(orient='records')
